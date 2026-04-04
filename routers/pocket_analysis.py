"""
Pocket Analysis Router
Tools: Fpocket, P2Rank
"""
# --- SYNC_NOTES (auto-generated from CLAUDE.md, do not edit) ---
# FPOCKET notes (from CLAUDE.md, do not manually edit):
#   - Legacy workflow (11 steps): P2Rank top pocket → take top 6 residues → DiffDock blind docking cross-validation → RFdiffusion
#   - 10. AF3 validation — ipTM >= 0.6
# --- /SYNC_NOTES ---

import os, json, re
from pathlib import Path
from fastapi import APIRouter
from schemas.models import FpocketRequest, P2RankRequest, PocketResult, TaskRef
from core.task_manager import task_manager
from core.docker_client import run_in_container_streaming, run_in_container
from core.config import settings

router = APIRouter()


def _parse_fpocket_output(output_dir: str) -> list:
    """Parse fpocket output: stats from *_info.txt + centers from pocket*_atm.pdb"""
    import glob
    import numpy as np
    pockets = []
    info_files = glob.glob(f"{output_dir}/**/*_info.txt", recursive=True)
    if not info_files:
        return pockets
    info_file = info_files[0]
    current = {}
    with open(info_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Pocket ") and line.endswith(":"):
                if current:
                    pockets.append(current)
                pocket_num = line.split()[1]
                current = {"pocket_id": int(pocket_num)}
            elif ":" in line and current is not None:
                k, v = line.split(":", 1)
                k = k.strip().lower().replace(" ", "_").replace(".", "").replace("/", "_")
                try:
                    current[k] = float(v.strip())
                except ValueError:
                    current[k] = v.strip()
    if current:
        pockets.append(current)

    # Enrich each pocket with center coordinates from pocket*_atm.pdb
    pockets_dir = os.path.join(output_dir, "pockets")
    for pocket in pockets:
        pid = pocket.get("pocket_id", 0)
        atm_pdb = os.path.join(pockets_dir, f"pocket{pid}_atm.pdb")
        if not os.path.exists(atm_pdb):
            pocket.setdefault("x_bary_(sph)", 0.0)
            pocket.setdefault("y_bary_(sph)", 0.0)
            pocket.setdefault("z_bary_(sph)", 0.0)
            continue
        coords = []
        with open(atm_pdb) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                    except ValueError:
                        pass
        if coords:
            arr = np.array(coords)
            cx, cy, cz = arr.mean(axis=0)
            pocket["x_bary_(sph)"] = round(float(cx), 3)
            pocket["y_bary_(sph)"] = round(float(cy), 3)
            pocket["z_bary_(sph)"] = round(float(cz), 3)
        else:
            pocket["x_bary_(sph)"] = 0.0
            pocket["y_bary_(sph)"] = 0.0
            pocket["z_bary_(sph)"] = 0.0

    return pockets


@router.post("/fpocket", response_model=TaskRef, summary="Detect binding pockets with Fpocket")
async def run_fpocket(req: FpocketRequest):
    """
    Run Fpocket to detect and rank binding pockets on a protein structure.
    Returns pocket coordinates, druggability scores, and volumes.
    """
    output_dir = os.path.join(settings.OUTPUT_DIR, req.job_name, "fpocket")
    os.makedirs(output_dir, exist_ok=True)

    async def _run(task):
        task.progress = 10
        task.progress_msg = "Running Fpocket pocket detection..."

        import shutil
        pdb_filename = os.path.basename(req.input_pdb)
        # Use provided path if it exists, otherwise fall back to inputs/
        src = req.input_pdb if os.path.exists(req.input_pdb) else f"/data/oih/inputs/{pdb_filename}"
        pdb_dest = f"/data/oih/outputs/{req.job_name}/{pdb_filename}"
        os.makedirs(f"/data/oih/outputs/{req.job_name}", exist_ok=True)
        shutil.copy(src, pdb_dest)

        cmd = [
            "fpocket",
            "-f", f"/data/oih/outputs/{req.job_name}/{pdb_filename}",
            "-m", str(req.min_sphere_size),
        ]

        retcode, output = await run_in_container_streaming(
            settings.CONTAINER_FPOCKET, cmd, task,
            timeout=settings.TIMEOUT_FPOCKET
        )

        if retcode != 0:
            raise RuntimeError(f"Fpocket failed (exit {retcode})")

        pdb_stem = pdb_filename.replace('.pdb','').replace('.cif','')
        fpocket_out = f"/data/oih/outputs/{req.job_name}/{pdb_stem}_out"
        pockets = _parse_fpocket_output(fpocket_out)

        # Filter by druggability score
        if req.min_druggability_score > 0:
            pockets = [
                p for p in pockets
                if p.get("drug_score", p.get("druggability_score", 0)) >= req.min_druggability_score
            ]

        task.progress_msg = f"Found {len(pockets)} pocket(s)."
        return {
            "pockets": pockets[:10],
            "output_dir": output_dir,
            "num_pockets": len(pockets),
        }

    task = await task_manager.submit("fpocket", req.model_dump(), _run)
    return TaskRef(task_id=task.task_id, status=task.status, tool="fpocket",
                   poll_url=f"/api/v1/tasks/{task.task_id}")


@router.post("/p2rank", response_model=TaskRef, summary="ML-based pocket prediction with P2Rank")
async def run_p2rank(req: P2RankRequest):
    """
    Run P2Rank for machine-learning based binding site prediction.
    Supports default, AlphaFold-optimized, and conservation-aware models.
    Returns ranked pockets with probability scores.
    """
    output_dir = os.path.join(settings.OUTPUT_DIR, req.job_name, "p2rank")
    os.makedirs(output_dir, exist_ok=True)

    async def _run(task):
        task.progress = 10
        task.progress_msg = "Running P2Rank prediction..."

        pdb_filename = os.path.basename(req.input_pdb)
        # Use provided path if it exists, otherwise fall back to inputs/
        pdb_src = req.input_pdb if os.path.exists(req.input_pdb) else f"/data/oih/inputs/{pdb_filename}"

        model_flag = ""
        if req.model == "alphafold":
            model_flag = "-c alphafold"
        elif req.model == "conservation":
            model_flag = "-c conservation"

        base_cmd = [
            "bash", "-c",
            f"/app/p2rank/prank predict {model_flag} "
            f"-f {pdb_src} "
            f"-o /data/oih/outputs/{req.job_name}/p2rank "
            f"2>&1"
        ]

        retcode, output = await run_in_container_streaming(
            settings.CONTAINER_P2RANK, base_cmd, task,
            timeout=settings.TIMEOUT_P2RANK
        )

        if retcode != 0:
            raise RuntimeError(f"P2Rank failed (exit {retcode})")

        # Parse predictions CSV
        pockets = []
        csv_files = list(Path(output_dir).glob("**/*.csv"))
        for csv_path in csv_files:
            if "predictions" in csv_path.name:
                import csv
                with open(csv_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        pockets.append({k.strip(): v.strip() for k, v in row.items()})

        task.progress_msg = f"Found {len(pockets)} pocket(s)."
        return {"pockets": pockets, "output_dir": output_dir}

    task = await task_manager.submit("p2rank", req.model_dump(), _run)
    return TaskRef(task_id=task.task_id, status=task.status, tool="p2rank",
                   poll_url=f"/api/v1/tasks/{task.task_id}")
