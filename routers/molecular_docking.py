"""
Molecular Docking Router
Tools: Vina-GPU, AutoDock-GPU, GNINA, DiffDock
"""
# --- SYNC_NOTES (auto-generated from CLAUDE.md, do not edit) ---
# AUTODOCK+DIFFDOCK+GNINA 注意事项（来自 CLAUDE.md，勿手动编辑）：
#   - 容器内 NVIDIA_VISIBLE_DEVICES=1，永远用 device=0 / gpu_id=0（不要用1）
# --- /SYNC_NOTES ---

import os, subprocess
from pathlib import Path
from fastapi import APIRouter, HTTPException
from schemas.models import DockingRequest, DockingEngine, DiffDockRequest, TaskRef
from core.task_manager import task_manager
from core.docker_client import run_in_container_streaming, run_in_container
from core.config import settings

router = APIRouter()


def _parse_vina_output(output: str) -> list:
    """Parse Vina/GNINA score table from stdout"""
    poses = []
    in_table = False
    for line in output.splitlines():
        if "-----+------------" in line:
            in_table = True
            continue
        if in_table and line.strip():
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                poses.append({
                    "pose_id": int(parts[0]),
                    "affinity_kcal_mol": float(parts[1]),
                    "rmsd_lb": float(parts[2]),
                    "rmsd_ub": float(parts[3]) if len(parts) > 3 else None,
                })
    return poses

def _parse_sdf_poses(sdf_path: str) -> list:
    """Parse poses from SDF output file"""
    poses = []
    try:
        with open(sdf_path) as f:
            sdf_content = f.read()
        blocks = [b for b in sdf_content.split("$$$$") if b.strip()]
        for i, block in enumerate(blocks):
            pose = {"pose_id": i+1}
            lines = block.splitlines()
            for j, line in enumerate(lines):
                for prop in ["minimizedAffinity", "CNNscore", "CNNaffinity"]:
                    if f"<{prop}>" in line and j+1 < len(lines):
                        try:
                            pose[prop] = float(lines[j+1].strip())
                        except:
                            pass
            if "minimizedAffinity" in pose:
                pose["affinity_kcal_mol"] = pose["minimizedAffinity"]
            if pose.get("affinity_kcal_mol"):
                poses.append(pose)
    except Exception as e:
        pass
    return poses


async def _prepare_ligand_pdbqt(container: str, smiles: str, job_name: str) -> str:
    """Convert SMILES → PDBQT using obabel inside container"""
    lig_dir = os.path.join(settings.INPUT_DIR, job_name)
    os.makedirs(lig_dir, exist_ok=True)

    smi_path = os.path.join(settings.INPUT_DIR, job_name, "ligand.smi")
    pdbqt_path_host = os.path.join(settings.INPUT_DIR, job_name, "ligand.pdbqt")
    pdbqt_path = f"/data/oih/inputs/{job_name}/ligand.pdbqt"

    # Write SMILES file
    with open(os.path.join(settings.INPUT_DIR, job_name, "ligand.smi"), "w") as f:
        f.write(smiles + "\n")

    import asyncio as _aio
    proc = await _aio.create_subprocess_exec(
        "obabel", smi_path, "-O", pdbqt_path_host, "--gen3d", "-p", "7.4",
        stdout=_aio.subprocess.PIPE,
        stderr=_aio.subprocess.PIPE,
    )
    stdout, stderr = await _aio.wait_for(proc.communicate(), timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"Ligand prep failed: {stderr.decode()}")
    return pdbqt_path


# ─── GNINA (recommended default) ─────────────────────────────────────────────

@router.post("/gnina", response_model=TaskRef, summary="Deep learning docking with GNINA")
async def run_gnina(req: DockingRequest):
    """
    Run GNINA molecular docking (CNN-scored, GPU-accelerated).
    GNINA uses deep learning scoring on top of Vina search.
    Best general-purpose docking engine.
    """
    output_dir = os.path.join(settings.OUTPUT_DIR, req.job_name, "gnina")
    os.makedirs(output_dir, exist_ok=True)

    async def _run(task):
        task.progress = 10
        task.progress_msg = "Preparing ligand for GNINA..."

        _receptor = req.receptor_pdb if os.path.exists(req.receptor_pdb) else f"/data/oih/inputs/{os.path.basename(req.receptor_pdb)}"
        lig_pdbqt = await _prepare_ligand_pdbqt(
            settings.CONTAINER_GNINA,
            req.ligand if req.ligand.startswith("C") or req.ligand.startswith("c") else req.ligand,
            req.job_name,
        )

        task.progress = 25
        task.progress_msg = "Running GNINA docking..."

        out_sdf = f"/data/oih/outputs/{req.job_name}/gnina/poses.sdf"
        cmd = [
            "gnina",
            "--receptor", _receptor,
            "--ligand", lig_pdbqt,
            "--out", out_sdf,
            "--center_x", str(req.center_x or 0),
            "--center_y", str(req.center_y or 0),
            "--center_z", str(req.center_z or 0),
            "--size_x", str(req.box_size_x),
            "--size_y", str(req.box_size_y),
            "--size_z", str(req.box_size_z),
            "--num_modes", str(req.num_poses),
            "--exhaustiveness", str(req.exhaustiveness),
            "--autobox_add", "4",
            "--device", "0",
        ]

        retcode, output = await run_in_container_streaming(
            settings.CONTAINER_GNINA, cmd, task,
            timeout=settings.TIMEOUT_DOCKING
        )

        if retcode != 0:
            raise RuntimeError(f"GNINA failed (exit {retcode})")

        poses = _parse_sdf_poses(out_sdf.replace("/data/oih/outputs", settings.OUTPUT_DIR))
        task.output_files = [out_sdf]
        task.progress_msg = f"Done. {len(poses)} pose(s). Best: {poses[0]['affinity_kcal_mol'] if poses else 'N/A'} kcal/mol"

        return {
            "poses": poses,
            "best_affinity": poses[0]["affinity_kcal_mol"] if poses else None,
            "output_sdf": out_sdf,
            "output_dir": output_dir,
        }

    task = await task_manager.submit("gnina", req.model_dump(), _run)
    return TaskRef(task_id=task.task_id, status=task.status, tool="gnina",
                   poll_url=f"/api/v1/tasks/{task.task_id}")


# ─── Vina-GPU ────────────────────────────────────────────────────────────────

@router.post("/vina-gpu", response_model=TaskRef, summary="GPU-accelerated AutoDock Vina")
async def run_vina_gpu(req: DockingRequest):
    """
    Run Vina-GPU for fast GPU-accelerated docking.
    Good for high-throughput screening.
    """
    output_dir = os.path.join(settings.OUTPUT_DIR, req.job_name, "vina_gpu")
    os.makedirs(output_dir, exist_ok=True)

    async def _run(task):
        task.progress = 10
        task.progress_msg = "Preparing for Vina-GPU..."

        _receptor = req.receptor_pdb if os.path.exists(req.receptor_pdb) else f"{settings.INPUT_DIR}/{os.path.basename(req.receptor_pdb)}"
        lig_pdbqt = await _prepare_ligand_pdbqt(
            settings.CONTAINER_VINA_GPU, req.ligand, req.job_name
        )

        # 准备receptor pdbqt
        import asyncio as _aio
        rec_pdbqt_host = f"{settings.INPUT_DIR}/{req.job_name}/receptor.pdbqt"
        rec_proc = await _aio.create_subprocess_exec(
            "obabel", _receptor, "-O", rec_pdbqt_host, "-xr",
            stdout=_aio.subprocess.PIPE, stderr=_aio.subprocess.PIPE
        )
        await _aio.wait_for(rec_proc.communicate(), timeout=60)

        out_pdbqt = f"/data/oih/outputs/{req.job_name}/vina_gpu/out.pdbqt"
        cmd = [
            "vina_gpu",
            "--receptor", f"/data/oih/inputs/{req.job_name}/receptor.pdbqt",
            "--ligand", lig_pdbqt,
            "--out", out_pdbqt,
            "--center_x", str(req.center_x or 0),
            "--center_y", str(req.center_y or 0),
            "--center_z", str(req.center_z or 0),
            "--size_x", str(req.box_size_x),
            "--size_y", str(req.box_size_y),
            "--size_z", str(req.box_size_z),
            "--num_modes", str(req.num_poses),
            "--exhaustiveness", str(req.exhaustiveness),
            "--thread", "8000",
        ]

        retcode, output = await run_in_container_streaming(
            settings.CONTAINER_VINA_GPU, cmd, task,
            timeout=settings.TIMEOUT_DOCKING
        )

        if retcode != 0:
            raise RuntimeError(f"Vina-GPU failed (exit {retcode})")

        poses = _parse_vina_output(output)
        task.output_files = [out_pdbqt]
        return {"poses": poses, "best_affinity": poses[0]["affinity_kcal_mol"] if poses else None, "output_dir": output_dir}

    task = await task_manager.submit("vina-gpu", req.model_dump(), _run)
    return TaskRef(task_id=task.task_id, status=task.status, tool="vina-gpu",
                   poll_url=f"/api/v1/tasks/{task.task_id}")


# ─── AutoDock-GPU ────────────────────────────────────────────────────────────

@router.post("/autodock-gpu", response_model=TaskRef, summary="AutoDock-GPU docking")
async def run_autodock_gpu(req: DockingRequest):
    """
    Run AutoDock-GPU. Uses GPU-parallelized LGA search.
    Better sampling than Vina at the cost of speed.
    """
    output_dir = os.path.join(settings.OUTPUT_DIR, req.job_name, "autodock_gpu")
    os.makedirs(output_dir, exist_ok=True)

    async def _run(task):
        import subprocess
        pdb_basename = os.path.basename(req.receptor_pdb)
        _receptor = req.receptor_pdb if os.path.exists(req.receptor_pdb) else f"/data/oih/inputs/{pdb_basename}"
        workdir = f"/data/oih/inputs/{req.job_name}"
        os.makedirs(workdir, exist_ok=True)

        cx = req.center_x or 0.0
        cy = req.center_y or 0.0
        cz = req.center_z or 0.0
        sx = req.box_size_x or 25.0
        sy = req.box_size_y or 25.0
        sz = req.box_size_z or 25.0

        # Step1: prody清理PDB（容器内）
        task.progress = 10
        task.progress_msg = "Step1: Cleaning PDB with prody..."
        prody_script = (
            "from prody import parsePDB, writePDB; "
            "s = parsePDB('" + _receptor + "', altloc='first'); "
            "p = s.select('protein'); "
            "writePDB('" + workdir + "/protein_clean.pdb', p)"
        )
        prody_cmd = ["bash", "-c", f"python3 -c '{prody_script}'"]
        retcode, stdout, stderr = await run_in_container(
            settings.CONTAINER_AUTODOCK_GPU, prody_cmd, timeout=60)
        if retcode != 0:
            raise RuntimeError(f"prody failed: {stderr[:300]}")

        # Step2: obabel生成receptor.pdbqt（宿主机）
        task.progress = 20
        task.progress_msg = "Step2: Generating receptor.pdbqt..."
        r = subprocess.run([
            "obabel", f"{workdir}/protein_clean.pdb",
            "-O", f"{workdir}/receptor.pdbqt",
            "-xr", "--partialcharge", "gasteiger"
        ], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f"obabel receptor failed: {r.stderr[:300]}")

        # Step3: 生成GPF文件（宿主机）
        task.progress = 30
        task.progress_msg = "Step3: Generating GPF..."
        npts_x = int(sx / 0.375)
        npts_y = int(sy / 0.375)
        npts_z = int(sz / 0.375)
        gpf_lines = [
            f"npts {npts_x} {npts_y} {npts_z}",
            "gridfld receptor.maps.fld",
            "spacing 0.375",
            "receptor_types A C N NA OA S",
            "ligand_types A C HD N NA OA SA",
            f"receptor {workdir}/receptor.pdbqt",
            f"gridcenter {cx} {cy} {cz}",
            "smooth 0.5",
            "map receptor.A.map",
            "map receptor.C.map",
            "map receptor.HD.map",
            "map receptor.N.map",
            "map receptor.NA.map",
            "map receptor.OA.map",
            "map receptor.SA.map",
            "elecmap receptor.e.map",
            "dsolvmap receptor.d.map",
            "dielectric -0.1465",
        ]
        with open(f"{workdir}/receptor.gpf", "w") as f:
            f.write("\n".join(gpf_lines))

        # Step4: autogrid4生成maps（宿主机）
        task.progress = 40
        task.progress_msg = "Step4: Running autogrid4..."
        r = subprocess.run(
            ["bash", "-c", f"cd {workdir} && autogrid4 -p {workdir}/receptor.gpf -l {workdir}/grid.glg"],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"autogrid4 failed: {r.stderr[:300]}")

        # Step5: obabel生成ligand.pdbqt（宿主机）
        task.progress = 50
        task.progress_msg = "Step5: Preparing ligand..."
        r = subprocess.run([
            "obabel", f"-:{req.ligand}", "-opdbqt", "--gen3d",
            "-O", f"{workdir}/ligand.pdbqt",
            "--partialcharge", "gasteiger"
        ], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f"obabel ligand failed: {r.stderr[:300]}")

        # Step6: autodock_gpu_128wi对接（容器内）
        task.progress = 60
        task.progress_msg = "Step6: Running AutoDock-GPU..."
        dock_cmd = ["bash", "-c",
            f"autodock_gpu_128wi "
            f"--ffile {workdir}/receptor.maps.fld "
            f"--lfile {workdir}/ligand.pdbqt "
            f"--nrun {req.num_poses * 2} "
            f"--devnum 1 "
            f"--resnam {output_dir}/result"
        ]
        retcode, output = await run_in_container_streaming(
            settings.CONTAINER_AUTODOCK_GPU, dock_cmd, task,
            timeout=settings.TIMEOUT_DOCKING)

        if retcode != 0:
            raise RuntimeError(f"AutoDock-GPU failed (exit {retcode})")

        task.progress_msg = "AutoDock-GPU complete."
        return {"output_dir": output_dir, "poses": [], "best_affinity": None}

    task = await task_manager.submit("autodock-gpu", req.model_dump(), _run)
    return TaskRef(task_id=task.task_id, status=task.status, tool="autodock-gpu",
                   poll_url=f"/api/v1/tasks/{task.task_id}")


# ─── DiffDock ────────────────────────────────────────────────────────────────

@router.post("/diffdock", response_model=TaskRef, summary="Diffusion-based blind docking with DiffDock")
async def run_diffdock(req: DiffDockRequest):
    """
    Run DiffDock for blind docking (no box required).
    Uses diffusion model to predict binding poses.
    Best when binding site is unknown.
    """
    output_dir = os.path.join(settings.OUTPUT_DIR, req.job_name, "diffdock")
    os.makedirs(output_dir, exist_ok=True)

    async def _run(task):
        task.progress = 10
        task.progress_msg = "Running DiffDock blind docking..."

        _receptor = req.receptor_pdb if os.path.exists(req.receptor_pdb) else f"/data/oih/inputs/{os.path.basename(req.receptor_pdb)}"

        # Write ligand SMILES file
        smi_dir = os.path.join(settings.INPUT_DIR, req.job_name)
        os.makedirs(smi_dir, exist_ok=True)
        with open(os.path.join(smi_dir, "ligand.smi"), "w") as f:
            f.write(req.ligand_smiles)

        # Build CSV input for DiffDock
        csv_content = f"complex_name,protein_path,protein_sequence,ligand_description\n{req.job_name},{_receptor},,{req.ligand_smiles}\n"
        with open(os.path.join(smi_dir, "diffdock_input.csv"), "w") as f:
            f.write(csv_content)

        cmd = [
            "python3", "/app/DiffDock/inference.py",
            "--protein_ligand_csv", f"/data/oih/inputs/{req.job_name}/diffdock_input.csv",
            "--out_dir", f"/data/oih/outputs/{req.job_name}/diffdock",
            "--inference_steps", str(req.inference_steps),
            "--samples_per_complex", str(req.samples_per_complex),
            "--batch_size", "10",
            "--actual_steps", str(req.inference_steps),
            "--no_final_step_noise",
        ]

        retcode, output = await run_in_container_streaming(
            settings.CONTAINER_DIFFDOCK, cmd, task,
            timeout=settings.TIMEOUT_DIFFDOCK
        )

        if retcode != 0:
            raise RuntimeError(f"DiffDock failed (exit {retcode})")

        # Parse confidence scores from output filenames
        pose_files = sorted(Path(output_dir).glob("**/*.sdf"))
        all_poses = []
        for pf in pose_files:
            confidence = None
            if "confidence" in pf.name:
                try:
                    confidence = float(pf.stem.split("confidence")[-1])
                except:
                    pass
            all_poses.append({"file": str(pf), "confidence": confidence})

        # Sort by confidence descending (higher/less negative = better), nulls last
        all_poses.sort(key=lambda x: x["confidence"] if x["confidence"] is not None else -999)
        all_poses = all_poses[::-1]  # reverse: best first

        ranked_poses = []
        for i, p in enumerate(all_poses[:req.num_poses]):
            ranked_poses.append({
                "rank": i + 1,
                "file": p["file"],
                "confidence": p["confidence"],
            })

        task.output_files = [str(p["file"]) for p in ranked_poses]
        task.progress_msg = f"Done. {len(ranked_poses)} pose(s)."
        return {"ranked_poses": ranked_poses, "output_dir": output_dir}

    task = await task_manager.submit("diffdock", req.model_dump(), _run)
    return TaskRef(task_id=task.task_id, status=task.status, tool="diffdock",
                   poll_url=f"/api/v1/tasks/{task.task_id}")


# ─── Smart Router ─────────────────────────────────────────────────────────────

@router.post("/dock", response_model=TaskRef, summary="Auto-select best docking engine")
async def smart_dock(req: DockingRequest):
    """
    Intelligently routes to the best docking engine based on request parameters:
    - No binding site known → DiffDock
    - Need CNN scoring → GNINA
    - High-throughput → Vina-GPU
    - Precision → AutoDock-GPU
    """
    if req.engine == DockingEngine.DIFFDOCK or (req.center_x is None):
        diffdock_req = DiffDockRequest(
            job_name=req.job_name,
            receptor_pdb=req.receptor_pdb,
            ligand_smiles=req.ligand,
            num_poses=req.num_poses,
        )
        return await run_diffdock(diffdock_req)
    elif req.engine == DockingEngine.VINA_GPU:
        return await run_vina_gpu(req)
    elif req.engine == DockingEngine.AUTODOCK_GPU:
        return await run_autodock_gpu(req)
    else:  # GNINA default
        return await run_gnina(req)
