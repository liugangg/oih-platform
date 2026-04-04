"""
Immunology Tools Router — DiscoTope3 B-cell epitope prediction + IgFold antibody structure
"""
# --- SYNC_NOTES (auto-generated from CLAUDE.md, do not edit) ---
# DiscoTope3 notes:
#   - Container NVIDIA_VISIBLE_DEVICES=1, always use device=0 (not 1)
#   - CLI must run from /app/discotope3/discotope3/ directory (bare imports)
#   - python3 -u (no python alias in container)
#   - struc_type: solved | alphafold
# --- /SYNC_NOTES ---

import os
import json
import csv
import logging
from fastapi import APIRouter
from schemas.models import DiscoTope3Request, IgFoldRequest, TaskRef
from core.task_manager import task_manager
from core.docker_client import run_in_container_streaming
from core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/discotope3", response_model=TaskRef, summary="Predict B-cell epitopes with DiscoTope3")
async def run_discotope3(req: DiscoTope3Request):
    """
    Run DiscoTope3 to predict B-cell epitope propensity on protein structures.
    Returns per-residue epitope scores with calibrated normalization.
    """
    output_dir = os.path.join(settings.OUTPUT_DIR, req.job_name, "discotope3")
    os.makedirs(output_dir, exist_ok=True)

    async def _run(task):
        task.progress = 5
        task.progress_msg = "Preparing DiscoTope3 input..."

        # Resolve input PDB path (upstream tool output or inputs/ fallback)
        src = req.input_pdb if os.path.exists(req.input_pdb) \
              else f"/data/oih/inputs/{os.path.basename(req.input_pdb)}"
        if not os.path.exists(src):
            raise FileNotFoundError(f"Input PDB not found: {src} (original: {req.input_pdb})")

        container_out = f"/data/oih/outputs/{req.job_name}/discotope3"

        # Build command — must cd into discotope3 source dir for bare imports
        # --models_dir must point to /app/discotope3/models/ (parent of source dir)
        cmd_parts = [
            "bash", "-c",
            "cd /app/discotope3/discotope3 && python3 -u main.py"
            f" --pdb_or_zip_file {src}"
            f" --out_dir {container_out}"
            f" --models_dir /app/discotope3/models"
            f" --struc_type {req.struc_type}"
            f" --calibrated_score_epi_threshold {req.calibrated_score_epi_threshold}"
            + (" --multichain_mode" if req.multichain_mode else "")
            + (" --cpu_only" if req.cpu_only else "")
        ]

        task.progress = 10
        task.progress_msg = "Running DiscoTope3 epitope prediction..."

        retcode, out = await run_in_container_streaming(
            settings.CONTAINER_DISCOTOPE3, cmd_parts, task,
            timeout=settings.TIMEOUT_DISCOTOPE3)

        if retcode != 0:
            raise RuntimeError(f"DiscoTope3 failed (exit {retcode}): {out[-2000:]}")

        task.progress = 80
        task.progress_msg = "Parsing DiscoTope3 results..."

        # Parse CSV output — DiscoTope3 writes <pdb_name>/<chain>_discotope3.csv in subdirs
        epitopes = []
        result_files = []
        for root, dirs, files in os.walk(output_dir):
            for fname in files:
                result_files.append(os.path.relpath(os.path.join(root, fname), output_dir))
                if fname.endswith(".csv"):
                    csv_path = os.path.join(root, fname)
                    try:
                        with open(csv_path) as f:
                            for row in csv.DictReader(f):
                                epitopes.append(row)
                    except Exception as e:
                        logger.warning(f"Failed to parse {fname}: {e}")

        # Count epitope residues — 'epitope' column is True/False string
        n_epitope = sum(
            1 for e in epitopes
            if str(e.get("epitope", "")).strip().lower() == "true"
        )

        task.progress = 100
        task.progress_msg = f"DiscoTope3 done: {n_epitope} epitope residues from {len(epitopes)} total"

        return {
            "epitopes": epitopes,
            "num_residues": len(epitopes),
            "num_epitope_residues": n_epitope,
            "threshold": req.calibrated_score_epi_threshold,
            "result_files": result_files,
            "output_dir": output_dir,
        }

    task = await task_manager.submit("discotope3", req.model_dump(), _run)
    return TaskRef(task_id=task.task_id, status=task.status, tool="discotope3",
                   poll_url=f"/api/v1/tasks/{task.task_id}")


@router.post("/igfold", response_model=TaskRef, summary="Predict antibody structure with IgFold")
async def run_igfold(req: IgFoldRequest):
    """
    Predict antibody/nanobody 3D structure from sequence using IgFold.
    ~2s per sequence on GPU. Use as pre-filter before AF3 validation.
    """
    output_dir = os.path.join(settings.OUTPUT_DIR, req.job_name, "igfold")
    os.makedirs(output_dir, exist_ok=True)

    async def _run(task):
        task.progress = 5
        task.progress_msg = "Preparing IgFold prediction..."

        output_pdb = f"/data/oih/outputs/{req.job_name}/igfold/pred.pdb"
        result_json = f"/data/oih/outputs/{req.job_name}/igfold/result.json"

        # Build inline Python script for IgFold
        sequences_json = json.dumps(req.sequences)
        script = f"""
import json, sys, warnings
warnings.filterwarnings("ignore")

from igfold import IgFoldRunner
runner = IgFoldRunner()
sequences = json.loads('{sequences_json}')
import os as _os
pred_stem = '/data/oih/outputs/{req.job_name}/igfold/pred'
pred = runner.fold(
    pred_stem,
    sequences=sequences,
    do_refine={str(req.do_refine)},
    do_renum=False,
)
# IgFold saves without .pdb extension — rename for consistency
if _os.path.exists(pred_stem) and not _os.path.exists(pred_stem + '.pdb'):
    _os.rename(pred_stem, pred_stem + '.pdb')

# Extract pRMSD scores (per-residue predicted RMSD, lower = better confidence)
# prmsd shape: [1, n_residues, 4] — take mean across 4 atom types per residue
prmsd_raw = pred.prmsd.detach().cpu().numpy()
prmsd_per_res = prmsd_raw[0].mean(axis=1).tolist()  # mean across atom types
mean_prmsd = sum(prmsd_per_res) / len(prmsd_per_res) if prmsd_per_res else 99.0

# Convert pRMSD to pseudo-pLDDT for compatibility (100 - prmsd*20, clamped 0-100)
plddt = [max(0.0, min(100.0, 100.0 - x * 20.0)) for x in prmsd_per_res]
mean_plddt = sum(plddt) / len(plddt) if plddt else 0.0

result = {{
    "output_pdb": "{output_pdb}",
    "mean_plddt": round(mean_plddt, 3),
    "mean_prmsd": round(mean_prmsd, 4),
    "plddt_scores": [round(x, 1) for x in plddt],
    "prmsd_scores": [round(x, 4) for x in prmsd_per_res],
    "num_residues": len(plddt),
    "sequences": sequences,
    "do_refine": {str(req.do_refine)},
}}

with open("{result_json}", "w") as f:
    json.dump(result, f)

print(json.dumps({{"mean_plddt": result["mean_plddt"], "num_residues": result["num_residues"]}}))
"""
        # Write script to temp file
        script_path = f"/data/oih/tmp/{req.job_name}_igfold.py"
        local_script = script_path.replace("/data/oih", settings.DATA_ROOT)
        os.makedirs(os.path.dirname(local_script), exist_ok=True)
        with open(local_script, "w") as f:
            f.write(script)

        task.progress = 10
        n_chains = len(req.sequences)
        total_res = sum(len(s) for s in req.sequences.values())
        task.progress_msg = f"Running IgFold on {n_chains} chain(s), {total_res} residues..."

        retcode, out = await run_in_container_streaming(
            settings.CONTAINER_IGFOLD, ["python3", "-u", script_path], task,
            timeout=settings.TIMEOUT_IGFOLD)

        if retcode != 0:
            raise RuntimeError(f"IgFold failed (exit {retcode}): {out[-2000:]}")

        task.progress = 90
        task.progress_msg = "Parsing IgFold results..."

        # Read result JSON
        local_result = result_json.replace("/data/oih", settings.DATA_ROOT)
        if not os.path.exists(local_result):
            raise RuntimeError(f"IgFold result file not found: {local_result}")

        with open(local_result) as f:
            result = json.load(f)

        task.progress = 100
        task.progress_msg = (
            f"IgFold done: {result['num_residues']} residues, "
            f"mean pLDDT={result['mean_plddt']:.1f}, mean pRMSD={result['mean_prmsd']:.2f} Å"
        )
        return result

    task = await task_manager.submit("igfold", req.model_dump(), _run)
    return TaskRef(task_id=task.task_id, status=task.status, tool="igfold",
                   poll_url=f"/api/v1/tasks/{task.task_id}")
