"""
Structure Prediction Router
Tools: AlphaFold3
"""
# --- SYNC_NOTES (auto-generated from CLAUDE.md, do not edit) ---
# ALPHAFOLD3 注意事项（来自 CLAUDE.md，勿手动编辑）：
#   - 容器内 NVIDIA_VISIBLE_DEVICES=1，永远用 device=0 / gpu_id=0（不要用1）
#   - 根因**：v3 任务 5 个 AF3 设计全部失败。rank1 超时 1800s（但实际已跑完，ipTM=0.48），rank2-5 被路由到 DEGRADED 队列 OOM exit 1
#   - 修复 1**：新增 `_wait_for_af3_task()` 无限等待函数，每 30s poll，仅在 OOM/exit1/cancelled/连续10次同错 时判定失败
#   - 不再降级到 DEGRADED 导致 OOM crash
#   - 原因**：CIF→PDB 转换失败时 `af3_pdb=None` → `"No PDB for FreeSASA"` 错误
# --- /SYNC_NOTES ---

import json
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from schemas.models import AlphaFold3Request, AlphaFold3Result, TaskRef
from core.task_manager import task_manager
from core.docker_client import run_in_container_streaming
from core.config import settings

router = APIRouter()


def _build_af3_json(req: AlphaFold3Request, input_dir: str) -> str:
    """Build AlphaFold3 input JSON"""
    sequences = []
    chain_id_counter = 0
    for chain in req.chains:
        if chain.type in ("protein", "rna", "dna"):
            ids = [chr(65 + chain_id_counter + i) for i in range(chain.count)]
            chain_id_counter += chain.count
            chain_dict = {
                "id": ids,
                "sequence": chain.sequence,
            }
            if chain.modifications:
                chain_dict["modifications"] = [
                    {"ptmType": m.ptmType, "ptmPosition": m.ptmPosition}
                    for m in chain.modifications
                ]
            sequences.append({
                "protein" if chain.type == "protein" else chain.type: chain_dict
            })
        elif chain.type == "ligand":
            sequences.append({
                "ligand": {
                    "id": "LIG",
                    "smiles": chain.smiles,
                }
            })

    af3_input = {
        "name": req.job_name,
        "sequences": sequences,
        "modelSeeds": list(range(req.num_seeds)),
        "dialect": "alphafold3",
        "version": 2,
    }

    json_path = os.path.join(input_dir, f"{req.job_name}.json")
    with open(json_path, "w") as f:
        json.dump(af3_input, f, indent=2)
    return json_path


@router.post("/alphafold3", response_model=TaskRef, summary="Predict protein structure with AlphaFold3")
async def predict_alphafold3(req: AlphaFold3Request):
    """
    Run AlphaFold3 structure prediction.
    Supports protein, RNA, DNA, ligand, and multi-chain complexes.
    Returns a task_id for async polling.
    """
    input_dir  = os.path.join(settings.INPUT_DIR,  req.job_name)
    output_dir = os.path.join(settings.OUTPUT_DIR, req.job_name, "alphafold3")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    json_path = _build_af3_json(req, input_dir)

    async def _run(task):
        task.progress = 5
        task.progress_msg = "Starting AlphaFold3..."

        cmd = [
            "python", "/app/alphafold/run_alphafold.py",
            f"--json_path=/data/oih/inputs/{req.job_name}/{req.job_name}.json",
            f"--output_dir=/data/oih/outputs/{req.job_name}/alphafold3",
            "--model_dir=/data/alphafold3_models",
            "--db_dir=/data/alphafold3_db",
            "--flash_attention_implementation=triton",
            # num_seeds already set in JSON via modelSeeds
        ]
        # run_relaxation not supported in this AF3 version

        retcode, output = await run_in_container_streaming(
            settings.CONTAINER_ALPHAFOLD3, cmd, task,
            timeout=settings.TIMEOUT_ALPHAFOLD3
        )

        if retcode != 0:
            raise RuntimeError(f"AlphaFold3 failed (exit {retcode})")

        # Collect output PDB files
        cif_files = [str(p) for p in Path(output_dir).glob("**/*.cif")]
        json_files = [str(p) for p in Path(output_dir).glob("**/*summary_confidences*.json")]
        task.output_files = cif_files + json_files
        task.progress = 100
        task.progress_msg = f"Done. {len(cif_files)} structure(s) generated."

        return {
            "cif_files": cif_files,
            "confidence_files": json_files,
            "output_dir": output_dir,
            "num_structures": len(cif_files),
        }

    task = await task_manager.submit("alphafold3", req.model_dump(), _run)

    return TaskRef(
        task_id=task.task_id,
        status=task.status,
        tool="alphafold3",
        poll_url=f"/api/v1/tasks/{task.task_id}",
    )


# ─── Fetch PDB from RCSB ─────────────────────────────────────────────────────

from pydantic import BaseModel, Field
from typing import Optional, List
import urllib.request

class FetchPDBRequest(BaseModel):
    pdb_id: str
    chains: Optional[str] = None  # e.g. "A" or "A,B"

@router.post("/fetch_pdb")
async def fetch_pdb(req: FetchPDBRequest):
    """Download PDB from RCSB and optionally extract specific chains"""
    pdb_id = req.pdb_id.upper().strip()
    out_dir = f"{settings.OUTPUT_DIR}/fetch_pdb"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{pdb_id}.pdb"

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download {pdb_id}: {e}")

    # Optional: extract specific chains using simple filtering
    if req.chains:
        chain_set = set(req.chains.upper().replace(" ", "").split(","))
        clean_path = f"{out_dir}/{pdb_id}_chain{''.join(sorted(chain_set))}.pdb"
        with open(out_path) as f_in, open(clean_path, "w") as f_out:
            for line in f_in:
                if line.startswith(("ATOM", "TER")):
                    if len(line) > 21 and line[21] in chain_set:
                        f_out.write(line)
                elif line.startswith("END"):
                    f_out.write(line)
        out_path = clean_path

    return {
        "status": "completed",
        "pdb_id": pdb_id,
        "output_pdb": out_path,
    }


# ─── Fetch Molecule from PubChem ──────────────────────────────────────────────

class FetchMoleculeRequest(BaseModel):
    query: str  # drug name, CID, or SMILES
    output_format: str = "sdf"  # sdf or smiles

@router.post("/fetch_molecule")
async def fetch_molecule(req: FetchMoleculeRequest):
    """Resolve drug/molecule name to SMILES and 3D SDF via PubChem"""
    import urllib.request, urllib.parse, json as _json

    query = req.query.strip()
    out_dir = f"{settings.OUTPUT_DIR}/fetch_molecule"
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: resolve name/CID to CID
    if query.isdigit():
        cid = query
    else:
        search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(query)}/cids/JSON"
        try:
            with urllib.request.urlopen(search_url, timeout=15) as resp:
                data = _json.loads(resp.read())
                cid = str(data["IdentifierList"]["CID"][0])
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Molecule '{query}' not found on PubChem: {e}")

    # Step 2: get SMILES
    smi_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES,IsomericSMILES,MolecularFormula,MolecularWeight,IUPACName/JSON"
    try:
        with urllib.request.urlopen(smi_url, timeout=15) as resp:
            props = _json.loads(resp.read())["PropertyTable"]["Properties"][0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get properties for CID {cid}: {e}")

    # Step 3: download 3D SDF
    sdf_path = f"{out_dir}/{query.replace(' ','_')}_{cid}.sdf"
    sdf_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF?record_type=3d"
    try:
        urllib.request.urlretrieve(sdf_url, sdf_path)
    except Exception:
        sdf_path = None  # 3D not available, 2D fallback
        try:
            sdf_url_2d = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF"
            urllib.request.urlretrieve(sdf_url_2d, f"{out_dir}/{query.replace(' ','_')}_{cid}_2d.sdf")
            sdf_path = f"{out_dir}/{query.replace(' ','_')}_{cid}_2d.sdf"
        except Exception:
            pass

    return {
        "status": "completed",
        "query": query,
        "cid": cid,
        "smiles": props.get("CanonicalSMILES") or props.get("SMILES") or props.get("ConnectivitySMILES", ""),
        "isomeric_smiles": props.get("IsomericSMILES") or props.get("SMILES") or props.get("ConnectivitySMILES", ""),
        "iupac_name": props.get("IUPACName", ""),
        "formula": props.get("MolecularFormula", ""),
        "molecular_weight": props.get("MolecularWeight", 0),
        "sdf_path": sdf_path,
    }


# ─── Extract Interface Residues from Known Complex ──────────────────────────

class ExtractInterfaceRequest(BaseModel):
    job_name: str
    complex_pdb: str = Field(..., description="Path to PDB with receptor + ligand chains")
    receptor_chain: str = Field(..., description="Chain ID for the receptor/antigen, e.g. 'C'")
    ligand_chains: List[str] = Field(..., description="Chain IDs for antibody/ligand, e.g. ['A','B']")
    cutoff_angstrom: float = Field(5.0, description="Distance cutoff for contacts (Å)")
    top_n: int = Field(8, description="Max number of interface residues to return")


@router.post("/extract_interface", summary="Extract antibody-antigen interface residues from a known complex PDB")
async def extract_interface_residues(req: ExtractInterfaceRequest):
    """
    Analyze a co-crystal PDB structure to find receptor residues at the
    antibody-antigen interface.  Uses BioPython NeighborSearch (CPU-only,
    runs inside the alphafold3 container which has BioPython installed).
    Returns the top-N receptor residues ranked by number of cross-chain
    atomic contacts within the distance cutoff.
    """
    import logging
    logger = logging.getLogger("oih")

    pdb_path = req.complex_pdb
    if not os.path.exists(pdb_path):
        raise HTTPException(status_code=400, detail=f"PDB file not found: {pdb_path}")

    # Build the analysis script to run inside the container
    script = (
        "from Bio.PDB import PDBParser, NeighborSearch\n"
        "import json, sys\n"
        "p = PDBParser(QUIET=True)\n"
        f"s = p.get_structure('complex', '{pdb_path}')\n"
        "ligand_atoms = []\n"
        f"for cid in {req.ligand_chains!r}:\n"
        "    try: ligand_atoms += list(s[0][cid].get_atoms())\n"
        "    except: pass\n"
        "if not ligand_atoms:\n"
        "    print(json.dumps({'error': 'no ligand atoms found'}))\n"
        "    sys.exit(0)\n"
        "ns = NeighborSearch(ligand_atoms)\n"
        "contacts = {}\n"
        f"for res in s[0]['{req.receptor_chain}']:\n"
        "    for atom in res:\n"
        f"        hits = ns.search(atom.coord, {req.cutoff_angstrom})\n"
        "        if hits:\n"
        f"            rid = '{req.receptor_chain}' + str(res.id[1])\n"
        "            contacts[rid] = contacts.get(rid, 0) + len(hits)\n"
        "sorted_res = sorted(contacts.items(), key=lambda x: x[1], reverse=True)\n"
        f"top = sorted_res[:{req.top_n}]\n"
        "print(json.dumps({'interface_residues': [r for r,c in top], "
        "'num_contacts': dict(sorted_res), 'total_interface': len(sorted_res)}))\n"
    )

    import subprocess, json as _json
    cmd = [
        "docker", "exec", "oih-discotope3",
        "python3", "-c", script,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(f"BioPython script failed: {proc.stderr[:500]}")
        result = _json.loads(proc.stdout.strip())
        if "error" in result:
            raise RuntimeError(result["error"])
    except _json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Failed to parse output: {proc.stdout[:300]}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Interface extraction timed out (60s)")

    result["method"] = "structural_database"
    result["source_pdb"] = os.path.basename(pdb_path)
    result["receptor_chain"] = req.receptor_chain
    result["ligand_chains"] = req.ligand_chains
    result["cutoff_angstrom"] = req.cutoff_angstrom

    logger.info("[extract_interface] %s: %d interface residues (top %d: %s)",
                result["source_pdb"], result.get("total_interface", 0),
                len(result["interface_residues"]), result["interface_residues"])

    return result
