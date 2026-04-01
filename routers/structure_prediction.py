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
#   - AF3 验证时按结构域截取抗原序列，避免全长序列降低ipTM精度
#   - `num_seeds=3` 用于 binder_design_pipeline AF3 验证（速度/准确性平衡）
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
            "--jackhmmer_n_cpu=32",
            "--nhmmer_n_cpu=32",
            # num_seeds already set in JSON via modelSeeds
        ]
        # run_relaxation not supported in this AF3 version

        # Dynamic timeout based on total sequence length
        total_len = sum(
            len(c.sequence or '') for c in req.chains if c.type in ('protein', 'rna', 'dna')
        )
        if total_len > 1000:
            af3_timeout = 7200   # 2h — very large complexes
        elif total_len > 500:
            af3_timeout = 5400   # 90min — large complexes
        else:
            af3_timeout = 3600   # 1h — includes MSA search time (jackhmmer can take 30min+)

        import logging as _log
        _log.getLogger("oih").info(
            "[AF3] %s: total_len=%d → timeout=%ds", req.job_name, total_len, af3_timeout)

        retcode, output = await run_in_container_streaming(
            settings.CONTAINER_ALPHAFOLD3, cmd, task,
            timeout=af3_timeout
        )

        if retcode != 0:
            # Capture last 20 lines of output for error diagnosis
            output_lines = (output or "").strip().splitlines()
            tail = "\n".join(output_lines[-20:]) if output_lines else "(no output)"
            raise RuntimeError(
                f"AlphaFold3 failed (exit {retcode})\n--- last 20 lines ---\n{tail}"
            )

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
    logger.info("[extract_interface] pdb=%s receptor=%s ligand=%s cutoff=%.1f top_n=%d",
                pdb_path, req.receptor_chain, req.ligand_chains, req.cutoff_angstrom, req.top_n)
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


# ─── PeSTo PPI Interface Prediction ─────────────────────────────────────────

class PeSToRequest(BaseModel):
    job_name: str = Field("pesto_predict", description="Job identifier")
    input_pdb: str = Field(..., description="Path to PDB file (single chain recommended)")
    chain_id: Optional[str] = Field(None, description="Chain to analyze (auto-extracts from complex)")
    threshold: float = Field(0.5, description="PPI score threshold for hotspot residues")


@router.post("/pesto_predict", summary="PeSTo PPI interface prediction")
async def pesto_predict(req: PeSToRequest):
    """Predict protein-protein interaction interface using PeSTo transformer.

    ROC AUC=0.92, replaces P2Rank+DiscoTope3 for binder design hotspot selection.
    For complex PDBs, specify chain_id to auto-extract the target chain.
    Runs on CPU in oih-proteinmpnn container (~10s per structure).
    """
    import subprocess
    import logging
    logger = logging.getLogger("oih")

    pdb_path = req.input_pdb
    if not os.path.exists(pdb_path):
        raise HTTPException(status_code=400, detail=f"PDB not found: {pdb_path}")

    # Auto-extract single chain if chain_id specified
    single_chain_pdb = pdb_path
    if req.chain_id:
        extract_script = (
            f"import gemmi\n"
            f"st = gemmi.read_structure('{pdb_path}')\n"
            f"model = st[0]\n"
            f"chains_to_remove = [c.name for c in model if c.name != '{req.chain_id}']\n"
            f"for cn in chains_to_remove:\n"
            f"    model.remove_chain(cn)\n"
            f"st.write_pdb('/tmp/pesto_input_{req.chain_id}.pdb')\n"
            f"print('OK')\n"
        )
        try:
            proc = subprocess.run(
                ["docker", "exec", "-w", "/app/pesto", "oih-proteinmpnn", "python3", "-c", extract_script],
                capture_output=True, text=True, timeout=30,
            )
            if "OK" in proc.stdout:
                single_chain_pdb = f"/tmp/pesto_input_{req.chain_id}.pdb"
                logger.info("[PeSTo] Extracted chain %s → %s", req.chain_id, single_chain_pdb)
        except Exception as e:
            logger.warning("[PeSTo] Chain extraction failed: %s, using full PDB", e)

    # Run PeSTo prediction
    output_json = f"/tmp/pesto_output_{req.job_name}.json"
    cmd = [
        "docker", "exec", "-w", "/app/pesto", "oih-proteinmpnn",
        "python3", "predict_cli.py",
        "--input", single_chain_pdb,
        "--output", output_json,
        "--threshold", str(req.threshold),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"PeSTo failed: {proc.stderr[:300]}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="PeSTo timed out (120s)")

    # Read result from container
    try:
        read_proc = subprocess.run(
            ["docker", "exec", "oih-proteinmpnn", "cat", output_json],
            capture_output=True, text=True, timeout=10,
        )
        import json as _json
        result = _json.loads(read_proc.stdout.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read PeSTo output: {e}")

    result["chain_id"] = req.chain_id
    result["input_pdb"] = req.input_pdb
    logger.info("[PeSTo] %s chain=%s: %d hotspots (>%.1f), max=%.3f",
                os.path.basename(pdb_path), req.chain_id or "all",
                result.get("num_hotspots", 0), req.threshold, result.get("max_score", 0))

    return result


# ─── ipSAE Interface Confidence Score ────────────────────────────────────────

class IpSAERequest(BaseModel):
    job_name: str = Field("ipsae_score", description="Job identifier")
    af3_output_dir: str = Field(..., description="Path to AF3 output directory containing *_confidences.json and *_model.cif")
    pae_cutoff: float = Field(10.0, description="PAE cutoff (default 10 for AF3)")
    dist_cutoff: float = Field(10.0, description="Distance cutoff (default 10 for AF3)")


@router.post("/ipsae_score", summary="Calculate ipSAE interface confidence for AF3 complex")
async def ipsae_score(req: IpSAERequest):
    """Calculate ipSAE interface confidence score for AF3 complex.

    Use after AF3 validation to detect false positives — designs with high ipTM
    but ipSAE=0.000 have no real interface. CPU-only, fast (~5s).

    Input: AF3 output directory containing *_confidences.json and *_model.cif.
    Output: ipSAE, ipSAE_d0chn, pDockQ, pDockQ2, LIS per chain pair.
    """
    import glob as _glob
    import logging
    logger = logging.getLogger("oih")

    af3_dir = req.af3_output_dir.rstrip("/")
    if not os.path.isdir(af3_dir):
        raise HTTPException(status_code=400, detail=f"Directory not found: {af3_dir}")

    # Auto-find confidences.json (exclude summary and seed/sample variants)
    conf_files = [
        f for f in _glob.glob(os.path.join(af3_dir, "*_confidences.json"))
        if "summary" not in os.path.basename(f)
    ]
    if not conf_files:
        conf_files = [
            f for f in _glob.glob(os.path.join(af3_dir, "*/*_confidences.json"))
            if "summary" not in os.path.basename(f) and "seed-" not in f
        ]
    if not conf_files:
        raise HTTPException(status_code=400, detail=f"No *_confidences.json found in {af3_dir}")

    # Auto-find model.cif (main model, not seed samples)
    cif_files = [
        f for f in _glob.glob(os.path.join(af3_dir, "*_model.cif"))
        if "seed-" not in f and "sample-" not in f
    ]
    if not cif_files:
        cif_files = [
            f for f in _glob.glob(os.path.join(af3_dir, "*/*_model.cif"))
            if "seed-" not in f and "sample-" not in f
        ]
    if not cif_files:
        raise HTTPException(status_code=400, detail=f"No *_model.cif found in {af3_dir}")

    conf_path = conf_files[0]
    cif_path = cif_files[0]

    # Output directory: try original dir first, fallback to /tmp
    try:
        test_file = os.path.join(af3_dir, ".ipsae_write_test")
        with open(test_file, "w") as tf:
            tf.write("test")
        os.remove(test_file)
        out_dir = af3_dir
    except (PermissionError, OSError):
        out_dir = f"/tmp/ipsae_results/{req.job_name}"

    os.makedirs(out_dir, exist_ok=True)

    logger.info("[ipSAE] %s: conf=%s cif=%s out=%s",
                req.job_name, os.path.basename(conf_path),
                os.path.basename(cif_path), out_dir)

    try:
        from ipsae import IpsaeCalculator
        calc = IpsaeCalculator(req.pae_cutoff, req.dist_cutoff)
        result = calc.calculate(conf_path, cif_path, out_dir)
    except Exception as e:
        logger.error("[ipSAE] %s failed: %s", req.job_name, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"ipSAE calculation failed: {e}")

    # Extract chain pair scores into a flat summary
    chains = result["unique_chains"]
    ipsae_scores = result["ipsae_scores"]
    pdockq_scores = result["pdockq_scores"]
    lis_scores = result["lis_scores"]
    global_scores = result.get("global_scores", {})

    chain_pairs = []
    for c1 in chains:
        for c2 in chains:
            if c1 >= c2:
                continue
            chain_pairs.append({
                "chain1": c1,
                "chain2": c2,
                "ipSAE": round(ipsae_scores["ipsae_d0res_asym"][c1][c2], 6),
                "ipSAE_d0chn": round(ipsae_scores["ipsae_d0chn_asym"][c1][c2], 6),
                "pDockQ": round(pdockq_scores["pDockQ"][c1][c2], 4),
                "pDockQ2": round(pdockq_scores["pDockQ2"][c1][c2], 4),
                "LIS": round(lis_scores[c1][c2], 4),
            })

    best = max(chain_pairs, key=lambda x: x["ipSAE"]) if chain_pairs else {}

    logger.info("[ipSAE] %s: best %s-%s ipSAE=%.4f pDockQ2=%.4f LIS=%.4f",
                req.job_name, best.get("chain1", "?"), best.get("chain2", "?"),
                best.get("ipSAE", 0), best.get("pDockQ2", 0), best.get("LIS", 0))

    return {
        "status": "completed",
        "job_name": req.job_name,
        "chain_pairs": chain_pairs,
        "best_pair": best,
        "ipTM": global_scores.get("iptm", None),
        "pTM": global_scores.get("ptm", None),
        "output_dir": out_dir,
        "confidences_file": conf_path,
        "model_cif": cif_path,
    }
