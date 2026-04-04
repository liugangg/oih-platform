"""
Protein Design Router
Tools: RFdiffusion, ProteinMPNN, BindCraft, FreeSASA
"""
# --- SYNC_NOTES (auto-generated from CLAUDE.md, do not edit) ---
# BINDCRAFT+PROTEINMPNN+RFDIFFUSION notes (from CLAUDE.md, do not manually edit):
#   - Container NVIDIA_VISIBLE_DEVICES=1, always use device=0 / gpu_id=0 (not 1)
#   - Timeout must be 7200s (not 3600s)
#   - num_designs use 10 (not 20, too slow)
#   - Hotspots must be spatially clustered (<= 15A centroid); dispersed hotspots cause extreme slowness
# --- /SYNC_NOTES ---

import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from schemas.models import (
    RFdiffusionRequest, RFdiffusionResult,
    ProteinMPNNRequest, ProteinMPNNResult,
    BindCraftRequest, BindCraftResult,
    FreeSASARequest, FreeSASAResult,
    TaskRef,
)
from core.task_manager import task_manager
from core.docker_client import run_in_container_streaming, run_in_container
from core.config import settings

router = APIRouter()


def _renumber_pdb_for_rfdiffusion(pdb_path: str, chain: str = "A") -> tuple[str, int, dict]:
    """
    Renumber protein ATOM records in a PDB chain sequentially (1,2,3,...) to
    eliminate gaps that crash RFdiffusion's ContigMap.
    HETATM records (water, ligands, sugars) are stripped — RFdiffusion only needs ATOM.
    Returns (new_pdb_path, chain_length, old_to_new_map).
    """
    lines = []
    # Collect unique ATOM residue IDs in order of appearance (skip HETATM)
    seen_resids = []
    seen_set = set()
    try:
        with open(pdb_path) as f:
            for line in f:
                lines.append(line)
                if line.startswith("ATOM") and len(line) > 26:
                    ch = line[21].strip()
                    if ch == chain:
                        resid_str = line[22:26].strip()
                        icode = line[26].strip()
                        key = (resid_str, icode)
                        if key not in seen_set:
                            seen_set.add(key)
                            seen_resids.append(key)
    except Exception:
        return pdb_path, 150, {}

    if not seen_resids:
        return pdb_path, 150, {}

    # Build old→new mapping
    old_to_new = {}
    for new_num, (old_resid, icode) in enumerate(seen_resids, 1):
        old_to_new[(old_resid, icode)] = new_num
        if old_resid.isdigit():
            old_to_new[int(old_resid)] = new_num

    # Rewrite PDB: renumber ATOM lines, drop HETATM for clean RFdiffusion input
    out_path = pdb_path.replace(".pdb", "_renum.pdb")
    with open(out_path, "w") as out:
        for line in lines:
            if line.startswith("HETATM"):
                continue  # strip waters/ligands/sugars
            if line.startswith("ATOM") and len(line) > 26:
                ch = line[21].strip()
                if ch == chain:
                    resid_str = line[22:26].strip()
                    icode = line[26].strip()
                    key = (resid_str, icode)
                    new_num = old_to_new.get(key, resid_str)
                    line = line[:22] + f"{new_num:>4}" + " " + line[27:]
                out.write(line)
            else:
                out.write(line)

    chain_len = len(seen_resids)
    return out_path, chain_len, old_to_new


# ─── RFdiffusion ──────────────────────────────────────────────────────────────

@router.post("/rfdiffusion", response_model=TaskRef, summary="De novo protein backbone design with RFdiffusion")
async def run_rfdiffusion(req: RFdiffusionRequest):
    """
    Run RFdiffusion for backbone generation.
    Supports: unconditional, motif scaffolding, partial diffusion, binder design.
    """
    output_dir = os.path.join(settings.OUTPUT_DIR, req.job_name, "rfdiffusion")
    os.makedirs(output_dir, exist_ok=True)

    async def _run(task):
        task.progress = 5
        task.progress_msg = "Building RFdiffusion command..."

        # Base command
        cmd = [
            "python3", "/app/RFdiffusion/scripts/run_inference.py",
            f"inference.output_prefix=/data/oih/outputs/{req.job_name}/rfdiffusion/{req.job_name}",
            "inference.model_directory_path=/data/models/rfdiffusion",
            f"inference.num_designs={req.num_designs}",
            f"diffuser.T={req.num_diffusion_steps}",
        ]

        if req.mode == "binder_design" and req.target_pdb:
            _target = req.target_pdb if os.path.exists(req.target_pdb) else f"/data/oih/inputs/{os.path.basename(req.target_pdb)}"

            if req.contigs:
                cmd.append(f"inference.input_pdb={_target}")
                cmd.append(f"contigmap.contigs=['{req.contigs}']")
                renum_map = {}
            else:
                # Renumber PDB sequentially to eliminate gaps that crash ContigMap
                renum_path, chain_len, renum_map = _renumber_pdb_for_rfdiffusion(_target, chain="A")
                cmd.append(f"inference.input_pdb={renum_path}")
                cmd.append(f"contigmap.contigs=['A1-{chain_len}/0 70-100']")
                import logging
                logging.getLogger(__name__).info(
                    "[RFdiffusion] Renumbered %s → %s (chain_len=%d, gaps_removed=%d)",
                    _target, renum_path, chain_len, len(renum_map) // 2 - chain_len if renum_map else 0,
                )

            if req.hotspot_residues:
                # Normalize hotspot format: accept "S310,F311" or "A310,A311" or "310,311"
                # RFdiffusion expects [A310,A311,...] where A = chain ID (not residue name)
                raw = req.hotspot_residues.split(",")
                normalized = []
                for h in raw:
                    h = h.strip()
                    # Strip leading amino acid 1-letter code if present (e.g. S310 → 310)
                    if len(h) > 1 and h[0].isalpha() and h[1:].isdigit():
                        resnum = int(h[1:])
                    elif h.isdigit():
                        resnum = int(h)
                    else:
                        normalized.append(f"A{h}")
                        continue
                    # Remap to renumbered residue if we renumbered
                    if renum_map and resnum in renum_map:
                        resnum = renum_map[resnum]
                    normalized.append(f"A{resnum}")
                hotspots = "[" + ",".join(normalized) + "]"
                cmd.append(f"ppi.hotspot_res={hotspots}")

        elif req.mode == "motif_scaffolding" and req.motif_pdb:
            _motif = req.motif_pdb if os.path.exists(req.motif_pdb) else f"/data/oih/inputs/{os.path.basename(req.motif_pdb)}"
            cmd += [
                f"inference.input_pdb={_motif}",
                f"contigmap.contigs=['{req.contigs}']",
            ]

        elif req.mode == "unconditional":
            cmd.append(f"contigmap.contigs=['{req.contigs or '150'}']")

        retcode, output = await run_in_container_streaming(
            settings.CONTAINER_RFDIFFUSION, cmd, task,
            timeout=settings.TIMEOUT_RFDIFFUSION
        )

        if retcode != 0:
            raise RuntimeError(f"RFdiffusion failed (exit {retcode})\n{output[-500:]}")

        pdb_files = [str(p) for p in Path(output_dir).glob("*.pdb")]
        task.output_files = pdb_files
        task.progress_msg = f"Done. {len(pdb_files)} backbone(s) generated."
        return {"pdb_files": pdb_files, "output_dir": output_dir}

    task = await task_manager.submit("rfdiffusion", req.model_dump(), _run)
    return TaskRef(task_id=task.task_id, status=task.status, tool="rfdiffusion",
                   poll_url=f"/api/v1/tasks/{task.task_id}")


# ─── ProteinMPNN ──────────────────────────────────────────────────────────────

@router.post("/proteinmpnn", response_model=TaskRef, summary="Sequence design with ProteinMPNN")
async def run_proteinmpnn(req: ProteinMPNNRequest):
    """
    Run ProteinMPNN to design amino acid sequences for a given backbone.
    Typically used after RFdiffusion.
    """
    output_dir = os.path.join(settings.OUTPUT_DIR, req.job_name, "proteinmpnn")
    os.makedirs(output_dir, exist_ok=True)

    async def _run(task):
        task.progress = 5
        task.progress_msg = "Running ProteinMPNN sequence design..."

        _pdb = req.input_pdb if os.path.exists(req.input_pdb) else f"/data/oih/inputs/{os.path.basename(req.input_pdb)}"

        # ── Auto-detect binder chain (shortest chain = binder) ──────────
        # 2026-03-24 blood lesson: hardcoding 'A' redesigned the 400aa
        # target instead of the 80aa binder for CD36/EGFR/Trop2/Nectin-4.
        _chain = req.chains_to_design
        if _chain == "auto":
            try:
                import gemmi
                st = gemmi.read_structure(_pdb)
                chains = [
                    (c.name, sum(1 for r in c if r.entity_type == gemmi.EntityType.Polymer))
                    for c in st[0]
                ]
                chains.sort(key=lambda x: x[1])
                _chain = chains[0][0]  # shortest = binder
                import logging
                logging.getLogger("oih").info(
                    "[MPNN] Auto-detected binder chain: %s (%d res) from %s",
                    _chain, chains[0][1], os.path.basename(_pdb))
            except Exception:
                _chain = "A"  # fallback

        cmd = [
            "python3", "/app/ProteinMPNN/protein_mpnn_run.py",
            "--pdb_path", _pdb,
            "--out_folder", f"/data/oih/outputs/{req.job_name}/proteinmpnn",
            "--num_seq_per_target", str(req.num_sequences),
            "--sampling_temp", str(req.sampling_temp),
            "--pdb_path_chains", _chain,
            "--batch_size", "1",
        ]
        if req.fixed_residues:
            cmd += ["--fixed_residues", req.fixed_residues]
        if req.use_soluble_model:
            cmd.append("--use_soluble_model")

        retcode, output = await run_in_container_streaming(
            settings.CONTAINER_PROTEINMPNN, cmd, task,
            timeout=settings.TIMEOUT_PROTEINMPNN
        )

        if retcode != 0:
            raise RuntimeError(f"ProteinMPNN failed (exit {retcode})")

        fasta_files = list(Path(output_dir).glob("**/*.fa")) + list(Path(output_dir).glob("**/*.fasta"))
        fasta_path = str(fasta_files[0]) if fasta_files else ""
        task.output_files = [fasta_path]
        task.progress_msg = f"Done. Sequences written to {fasta_path}"
        return {"fasta_file": fasta_path, "output_dir": output_dir}

    task = await task_manager.submit("proteinmpnn", req.model_dump(), _run)
    return TaskRef(task_id=task.task_id, status=task.status, tool="proteinmpnn",
                   poll_url=f"/api/v1/tasks/{task.task_id}")


# ─── BindCraft ────────────────────────────────────────────────────────────────

@router.post("/bindcraft", response_model=TaskRef, summary="End-to-end binder design with BindCraft")
async def run_bindcraft(req: BindCraftRequest):
    """
    Run BindCraft for complete binder design pipeline:
    hallucination → filtering → ranking.
    Returns top binders with predicted binding scores.
    """
    job_dir = os.path.join(settings.OUTPUT_DIR, req.job_name, "bindcraft")
    os.makedirs(job_dir, exist_ok=True)

    import json, tempfile
    settings_dict = {
        "design_path": f"/data/oih/outputs/{req.job_name}/bindcraft",
        "target_pdb": req.target_pdb if os.path.exists(req.target_pdb) else f"/data/oih/inputs/{os.path.basename(req.target_pdb)}",
        "hotspot_res": req.target_hotspots or "",
        "lengths": [70, 120],
        "number_of_final_designs": req.num_designs,
    }
    if req.advanced_settings:
        settings_dict.update(req.advanced_settings)

    settings_path = os.path.join(settings.INPUT_DIR, req.job_name, "bindcraft_settings.json")
    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    with open(settings_path, "w") as f:
        json.dump(settings_dict, f)

    filters_path = os.path.join(settings.INPUT_DIR, req.job_name, "filters.json")
    with open(filters_path, "w") as f:
        json.dump(req.filters or {}, f)

    async def _run(task):
        task.progress = 5
        task.progress_msg = "Launching BindCraft..."

        cmd = [
            "bash", "-c",
            "cd /app/BindCraft && python3 -u bindcraft.py"
            f" --settings '/data/oih/inputs/{req.job_name}/bindcraft_settings.json'"
            f" --filters '/data/oih/inputs/{req.job_name}/filters.json'"
            " --advanced './settings_advanced/default_4stage_multimer.json'",
        ]

        retcode, output = await run_in_container_streaming(
            settings.CONTAINER_BINDCRAFT, cmd, task,
            timeout=settings.TIMEOUT_BINDCRAFT
        )

        # BindCraft may exit 1 if 0 designs pass filters (normal for hard targets)
        csv_files = list(Path(job_dir).glob("*.csv"))
        pdb_files = [str(p) for p in Path(job_dir).glob("final_designs/**/*.pdb")]
        # Also check Accepted/ and Ranked/ dirs for PDBs
        for subdir in ["Accepted", "Ranked", "MPNN"]:
            pdb_files.extend(str(p) for p in Path(job_dir).glob(f"{subdir}/**/*.pdb"))
        pdb_files = list(set(pdb_files))  # deduplicate

        if retcode != 0 and not pdb_files and not csv_files:
            raise RuntimeError(f"BindCraft failed (exit {retcode}): {output[-2000:]}")

        task.output_files = [str(c) for c in csv_files] + pdb_files
        if pdb_files:
            task.progress_msg = f"Done. {len(pdb_files)} designs generated."
        else:
            task.progress_msg = f"BindCraft completed but 0 designs passed filters (exit {retcode})."

        return {
            "csv_file": str(csv_files[0]) if csv_files else "",
            "pdb_files": pdb_files,
            "output_dir": job_dir,
            "n_designs": len(pdb_files),
        }

    task = await task_manager.submit("bindcraft", req.model_dump(), _run)
    return TaskRef(task_id=task.task_id, status=task.status, tool="bindcraft",
                   poll_url=f"/api/v1/tasks/{task.task_id}")


# ─── FreeSASA ────────────────────────────────────────────────────────────────

@router.post("/freesasa", response_model=TaskRef, summary="Calculate SASA and recommend ADC conjugation sites")
async def run_freesasa(req: FreeSASARequest):
    """
    Compute per-residue SASA with FreeSASA, then filter Lys (K) and Cys (C)
    residues with SASA > 40 Å² as recommended conjugation sites for ADC design.
    """
    _pdb = req.input_pdb if os.path.exists(req.input_pdb) else f"/data/oih/inputs/{os.path.basename(req.input_pdb)}"
    if not os.path.exists(_pdb):
        raise HTTPException(status_code=400, detail=f"PDB file not found: {_pdb}")

    async def _run(task):
        import freesasa

        task.progress = 10
        task.progress_msg = "Computing SASA with FreeSASA..."

        structure = freesasa.Structure(_pdb)
        result = freesasa.calc(structure)

        # Iterate residues, collect exposed Lys/Cys
        SASA_THRESHOLD = 40.0
        TARGET_RESIDUES = {"LYS", "CYS"}
        ONE_LETTER = {"LYS": "K", "CYS": "C"}

        conjugation_sites = []
        residue_areas = result.residueAreas()
        # residueAreas() returns {chain_id: {res_num_str: ResidueArea}}
        for chain_id, residues in residue_areas.items():
            for res_num, ra in residues.items():
                if ra.residueType in TARGET_RESIDUES and ra.total > SASA_THRESHOLD:
                    conjugation_sites.append({
                        "residue": f"{ONE_LETTER[ra.residueType]}{res_num}",
                        "chain": chain_id,
                        "sasa": round(ra.total, 1),
                    })

        # Sort by SASA descending
        conjugation_sites.sort(key=lambda x: x["sasa"], reverse=True)

        task.progress = 100
        task.progress_msg = f"Done. Found {len(conjugation_sites)} conjugation site(s)."
        return {"conjugation_sites": conjugation_sites}

    task = await task_manager.submit("freesasa", req.model_dump(), _run)
    return TaskRef(task_id=task.task_id, status=task.status, tool="freesasa",
                   poll_url=f"/api/v1/tasks/{task.task_id}")
