"""
Full Pipeline Router
Orchestrates multi-tool workflows that Qwen can trigger as single calls
"""
import asyncio, json, os, re
import numpy as np
from fastapi import APIRouter, HTTPException
from schemas.models import (
    DrugDiscoveryPipelineRequest, BinderDesignPipelineRequest,
    PocketGuidedBinderPipelineRequest,
    AlphaFold3Request, AF3Chain, RFdiffusionRequest, ProteinMPNNRequest,
    FpocketRequest, P2RankRequest, DiffDockRequest,
    DockingRequest, DockingEngine, GROMACSRequest, GromacsPreset,
    FreeSASARequest, LinkerSelectRequest, RDKitConjugateRequest,
    DiscoTope3Request,
    TaskRef,
)
from core.task_manager import task_manager, TaskStatus
from core.config import settings
import httpx
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================
# AF3 Antigen Domain Registry
# Literature-based: sequence truncation preserves complete domains, avoids disrupting disulfide bonds and folding units
# Principles:
#   1. Truncate by UniProt/Pfam domain boundaries, not by residue count
#   2. Multiple druggable domains → run AF3 separately for each, validate independently
#   3. Unknown proteins → use hotspot residues to locate domain boundaries
#   4. padding=30aa preserves boundary flexible regions
# Reference: Yin et al. Protein Science 2024; PMC12360200 (AF3 nanobody benchmark)
# ============================================================

DOMAIN_REGISTRY = {
    # HER2 / ERBB2 (UniProt P04626)
    "HER2": {
        "uniprot": "P04626",
        "signal_peptide_offset": 22,
        "druggable_domains": [
            {
                "name": "domain2",
                "range": (172, 308),
                "hotspot_center": 240,
                "description": "Dimerization arm, pertuzumab epitope",
            },
            {
                "name": "domain4",
                "range": (488, 630),
                "hotspot_center": 565,
                "description": "Trastuzumab epitope, C558-C573",
            },
        ],
    },
    # CD36 (UniProt P16671) — 3 sub-domains for targeted AF3
    "CD36": {
        "uniprot": "P16671",
        "signal_peptide_offset": 0,
        "druggable_domains": [
            {
                "name": "pesto_ppi_core",
                "range": (140, 240),
                "hotspot_center": 191,
                "description": "PeSTo PPI interface core (L187-P193, score 0.50-0.87) + buffer",
            },
            {
                "name": "clesh_domain",
                "range": (50, 180),
                "hotspot_center": 106,
                "description": "CLESH domain (TSP-1 binding, E101/D106/E108/D109, mutagenesis validated)",
            },
            {
                "name": "extracellular_full",
                "range": (30, 439),
                "hotspot_center": 200,
                "description": "Full extracellular domain (fallback only)",
            },
        ],
    },
    # EGFR (UniProt P00533)
    "EGFR": {
        "uniprot": "P00533",
        "signal_peptide_offset": 24,
        "druggable_domains": [
            {
                "name": "domain1",
                "range": (25, 189),
                "hotspot_center": 100,
                "description": "Ligand binding domain I",
            },
            {
                "name": "domain3",
                "range": (361, 481),
                "hotspot_center": 420,
                "description": "Ligand binding domain III, cetuximab epitope",
            },
        ],
    },
    # PD-L1 (UniProt Q9NZQ7)
    "PDL1": {
        "uniprot": "Q9NZQ7",
        "signal_peptide_offset": 18,
        "druggable_domains": [
            {
                "name": "ig_v_domain",
                "range": (19, 127),
                "hotspot_center": 73,
                "description": "IgV domain, PD-1 binding interface",
            },
        ],
    },
    # Nectin-4 (UniProt Q96NY8)
    "NECTIN4": {
        "uniprot": "Q96NY8",
        "signal_peptide_offset": 0,
        "druggable_domains": [
            {
                "name": "d1_d2",
                "range": (32, 240),
                "hotspot_center": 212,
                "description": "Ig-like D1+D2 domains, PeSTo L208/R212/Y214 (score 0.95-0.97)",
            },
        ],
    },
    # TROP2 (UniProt P09758)
    "TROP2": {
        "uniprot": "P09758",
        "signal_peptide_offset": 0,
        "druggable_domains": [
            {
                "name": "ectodomain",
                "range": (30, 274),
                "hotspot_center": 150,
                "description": "Full ectodomain (PeSTo max=0.42, flat surface — difficult target)",
            },
        ],
    },
    # TrkA (UniProt P04629)
    "TRKA": {
        "uniprot": "P04629",
        "signal_peptide_offset": 0,
        "druggable_domains": [
            {
                "name": "d5_domain",
                "range": (280, 400),
                "hotspot_center": 302,
                "description": "NGF binding D5 domain, PeSTo P302/C300/S304 (score 0.999)",
            },
            {
                "name": "ectodomain",
                "range": (34, 423),
                "hotspot_center": 300,
                "description": "Full ECD (fallback)",
            },
        ],
    },
}

DOMAIN_PADDING = 30  # Retain 30aa padding on each side of domain boundary


def _get_af3_antigen_regions(
    target_name: str,
    target_seq: str,
    hotspot_residues: list = None,
) -> list:
    """
    Return list of antigen domain regions for AF3 validation.

    Known proteins: extract domain boundaries from DOMAIN_REGISTRY
    Unknown proteins: use hotspot_residues to locate the minimal complete region containing hotspots
    Multiple domains: return multiple dicts; pipeline submits separate AF3 tasks for each domain

    Return format:
    [{"domain_name": str, "sequence": str, "offset": int, "description": str, "original_range": tuple}, ...]
    """
    regions = []

    # Look up known proteins (case-insensitive)
    registry_key = None
    for key in DOMAIN_REGISTRY:
        if key.upper() in target_name.upper():
            registry_key = key
            break

    # Parse hotspot residue numbers
    hotspot_nums = []
    if hotspot_residues:
        for h in hotspot_residues:
            try:
                hotspot_nums.append(int(re.sub(r'[A-Za-z_:]', '', str(h))))
            except (ValueError, TypeError):
                pass

    if registry_key:
        entry = DOMAIN_REGISTRY[registry_key]
        offset_correction = entry.get("signal_peptide_offset", 0)

        for domain in entry["druggable_domains"]:
            start, end = domain["range"]

            if hotspot_nums:
                # Check if hotspot falls within this domain (accounting for signal peptide offset)
                adjusted_hotspots = [h - offset_correction for h in hotspot_nums]
                if not any(start - 20 <= h <= end + 20 for h in adjusted_hotspots):
                    continue

            # Add padding, clamped to sequence boundaries
            seq_start = max(0, start - DOMAIN_PADDING - offset_correction)
            seq_end = min(len(target_seq), end + DOMAIN_PADDING - offset_correction)

            regions.append({
                "domain_name": domain["name"],
                "sequence": target_seq[seq_start:seq_end],
                "offset": seq_start,
                "description": domain["description"],
                "original_range": (start, end),
            })

    else:
        # Unknown protein: locate region using hotspots
        if hotspot_nums:
            center = sum(hotspot_nums) // len(hotspot_nums)
            seq_start = max(0, center - 100)
            seq_end = min(len(target_seq), center + 100)
        else:
            seq_start, seq_end = 0, len(target_seq)

        regions.append({
            "domain_name": "unknown_region",
            "sequence": target_seq[seq_start:seq_end],
            "offset": seq_start,
            "description": f"Auto-detected region around center {center if hotspot_nums else 'full'}",
            "original_range": (seq_start, seq_end),
        })

    return regions if regions else [{
        "domain_name": "full_length",
        "sequence": target_seq,
        "offset": 0,
        "description": "Full length (no domain match found)",
        "original_range": (0, len(target_seq)),
    }]


def _make_dry_run_coroutine(mock_result: dict):
    """Return an async callable that immediately returns mock_result."""
    async def _dry(task):
        task.progress = 100
        task.progress_msg = "dry_run complete"
        return mock_result
    return _dry


async def _wait_for_task(task_id: str, timeout: int = 7200) -> dict:
    """Block until task completes or fails.

    If timeout <= 0, wait indefinitely (used for AF3 tasks).
    """
    elapsed = 0
    while timeout <= 0 or elapsed < timeout:
        task = task_manager.get_task(task_id)
        if task.status == TaskStatus.COMPLETED:
            return task.result
        elif task.status == TaskStatus.FAILED:
            raise RuntimeError(f"Sub-task {task_id} failed: {task.error}")
        elif task.status == TaskStatus.CANCELLED:
            raise RuntimeError(f"Sub-task {task_id} was cancelled")
        await asyncio.sleep(10)
        elapsed += 10
    raise TimeoutError(f"Sub-task {task_id} timed out after {timeout}s")


async def _wait_for_af3_task(task_id: str) -> dict:
    """Wait for AF3 task with unlimited patience.

    Polls every 30s. Only fails on:
      - status == failed AND error contains OOM/exit 1/out of memory
      - status == cancelled
      - 10 consecutive polls returning the same non-timeout error
    Keeps waiting if status is running/pending (no upper time limit).
    """
    last_error = None
    same_error_count = 0

    while True:
        task = task_manager.get_task(task_id)
        if task.status == TaskStatus.COMPLETED:
            return task.result
        elif task.status == TaskStatus.CANCELLED:
            raise RuntimeError(f"AF3 task {task_id} was cancelled")
        elif task.status == TaskStatus.FAILED:
            err = task.error or ""
            err_lower = err.lower()
            # Fatal errors: OOM, process crash, or timeout (no point retrying)
            if any(k in err_lower for k in ("oom", "out of memory", "exit 1", "exit code", "timed out", "timeout")):
                raise RuntimeError(f"AF3 task {task_id} failed (fatal): {err}")
            # Track repeated non-fatal errors
            if err == last_error:
                same_error_count += 1
            else:
                last_error = err
                same_error_count = 1
            if same_error_count >= 10:
                raise RuntimeError(f"AF3 task {task_id} failed (10 consecutive same error): {err}")
            # Otherwise keep waiting — might be transient
            logger.info("[AF3 wait] task %s status=failed err=%s (count=%d/10), retrying...",
                        task_id[:8], err[:80], same_error_count)
        # running / pending — keep waiting
        await asyncio.sleep(30)


# ─── Pure helper functions (stateless, fully testable) ────────────────────────

_AA3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def _parse_mpnn_fasta(fasta_path: str) -> list:
    """Parse ProteinMPNN FASTA output into list of {sequence, score} dicts.

    Format:
      >header, score=0.8220, global_score=1.4925, ...
      SEQUENCE

    Skips the first entry (original/fixed sequence) and returns designed sequences.
    """
    results = []
    try:
        with open(fasta_path) as f:
            lines = f.read().strip().split("\n")
        i = 0
        while i < len(lines):
            if lines[i].startswith(">"):
                header = lines[i]
                seq = lines[i + 1] if i + 1 < len(lines) else ""
                i += 2
                # Skip the first entry (original sequence from the PDB)
                if "sample=" not in header and "T=" not in header:
                    continue
                # Extract score from header
                score = None
                for part in header.split(","):
                    part = part.strip()
                    if part.startswith("score="):
                        try:
                            score = float(part.split("=")[1])
                        except ValueError:
                            pass
                        break
                results.append({"sequence": seq, "score": score, "header": header.lstrip(">")})
            else:
                i += 1
    except Exception as e:
        logger.warning("_parse_mpnn_fasta(%s) failed: %s", fasta_path, e)
    return results


def _extract_sequence_from_pdb(pdb_path: str, chain_id: str = None) -> str:
    """Extract protein sequence from PDB ATOM records (CA atoms).

    Args:
        pdb_path: Path to PDB file.
        chain_id: Specific chain to extract. If None, extracts the FIRST chain
                  encountered (not all chains concatenated).

    2026-03-27 bug fix: old code iterated all chains without filtering,
    concatenating target(400aa) + binder(69aa) = 469aa as one sequence.
    Now strictly extracts one chain at a time.
    """
    seq = []
    seen = set()
    first_chain = None
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    chain = line[21]
                    # Lock to first chain encountered, or user-specified chain
                    if first_chain is None:
                        first_chain = chain_id or chain
                    if chain != first_chain:
                        continue
                    resname = line[17:20].strip()
                    resi = line[22:27].strip()
                    key = (chain, resi)
                    if key not in seen:
                        seen.add(key)
                        seq.append(_AA3TO1.get(resname, "X"))
    except Exception as e:
        logger.warning("_extract_sequence_from_pdb(%s, chain=%s) failed: %s",
                       pdb_path, chain_id, e)
    return "".join(seq)


# ─── Pocket scoring helpers ──────────────────────────────────────────────────

_CHARGED_RESIDUES = {"K", "R", "E", "D", "H"}


def _parse_p2rank_residues(residue_ids_raw: str) -> list[str]:
    """Parse P2Rank residue_ids string into normalised ['A42', 'A45', ...] list."""
    residues = []
    for token in residue_ids_raw.split():
        token = token.strip()
        if not token:
            continue
        for sep in ("_", ":"):
            if sep in token:
                parts = token.split(sep)
                if len(parts) == 2:
                    chain, resnum = parts
                    residues.append(f"{chain}{resnum}")
                break
        else:
            residues.append(token)
    return residues


def _compute_bfactor_conservation(pdb_path: str, pocket_residues: list[str]) -> float:
    """Compute conservation score from B-factors of pocket residues.

    conservation_score = 1 - mean_normalised_bfactor
    Low B-factor → rigid → conserved → better target → higher score.
    """
    all_bfactors = []
    pocket_bfactors = []
    # Collect chain+resnum → set for quick lookup
    pocket_set = set(pocket_residues)
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    bfactor = float(line[60:66].strip())
                    chain = line[21]
                    resnum = line[22:27].strip()
                    key = f"{chain}{resnum}"
                    all_bfactors.append(bfactor)
                    if key in pocket_set:
                        pocket_bfactors.append(bfactor)
    except Exception as e:
        logger.warning("_compute_bfactor_conservation failed: %s", e)
        return 0.5  # neutral fallback

    if not all_bfactors or not pocket_bfactors:
        return 0.5

    b_min = min(all_bfactors)
    b_max = max(all_bfactors)
    b_range = b_max - b_min if b_max > b_min else 1.0
    normalised = [(b - b_min) / b_range for b in pocket_bfactors]
    mean_norm = sum(normalised) / len(normalised)
    return round(1.0 - mean_norm, 4)



def _compute_electrostatics_from_pdb(pdb_path: str, pocket_residues: list[str]) -> float:
    """Fraction of charged residues (K/R/E/D/H) among pocket residues."""
    pocket_set = set(pocket_residues)
    total = 0
    charged = 0
    seen = set()
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    chain = line[21]
                    resnum = line[22:27].strip()
                    key = f"{chain}{resnum}"
                    if key in pocket_set and key not in seen:
                        seen.add(key)
                        resname = line[17:20].strip()
                        aa = _AA3TO1.get(resname, "X")
                        total += 1
                        if aa in _CHARGED_RESIDUES:
                            charged += 1
    except Exception as e:
        logger.warning("_compute_electrostatics_from_pdb failed: %s", e)
    return round(charged / total, 4) if total else 0.0


def _compute_sasa_score_for_pocket(sasa_per_residue: dict, pocket_residues: list[str]) -> float:
    """Mean SASA of pocket residues / 150.0, capped at 1.0.

    Higher SASA = more exposed = better for protein binder.
    sasa_per_residue: {chain_resnum: total_sasa} e.g. {'A42': 85.3, ...}
    """
    values = [sasa_per_residue[r] for r in pocket_residues if r in sasa_per_residue]
    if not values:
        return 0.0
    mean_sasa = sum(values) / len(values)
    return round(min(mean_sasa / 150.0, 1.0), 4)


def _cluster_hotspots(residue_keys: list[str], pdb_path: str, max_dist: float = 15.0, max_n: int = 5) -> list[str]:
    """Spatially cluster hotspot residues: keep the largest cluster within max_dist of centroid.

    residue_keys: ['A245', 'A246', ...] — chain+resnum format
    pdb_path:     target PDB for coordinate lookup
    Returns:      filtered list, max max_n residues from tightest cluster
    """
    if len(residue_keys) <= max_n:
        return residue_keys

    # Parse CA coordinates
    ca_coords = {}
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    chain = line[21]
                    resnum = line[22:27].strip()
                    key = f"{chain}{resnum}"
                    if key in set(residue_keys):
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        ca_coords[key] = (x, y, z)
    except Exception as e:
        logger.warning("_cluster_hotspots: PDB parse failed: %s", e)
        return residue_keys[:max_n]

    if len(ca_coords) < 2:
        return residue_keys[:max_n]

    # Compute centroid
    xs = [c[0] for c in ca_coords.values()]
    ys = [c[1] for c in ca_coords.values()]
    zs = [c[2] for c in ca_coords.values()]
    cx, cy, cz = sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)

    # Sort by distance to centroid, filter by max_dist
    dists = {}
    for key, (x, y, z) in ca_coords.items():
        dists[key] = ((x-cx)**2 + (y-cy)**2 + (z-cz)**2) ** 0.5

    sorted_res = sorted(dists.keys(), key=lambda r: dists[r])
    clustered = [r for r in sorted_res if dists[r] <= max_dist][:max_n]

    if not clustered:
        clustered = sorted_res[:max_n]

    logger.info("_cluster_hotspots: %d → %d (centroid dist: %.1f-%.1f Å, cutoff=%.0f Å)",
                len(residue_keys), len(clustered),
                min(dists.values()), max(dists[r] for r in clustered),
                max_dist)
    return clustered


def _multi_cluster_hotspots(residue_keys: list, pdb_path: str,
                            distance_threshold: float = 15.0, max_per_cluster: int = 5) -> list:
    """Split hotspot residues into multiple spatial clusters.

    Uses greedy agglomerative clustering: residues within distance_threshold
    of any member are joined. Returns list of clusters, each a list of residue keys.
    Single-residue clusters are merged into the nearest valid cluster.

    Example: [['A164','A166'], ['A397','A398','A400']]
    """
    if not residue_keys or len(residue_keys) <= 3:
        return [residue_keys]

    # Parse CA coordinates from PDB
    ca_coords = {}
    try:
        with open(pdb_path) as f:
            res_set = set(residue_keys)
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    chain = line[21]
                    resnum = line[22:27].strip()
                    key = f"{chain}{resnum}"
                    if key in res_set:
                        ca_coords[key] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    except Exception:
        return [residue_keys]

    if len(ca_coords) < 2:
        return [residue_keys]

    def _dist(a, b):
        return sum((ai - bi) ** 2 for ai, bi in zip(a, b)) ** 0.5

    # Greedy clustering: seed from first unassigned, grow by distance
    keys = list(ca_coords.keys())
    used = set()
    clusters = []

    for seed in keys:
        if seed in used:
            continue
        cluster = [seed]
        used.add(seed)
        # Expand: add any unassigned residue within threshold of ANY cluster member
        changed = True
        while changed:
            changed = False
            for other in keys:
                if other in used:
                    continue
                if any(_dist(ca_coords[other], ca_coords[m]) <= distance_threshold for m in cluster):
                    cluster.append(other)
                    used.add(other)
                    changed = True
        clusters.append(cluster[:max_per_cluster])

    # Merge single-residue clusters into nearest valid cluster
    valid = [c for c in clusters if len(c) >= 2]
    singles = [c[0] for c in clusters if len(c) == 1]
    if singles and valid:
        for s in singles:
            # Find nearest valid cluster
            best_idx, best_dist = 0, float('inf')
            for ci, cl in enumerate(valid):
                for m in cl:
                    d = _dist(ca_coords.get(s, (0, 0, 0)), ca_coords.get(m, (0, 0, 0)))
                    if d < best_dist:
                        best_dist = d
                        best_idx = ci
            if len(valid[best_idx]) < max_per_cluster:
                valid[best_idx].append(s)
    elif singles:
        valid = [singles[:max_per_cluster]]

    logger.info("_multi_cluster_hotspots: %d residues → %d clusters: %s",
                len(residue_keys), len(valid), [len(c) for c in valid])
    return valid if valid else [residue_keys[:max_per_cluster]]


def _compute_epitope_score_for_pocket(
    epitope_residues: list[dict],
    pocket_residues: list[str],
    pdb_path: str,
    score_threshold: float = 0.5,
    distance_cutoff: float = 8.0,
) -> float:
    """Fraction of pocket residues within 8 Å of any DiscoTope3 epitope residue.

    epitope_residues: list of dicts from DiscoTope3 CSV (chain, res_id, DiscoTope-3.0_score, ...)
    pocket_residues:  ['A42', 'A45', ...] from P2Rank
    pdb_path:         target PDB for coordinate lookup
    Returns:          float 0.0–1.0
    """
    if not epitope_residues or not pocket_residues:
        return 0.0

    # Collect epitope residue keys (chain+resid) that exceed threshold
    epi_keys = set()
    for e in epitope_residues:
        try:
            if float(e.get("DiscoTope-3.0_score", 0)) >= score_threshold:
                epi_keys.add(f"{e['chain']}{e['res_id']}")
        except (ValueError, KeyError):
            continue

    if not epi_keys:
        return 0.0

    # Parse CA coordinates from PDB
    ca_coords = {}  # chain+resnum → (x, y, z)
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    chain = line[21]
                    resnum = line[22:27].strip()
                    key = f"{chain}{resnum}"
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ca_coords[key] = (x, y, z)
    except Exception as e:
        logger.warning("_compute_epitope_score_for_pocket: PDB parse failed: %s", e)
        return 0.0

    # Build epitope coordinate set
    epi_coords = []
    for k in epi_keys:
        if k in ca_coords:
            epi_coords.append(ca_coords[k])

    if not epi_coords:
        return 0.0

    # Count pocket residues within distance_cutoff of any epitope residue
    pocket_set = set(pocket_residues)
    near_count = 0
    total = 0
    cutoff_sq = distance_cutoff ** 2
    for pr in pocket_set:
        if pr not in ca_coords:
            continue
        total += 1
        px, py, pz = ca_coords[pr]
        for ex, ey, ez in epi_coords:
            dsq = (px - ex) ** 2 + (py - ey) ** 2 + (pz - ez) ** 2
            if dsq <= cutoff_sq:
                near_count += 1
                break

    return round(near_count / total, 4) if total else 0.0


def _compute_rag_score(rag_residues: list[str], pocket_residues: list[str]) -> float:
    """Score pocket by overlap with literature-mentioned residues.

    Normalizes both to plain residue numbers (strip chain letters) for comparison.
    Returns: 0.0 (no overlap), 0.5 (1-2 residues), 1.0 (3+ residues)
    """
    if not rag_residues:
        return 0.0
    # Normalize: 'A42' → '42', '310' → '310'
    def _to_num(r: str) -> str:
        return r.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_:")
    pocket_nums = set(_to_num(r) for r in pocket_residues)
    rag_nums = set(_to_num(r) for r in rag_residues)
    overlap = len(pocket_nums & rag_nums)
    if overlap >= 3:
        return 1.0
    elif overlap >= 1:
        return 0.5
    return 0.0


def _extract_residue_numbers_from_text(text: str) -> list[str]:
    """Extract residue references from RAG text.

    Matches patterns like: residue 42, Arg42, R42, A:42, position 310,
    S310, T311, etc. Returns normalised list like ['42', '310', '311'].
    """
    patterns = [
        r'(?:residue|position|res\.?)\s*(\d+)',
        r'[A-Z](\d{2,4})',   # e.g. S310, R42
        r'[A-Z][_:](\d+)',   # e.g. A:42, A_310
    ]
    numbers = set()
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            numbers.add(m.group(1))
    return sorted(numbers)


def _compute_pesto_score_for_pocket(pdb_path: str, pocket_residues: list, pesto_cache: dict = None) -> float:
    """Compute PeSTo PPI interface score for pocket residues.

    Runs PeSTo model inside oih-proteinmpnn container (CPU only).
    Returns: mean PPI score of pocket residues (0.0-1.0).
    Uses cache to avoid re-running PeSTo on same PDB.
    """
    import subprocess

    # Check cache
    if pesto_cache is not None and pdb_path in pesto_cache:
        scores = pesto_cache[pdb_path]
    else:
        # Run PeSTo in container — returns {pdb_resid: score} with proper residue numbering
        script = (
            "import sys,json,torch as pt,numpy as np,gemmi;"
            "sys.path.insert(0,'/app/pesto');"
            "from config import config_model;from model import Model;"
            "from src.dataset import StructuresDataset,collate_batch_features;"
            "from src.data_encoding import encode_structure,encode_features,extract_topology;"
            "from src.structure import concatenate_chains;"
            "m=Model(config_model);m.load_state_dict(pt.load('/app/pesto/model.pt',map_location='cpu'));m.eval();"
            f"ds=StructuresDataset(['{pdb_path}'],with_preprocessing=True);"
            "result={};"
            "with pt.no_grad():\n"
            " for su,fp in ds:\n"
            "  st=concatenate_chains(su);X,M=encode_structure(st);q=encode_features(st)[0];"
            "  ids,_,_,_,_=extract_topology(X,64);X,ids,q,M=collate_batch_features([[X,ids,q,M]]);"
            "  z=m(X,ids,q,M.float());p=pt.sigmoid(z[:,0]).cpu().numpy();\n"
            f"  gs=gemmi.read_structure('{pdb_path}');ch=gs[0][0];residues=list(ch);"
            "  result={f'{ch.name}{residues[i].seqid.num}':float(p[i]) for i in range(min(len(p),len(residues)))};"
            "  break\n"
            "print(json.dumps(result))"
        )
        try:
            proc = subprocess.run(
                ["docker", "exec", "-w", "/app/pesto", "oih-proteinmpnn", "python3", "-c", script],
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode != 0:
                logger.warning("PeSTo failed: %s", proc.stderr[:200])
                return 0.0
            import json as _json
            scores = _json.loads(proc.stdout.strip())
            if pesto_cache is not None:
                pesto_cache[pdb_path] = scores
        except Exception as e:
            logger.warning("PeSTo error: %s", e)
            return 0.0

    # scores dict is now keyed by PDB residue ID (e.g. "A300", "A302")
    # pocket_residues are like ['A42', 'A45'] — direct lookup
    if not scores:
        return 0.0

    all_scores = list(scores.values())

    if pocket_residues:
        # Direct lookup: pocket residues match PeSTo keys (both use chain+resid)
        matched = [scores.get(r, 0) for r in pocket_residues if r in scores]
        if matched:
            return round(float(np.mean(matched)), 4)
        # Fallback: try without chain letter
        for r in pocket_residues:
            num = re.sub(r'[A-Za-z_:]', '', str(r))
            for key, val in scores.items():
                if num in key:
                    matched.append(val)
                    break
        if matched:
            return round(float(np.mean(matched)), 4)

    # No pocket specified: return mean of top quartile (whole-protein PPI signal)
    sorted_scores = sorted(all_scores, reverse=True)
    top_quarter = sorted_scores[:max(1, len(sorted_scores) // 4)]
    return round(float(np.mean(top_quarter)) if top_quarter else 0, 4)


async def _rag_search_pocket_context(target_name: str, pdb_id: str) -> dict:
    """Query RAG system for binding site / epitope literature.

    Uses multi-query strategy to maximize coverage:
      Q1: protein-protein interaction interface + known binders
      Q2: crystal structure complex + co-crystal
      Q3: domain binding + functional residues + mutagenesis

    Returns: {text: str, residues: list[str], domains: list[str], kd_values: list[str]}
    """
    # Layer 1: PPI interfaces + co-crystal + mutagenesis (highest priority)
    # These give experimentally validated binding residues — far more reliable
    # than epitope predictions for binder design.
    primary_queries = [
        f"{target_name} protein-protein interaction co-crystal structure complex",
        f"{target_name} known ligand binding partner domain interaction residues",
        f"{target_name} receptor ligand interface mutagenesis",
    ]
    # Layer 2 (fallback only if Layer 1 finds no residues): epitope/antibody
    fallback_queries = [
        f"{target_name} epitope antibody binding site",
        f"{target_name} {pdb_id} binding site experimental validation",
    ]

    result = {"text": "", "residues": [], "domains": [], "kd_values": [],
              "rag_layer": None}

    all_papers = []
    try:
        from retrieval.rag_router import get_retriever
        retriever = get_retriever()
        seen_titles = set()

        # Layer 1: PPI interfaces
        for query in primary_queries:
            try:
                papers_obj = await retriever.retrieve(
                    query, n_pubmed=4, n_biorxiv=2, n_local=3, years_back=5,
                )
                for p in (papers_obj or []):
                    pd = p.to_dict()
                    title = pd.get("title", "")
                    if title not in seen_titles:
                        seen_titles.add(title)
                        all_papers.append(pd)
            except Exception:
                continue

        # Check if Layer 1 found usable residues
        if all_papers:
            combined_l1 = " ".join(
                (p.get("abstract", "") or p.get("text", "")) for p in all_papers
            )
            l1_residues = _extract_residue_numbers_from_text(combined_l1)
            if l1_residues:
                result["rag_layer"] = "ppi_interface"
                logger.info("[RAG] Layer 1 (PPI): %d papers, %d residues found",
                            len(all_papers), len(l1_residues))

        # Layer 2 fallback: only if Layer 1 found no residues
        if not result.get("rag_layer"):
            for query in fallback_queries:
                try:
                    papers_obj = await retriever.retrieve(
                        query, n_pubmed=4, n_biorxiv=2, n_local=3, years_back=5,
                    )
                    for p in (papers_obj or []):
                        pd = p.to_dict()
                        title = pd.get("title", "")
                        if title not in seen_titles:
                            seen_titles.add(title)
                            all_papers.append(pd)
                except Exception:
                    continue
            result["rag_layer"] = "epitope_fallback"
            logger.info("[RAG] Layer 2 (epitope fallback): %d total papers", len(all_papers))

    except Exception as e:
        logger.warning("[pocket_scoring] RAG search failed: %s", e)
        return result

    if not all_papers:
        return result
    papers = all_papers

    # Combine abstracts/text
    texts = []
    for p in papers:
        abstract = p.get("abstract", "") or p.get("text", "") or ""
        title = p.get("title", "") or ""
        texts.append(f"{title}. {abstract}")
    combined = " ".join(texts)
    result["text"] = combined[:5000]

    # Extract residue numbers
    result["residues"] = _extract_residue_numbers_from_text(combined)

    # Extract domain names
    domain_patterns = [
        r'(domain\s+[IVX]+\b)',
        r'(subdomain\s+\w+)',
        r'(ECD|extracellular domain)',
        r'(kinase domain)',
        r'(dimerization\s+(?:arm|domain|interface))',
    ]
    domains = set()
    for pat in domain_patterns:
        for m in re.finditer(pat, combined, re.IGNORECASE):
            domains.add(m.group(1).strip())
    result["domains"] = sorted(domains)

    # Extract Kd values
    kd_patterns = [
        r'[Kk][Dd]\s*[=≈~:]\s*([\d.]+\s*[nμupM]+)',
        r'([\d.]+\s*[nμupM]+)\s*(?:affinity|binding)',
    ]
    kd_vals = set()
    for pat in kd_patterns:
        for m in re.finditer(pat, combined):
            kd_vals.add(m.group(1).strip())
    result["kd_values"] = sorted(kd_vals)

    return result


def _compute_freesasa_per_residue(pdb_path: str) -> dict[str, float]:
    """Run FreeSASA on PDB, return {chain+resnum: total_sasa} for all residues."""
    sasa_map = {}
    try:
        import freesasa
        structure = freesasa.Structure(pdb_path)
        result = freesasa.calc(structure)
        residue_areas = result.residueAreas()
        for chain_id, residues in residue_areas.items():
            for res_num, ra in residues.items():
                key = f"{chain_id}{res_num}"
                sasa_map[key] = round(ra.total, 2)
    except Exception as e:
        logger.warning("_compute_freesasa_per_residue failed: %s", e)
    return sasa_map


async def _qwen_select_pocket(scored_pockets: list[dict], rag_text: str,
                                target_name: str, pdb_id: str) -> dict:
    """Ask Qwen to select the best pocket for nanobody binder design.

    Returns: {selected_pocket_id: int, hotspot_residues: list[str],
              selection_reason: str}
    """
    pocket_summary = json.dumps(scored_pockets, indent=2, ensure_ascii=False)
    rag_snippet = rag_text[:3000] if rag_text else "(no literature found)"

    prompt = f"""You are a structural biologist. Select the best pocket for nanobody binder design on {target_name} (PDB: {pdb_id}).

## Scored Pockets
{pocket_summary}

## Literature Context
{rag_snippet}

Consider: biological relevance, surface exposure, literature evidence, and B-cell epitope overlap (epitope score).
Pockets with high epitope score overlap with predicted antibody binding sites (DiscoTope3), making them ideal for nanobody binder targeting.
The PPI-optimized scoring formula is: composite = rag(0.30) + pesto_ppi(0.25) + conservation(0.20) + sasa(0.10) + electrostatics(0.15).
Return ONLY valid JSON (no markdown):
{{"selected_pocket_id": <int>, "hotspot_residues": ["A42", "A45", ...], "selection_reason": "2-3 sentences"}}

Rules:
- selected_pocket_id is the pocket_id from the list above
- hotspot_residues: pick the 6 most important residues from that pocket for RFdiffusion targeting, prefer residues with high epitope propensity
- Prefer pockets with high composite score AND literature support AND PeSTo PPI interface overlap"""

    try:
        async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
            resp = await client.post(
                f"{settings.QWEN_BASE_URL}/chat/completions",
                json={
                    "model": "Qwen3-14B",
                    "messages": [
                        {"role": "system", "content": "You are a structural biology expert. Return only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1024,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            content = msg.get("content") or ""
            # Qwen3 thinking mode: content may be None, answer in reasoning
            if not content.strip():
                content = msg.get("reasoning_content") or msg.get("reasoning") or ""
            if not content.strip():
                logger.warning("[pocket_scoring] Qwen returned empty content. Full msg keys: %s",
                               list(msg.keys()))
                return {}
    except Exception as e:
        logger.warning("[pocket_scoring] Qwen selection failed: %s: %s", type(e).__name__, e,
                       exc_info=True)
        return {}

    logger.info("[pocket_scoring] Qwen raw response (%d chars): %.300s", len(content), content)

    # Strip </think> blocks and markdown fences
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    content = re.sub(r'```(?:json)?\s*', '', content).strip()
    content = re.sub(r'```\s*$', '', content).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        m = re.search(r'\{[^{}]*"selected_pocket_id"[^{}]*\}', content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        logger.warning("[pocket_scoring] Could not parse Qwen response: %s", content[:200])
        return {}


def _pocket_to_box(pocket: dict) -> tuple:
    """
    Extract docking box parameters from a single fpocket result pocket dict.

    Fpocket info.txt key naming after parser normalisation:
        "x bary (sph)" → "x_bary_(sph)"  (spaces→_, dots removed, slashes→_)

    Args:
        pocket: Single pocket dict from fpocket _info.txt parsing

    Returns:
        (center_x, center_y, center_z, box_side_angstrom)
        box_side is clamped to [15, 40] Å; derived as cbrt(volume)*1.2
    """
    center_x = float(pocket.get("x_bary_(sph)", 0.0))
    center_y = float(pocket.get("y_bary_(sph)", 0.0))
    center_z = float(pocket.get("z_bary_(sph)", 0.0))
    volume = float(pocket.get("real_volume", 15625.0))  # Å³
    box_side = max(15.0, min(40.0, (volume ** (1.0 / 3.0)) * 1.2))
    return center_x, center_y, center_z, box_side


def _select_gnina_poses(poses: list) -> tuple:
    """
    Identify best CNNscore pose and best minimizedAffinity pose from GNINA output.

    Args:
        poses: List of pose dicts from _parse_sdf_poses() in molecular_docking.py
               Each dict may contain: pose_id, affinity_kcal_mol, minimizedAffinity, CNNscore

    Returns:
        (best_cnn_pose, best_affinity_pose)
        - best_cnn_pose:     pose with highest CNNscore; falls back to poses[0] if no CNN data
        - best_affinity_pose: pose with lowest (most negative) minimizedAffinity / affinity_kcal_mol
        Either is None if poses is empty.
    """
    if not poses:
        return None, None
    best_affinity = min(
        poses,
        key=lambda p: p.get("minimizedAffinity", p.get("affinity_kcal_mol", 0.0)),
    )
    cnn_scored = [p for p in poses if "CNNscore" in p]
    best_cnn = max(cnn_scored, key=lambda p: p["CNNscore"]) if cnn_scored else poses[0]
    return best_cnn, best_affinity


# ─── CIF / SDF atom extraction ────────────────────────────────────────────────

def _extract_ligand_heavy_atoms_from_cif(cif_path: str) -> list:
    """
    Extract heavy atom (element, x, y, z) tuples for the first NonPolymer entity
    (i.e. the small-molecule ligand) found in an AF3 mmCIF file.

    Skips water (EntityType.Water) and polymer chains (EntityType.Polymer).
    Hydrogen / deuterium atoms are excluded.

    Returns [] if no ligand is present (protein-only run).
    """
    import gemmi
    st = gemmi.read_structure(cif_path)
    for model in st:
        for chain in model:
            for residue in chain:
                if residue.entity_type == gemmi.EntityType.NonPolymer:
                    atoms = [
                        (atom.element.name, atom.pos.x, atom.pos.y, atom.pos.z)
                        for atom in residue
                        if atom.element.name not in ("H", "D")
                    ]
                    if atoms:
                        return atoms   # first ligand residue found
    return []


def _extract_ligand_heavy_atoms_from_sdf(sdf_path: str) -> list:
    """
    Extract heavy atom (element, x, y, z) tuples from the first pose in an SDF file.

    Uses RDKit with H removal.  Returns [] if parsing fails or file missing.
    """
    from rdkit import Chem
    if not os.path.exists(sdf_path):
        return []
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=True)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None or not mol.GetNumConformers():
        return []
    conf = mol.GetConformer()
    return [
        (atom.GetSymbol(), float(conf.GetAtomPosition(atom.GetIdx()).x),
         float(conf.GetAtomPosition(atom.GetIdx()).y),
         float(conf.GetAtomPosition(atom.GetIdx()).z))
        for atom in mol.GetAtoms()
    ]


def _compute_optimal_atom_matching(atoms_a: list, atoms_b: list) -> tuple:
    """
    Hungarian-optimal atom matching between two heavy-atom coordinate sets.

    Each atom is (element, x, y, z).  Only same-element pairs are allowed;
    the cost matrix is pairwise squared distance with INF for element mismatch.

    Args:
        atoms_a: AF3 ligand atoms — (elem, x, y, z) list
        atoms_b: GNINA ligand atoms — (elem, x, y, z) list

    Returns:
        (rmsd, row_ind, col_ind)
        - rmsd: RMSD in Å over all matched pairs
        - row_ind[k]: index into atoms_a matched to col_ind[k] in atoms_b
        Returns (float('inf'), None, None) on:
          - different atom counts
          - element composition mismatch
          - infeasible matching (element pairings impossible)
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if len(atoms_a) != len(atoms_b):
        logger.debug(
            "_compute_optimal_atom_matching: count mismatch %d vs %d",
            len(atoms_a), len(atoms_b),
        )
        return float("inf"), None, None

    elems_a = sorted(e for e, *_ in atoms_a)
    elems_b = sorted(e for e, *_ in atoms_b)
    if elems_a != elems_b:
        logger.debug(
            "_compute_optimal_atom_matching: element mismatch %s vs %s",
            elems_a, elems_b,
        )
        return float("inf"), None, None

    n = len(atoms_a)
    INF = 1e18
    cost = np.full((n, n), INF)
    for i, (ea, xa, ya, za) in enumerate(atoms_a):
        for j, (eb, xb, yb, zb) in enumerate(atoms_b):
            if ea == eb:
                cost[i, j] = (xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2

    row_ind, col_ind = linear_sum_assignment(cost)

    if np.any(cost[row_ind, col_ind] >= INF / 2):
        logger.debug("_compute_optimal_atom_matching: infeasible element pairing")
        return float("inf"), None, None

    rmsd = float(np.sqrt(np.mean(cost[row_ind, col_ind])))
    return rmsd, row_ind.tolist(), col_ind.tolist()


# ─── CIF→PDB conversion (synchronous, gemmi) ─────────────────────────────────

def convert_af3_cif_to_pdb(cif_path: str, pdb_path: str, protein_only: bool = True) -> str:
    """
    Convert an AF3 mmCIF output file to PDB format using gemmi, then complete
    any missing sidechain heavy atoms via pdbfixer so the result is accepted
    by GROMACS pdb2gmx without the -missing workaround.

    Args:
        cif_path:     Input .cif file from AF3 (contains protein + ligand chains).
        pdb_path:     Desired output .pdb path.
        protein_only: If True (default), strip non-polymer / water chains so the
                      resulting PDB contains only ATOM records — suitable for
                      GROMACS pdb2gmx with AMBER99SB-ILDN.
                      If False, write all chains (protein + HETATM ligand).

    Returns:
        pdb_path on success.

    Raises:
        RuntimeError if gemmi or pdbfixer fails or the output file is not created.
    """
    import gemmi
    st = gemmi.read_structure(cif_path)

    if protein_only:
        for model in st:
            remove_chains = [
                ch.name for ch in model
                if all(
                    res.entity_type != gemmi.EntityType.Polymer
                    for res in ch
                )
            ]
            for name in remove_chains:
                model.remove_chain(name)

    # Write intermediate PDB for pdbfixer
    tmp_pdb = pdb_path + ".tmp_gemmi.pdb"
    st.write_pdb(tmp_pdb)

    # Complete missing sidechain heavy atoms with pdbfixer
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
        fixer = PDBFixer(filename=tmp_pdb)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        with open(pdb_path, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        logger.info(f"convert_af3_cif_to_pdb: pdbfixer completed missing atoms → {pdb_path}")
    except Exception as e:
        logger.warning(f"convert_af3_cif_to_pdb: pdbfixer failed ({e}), falling back to raw gemmi output")
        import shutil
        shutil.copy(tmp_pdb, pdb_path)
    finally:
        if os.path.exists(tmp_pdb):
            os.remove(tmp_pdb)

    if not os.path.exists(pdb_path):
        raise RuntimeError(f"convert_af3_cif_to_pdb: output not created: {pdb_path}")
    return pdb_path


def _write_af3_ligand_as_sdf(
    af3_atoms: list,
    row_ind: list,
    col_ind: list,
    gnina_sdf: str,
    out_sdf: str,
) -> bool:
    """
    Write AF3 ligand coordinates onto the GNINA mol topology and save as SDF.

    The GNINA SDF carries the bond graph (topology); AF3 CIF carries the 3-D
    coordinates from the structure prediction.  We replace GNINA's conformer
    positions with the AF3 positions using the atom mapping from
    _compute_optimal_atom_matching.

    Args:
        af3_atoms: (elem, x, y, z) list from _extract_ligand_heavy_atoms_from_cif
        row_ind:   row indices from _compute_optimal_atom_matching (AF3 side)
        col_ind:   col indices from _compute_optimal_atom_matching (GNINA side)
        gnina_sdf: single-pose SDF with the correct molecular graph
        out_sdf:   output SDF path

    Returns:
        True if out_sdf was written successfully.
    """
    from rdkit import Chem
    from rdkit.Chem import Conformer
    import numpy as np

    suppl = Chem.SDMolSupplier(gnina_sdf, removeHs=True, sanitize=True)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        return False

    # Build gnina_atom_idx → af3_atom_idx map from the matching
    gnina_to_af3 = {col_ind[k]: row_ind[k] for k in range(len(row_ind))}

    conf = Conformer(mol.GetNumAtoms())
    for j in range(mol.GetNumAtoms()):
        af3_idx = gnina_to_af3.get(j)
        if af3_idx is None:
            return False   # mapping incomplete
        _, x, y, z = af3_atoms[af3_idx]
        conf.SetAtomPosition(j, (x, y, z))

    mol.AddConformer(conf, assignId=True)
    writer = Chem.SDWriter(out_sdf)
    writer.write(mol, confId=mol.GetNumConformers() - 1)
    writer.close()
    return os.path.exists(out_sdf)


# ─── Async helpers ────────────────────────────────────────────────────────────

async def _extract_sdf_pose(sdf_path: str, pose_id: int, out_path: str) -> bool:
    """
    Extract a single pose (1-based pose_id) from a multi-pose SDF using host obabel.
    Returns True if the output file was created successfully.
    """
    import asyncio as _aio
    proc = await _aio.create_subprocess_exec(
        "obabel", sdf_path, "-O", out_path,
        "-f", str(pose_id), "-l", str(pose_id),
        stdout=_aio.subprocess.PIPE,
        stderr=_aio.subprocess.PIPE,
    )
    await _aio.wait_for(proc.communicate(), timeout=30)
    return os.path.exists(out_path)


async def _compute_rmsd_af3_gnina(
    af3_cif: str,
    gnina_sdf: str,
    receptor_pdb: str,
) -> tuple:
    """
    Compute ligand RMSD between AF3 complex prediction and GNINA best pose.

    Algorithm:
      1. Extract ligand heavy atoms from AF3 mmCIF  (_extract_ligand_heavy_atoms_from_cif)
      2. Extract ligand heavy atoms from GNINA SDF   (_extract_ligand_heavy_atoms_from_sdf)
      3. Hungarian-optimal element-constrained atom matching (_compute_optimal_atom_matching)
      4. Return RMSD in Å and the atom mapping (for downstream coord transplanting)

    Args:
        af3_cif:      Host-side path to the AF3 top-ranked .cif file.
        gnina_sdf:    Host-side path to the single-pose GNINA SDF (best-affinity pose).
        receptor_pdb: Receptor PDB path (unused currently; kept for future chain-context use).

    Returns:
        (rmsd, row_ind, col_ind)
        - rmsd     : RMSD in Å; float('inf') on any failure
        - row_ind  : AF3 atom indices matched (None on failure)
        - col_ind  : GNINA atom indices matched (None on failure)
    """
    try:
        af3_atoms = _extract_ligand_heavy_atoms_from_cif(af3_cif)
        if not af3_atoms:
            logger.warning("_compute_rmsd_af3_gnina: no ligand in CIF %s", af3_cif)
            return float("inf"), None, None

        gnina_atoms = _extract_ligand_heavy_atoms_from_sdf(gnina_sdf)
        if not gnina_atoms:
            logger.warning("_compute_rmsd_af3_gnina: cannot parse SDF %s", gnina_sdf)
            return float("inf"), None, None

        rmsd, row_ind, col_ind = _compute_optimal_atom_matching(af3_atoms, gnina_atoms)
        logger.info(
            "_compute_rmsd_af3_gnina: RMSD=%.3f Å  (%d heavy atoms, AF3=%s, GNINA=%s)",
            rmsd if rmsd != float("inf") else -1,
            len(af3_atoms),
            os.path.basename(af3_cif),
            os.path.basename(gnina_sdf),
        )
        return rmsd, row_ind, col_ind

    except Exception:
        logger.warning("_compute_rmsd_af3_gnina failed", exc_info=True)
        return float("inf"), None, None


# ─── Drug Discovery Pipeline ──────────────────────────────────────────────────

@router.post("/drug-discovery", response_model=TaskRef,
             summary="Full drug discovery: fetch PDB → fpocket → dock (GNINA) → AF3 validate → 10ns MD")
async def drug_discovery_pipeline(req: DrugDiscoveryPipelineRequest):
    """
    Complete 6-step drug discovery pipeline.  Each step reads paths from the
    previous step's output — no hardcoded paths.

    Steps:
      1. fetch_pdb      — Download target from RCSB (if pdb_id given)
      2. fpocket        — Detect pockets; extract center_x/y/z + box_side for top pocket
      3. fetch_molecule — Resolve ligand name → canonical SMILES via PubChem (if ligand_name given)
      4. gnina          — Dock; identify best_CNNscore pose AND best_minimizedAffinity pose
      5. alphafold3     — [Optional] Predict protein+ligand complex; compute RMSD vs GNINA
                          Hungarian-optimal element-matched RMSD via gemmi + scipy
      6. gromacs        — [Optional] 10ns NVT+NPT+MD on:
                            AF3 complex (protein PDB + AF3 ligand SDF) if RMSD < rmsd_threshold_angstrom
                            GNINA best-affinity pose otherwise

    Input priority:
        target:  pdb_id (RCSB fetch) > target_pdb (local path)
        ligand:  ligand_name (PubChem) > ligand_smiles (direct SMILES)
    """
    async def _run(task):
        results = {"job_name": req.job_name, "steps": {}}

        # ── Step 1: Fetch target PDB ──────────────────────────────────────────
        task.progress = 5
        task.progress_msg = "Step 1/6: Fetching target structure..."

        if req.pdb_id:
            from routers.structure_prediction import fetch_pdb, FetchPDBRequest
            fetch_result = await fetch_pdb(FetchPDBRequest(pdb_id=req.pdb_id))
            target_pdb = fetch_result["output_pdb"]
            results["steps"]["fetch_pdb"] = fetch_result
            logger.info(f"[pipeline] fetch_pdb({req.pdb_id}) → {target_pdb}")
        elif req.target_pdb:
            target_pdb = req.target_pdb
            results["steps"]["fetch_pdb"] = {"skipped": True, "target_pdb": target_pdb}
        else:
            raise ValueError("Provide either pdb_id (RCSB fetch) or target_pdb (local path)")

        # ── Step 2: Fpocket — detect binding pockets ──────────────────────────
        task.progress = 15
        task.progress_msg = "Step 2/6: Detecting binding pockets (fpocket)..."

        from routers.pocket_analysis import run_fpocket
        fpocket_ref = await run_fpocket(FpocketRequest(
            job_name=f"{req.job_name}_fpocket",
            input_pdb=target_pdb,
        ))
        fpocket_result = await _wait_for_task(fpocket_ref.task_id, timeout=300)
        results["steps"]["fpocket"] = fpocket_result

        pockets = fpocket_result.get("pockets", [])
        if not pockets:
            raise RuntimeError("fpocket found 0 pockets — check input PDB quality")
        best_pocket = pockets[0]
        center_x, center_y, center_z, box_side = _pocket_to_box(best_pocket)
        logger.info(
            f"[pipeline] fpocket top pocket: center=({center_x:.2f},{center_y:.2f},{center_z:.2f}) "
            f"box={box_side:.1f}Å  drug_score={best_pocket.get('drug_score', 'N/A')}"
        )

        # ── Step 3: Resolve ligand SMILES ─────────────────────────────────────
        task.progress = 28
        task.progress_msg = "Step 3/6: Resolving ligand SMILES..."

        if req.ligand_name:
            from routers.structure_prediction import fetch_molecule, FetchMoleculeRequest
            mol_result = await fetch_molecule(FetchMoleculeRequest(query=req.ligand_name))
            ligand_smiles = mol_result["smiles"]
            results["steps"]["fetch_molecule"] = mol_result
            logger.info(f"[pipeline] fetch_molecule('{req.ligand_name}') → {ligand_smiles[:60]}")
        elif req.ligand_smiles:
            ligand_smiles = req.ligand_smiles
            results["steps"]["fetch_molecule"] = {"skipped": True, "smiles": ligand_smiles}
        else:
            raise ValueError("Provide either ligand_name (PubChem lookup) or ligand_smiles (direct SMILES)")

        # ── Step 4: GNINA docking ─────────────────────────────────────────────
        task.progress = 38
        task.progress_msg = "Step 4/6: GNINA docking..."

        from routers.molecular_docking import run_gnina
        gnina_ref = await run_gnina(DockingRequest(
            job_name=f"{req.job_name}_dock",
            engine=DockingEngine.GNINA,
            receptor_pdb=target_pdb,
            ligand=ligand_smiles,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            box_size_x=box_side,
            box_size_y=box_side,
            box_size_z=box_side,
            num_poses=9,
            exhaustiveness=8,
        ))
        gnina_result = await _wait_for_task(gnina_ref.task_id, timeout=3600)
        results["steps"]["gnina"] = gnina_result

        poses = gnina_result.get("poses", [])
        gnina_sdf = gnina_result.get("output_sdf", "")
        best_cnn_pose, best_affinity_pose = _select_gnina_poses(poses)
        logger.info(
            f"[pipeline] GNINA: {len(poses)} poses. "
            f"Best affinity={best_affinity_pose.get('affinity_kcal_mol') if best_affinity_pose else 'N/A'} kcal/mol, "
            f"Best CNNscore={best_cnn_pose.get('CNNscore') if best_cnn_pose else 'N/A'}"
        )
        results["steps"]["gnina"]["best_cnn_pose"] = best_cnn_pose
        results["steps"]["gnina"]["best_affinity_pose"] = best_affinity_pose

        # Extract best-affinity pose to its own SDF (GROMACS acpype takes pose 1)
        gnina_sdf_host = gnina_sdf.replace("/data/oih/outputs", settings.OUTPUT_DIR)
        best_affinity_sdf_host = gnina_sdf_host.replace("poses.sdf", "best_affinity_pose.sdf")
        best_affinity_sdf = gnina_sdf.replace("poses.sdf", "best_affinity_pose.sdf")
        if best_affinity_pose and gnina_sdf:
            extracted = await _extract_sdf_pose(
                gnina_sdf_host,
                best_affinity_pose["pose_id"],
                best_affinity_sdf_host,
            )
            if not extracted:
                best_affinity_sdf = gnina_sdf       # fallback: full SDF, acpype grabs pose 1
                best_affinity_sdf_host = gnina_sdf_host

        # Default MD inputs (may be overridden by AF3 step)
        md_input_pdb = target_pdb           # protein-only PDB; GROMACS handles ligand via ligand_sdf
        md_ligand_sdf = best_affinity_sdf   # container path
        md_source = "gnina_best_affinity"

        # ── Step 5: AlphaFold3 complex prediction + RMSD ─────────────────────
        if req.run_af3_validation and req.protein_sequence:
            task.progress = 55
            task.progress_msg = "Step 5/6: AlphaFold3 complex prediction..."

            from routers.structure_prediction import predict_alphafold3
            af3_ref = await predict_alphafold3(AlphaFold3Request(
                job_name=f"{req.job_name}_af3_complex",
                chains=[
                    AF3Chain(type="protein", sequence=req.protein_sequence),
                    AF3Chain(type="ligand", smiles=ligand_smiles),
                ],
                num_seeds=3,
            ))
            af3_result = await _wait_for_af3_task(af3_ref.task_id)
            results["steps"]["alphafold3"] = af3_result

            af3_cif = af3_result.get("cif_files", [None])[0]
            if af3_cif:
                rmsd, row_ind, col_ind = await _compute_rmsd_af3_gnina(
                    af3_cif, best_affinity_sdf_host, target_pdb
                )
                failed = rmsd == float("inf")
                results["steps"]["alphafold3"]["rmsd_vs_gnina_angstrom"] = None if failed else round(rmsd, 3)
                results["steps"]["alphafold3"]["rmsd_note"] = (
                    f"RMSD={rmsd:.2f} Å vs GNINA best-affinity pose "
                    f"(threshold={req.rmsd_threshold_angstrom} Å)"
                    if not failed else
                    "RMSD computation failed — using GNINA pose for MD"
                )
                logger.info("[pipeline] AF3 RMSD vs GNINA: %s",
                            f"{rmsd:.2f} Å" if not failed else "failed")

                if not failed and rmsd < req.rmsd_threshold_angstrom:
                    # AF3 and GNINA agree → use AF3 structural context for MD
                    # Protein chain: CIF → PDB (protein-only, for pdb2gmx)
                    # Ligand:        AF3 coords transplanted onto GNINA topology → SDF (for acpype)
                    af3_protein_pdb = af3_cif.replace(".cif", "_protein.pdb")
                    af3_ligand_sdf  = af3_cif.replace(".cif", "_ligand_af3pose.sdf")
                    af3_atoms_for_write = _extract_ligand_heavy_atoms_from_cif(af3_cif)
                    try:
                        convert_af3_cif_to_pdb(af3_cif, af3_protein_pdb, protein_only=True)
                        wrote_sdf = _write_af3_ligand_as_sdf(
                            af3_atoms_for_write, row_ind, col_ind,
                            best_affinity_sdf_host, af3_ligand_sdf,
                        )
                        if wrote_sdf:
                            # Use container-side paths for GROMACS
                            md_input_pdb = af3_protein_pdb.replace(settings.OUTPUT_DIR, "/data/oih/outputs")
                            md_ligand_sdf = af3_ligand_sdf.replace(settings.OUTPUT_DIR, "/data/oih/outputs")
                            md_source = "af3_complex"
                            logger.info(
                                "[pipeline] RMSD %.2f Å < %.1f Å → AF3 protein+ligand selected for MD",
                                rmsd, req.rmsd_threshold_angstrom,
                            )
                        else:
                            logger.warning("[pipeline] _write_af3_ligand_as_sdf failed — falling back to GNINA")
                            md_source = "gnina_best_affinity (af3_ligand_sdf_failed)"
                    except Exception as _e:
                        logger.warning("[pipeline] AF3→PDB/SDF conversion failed: %s — fallback to GNINA", _e)
                        md_source = "gnina_best_affinity (af3_conversion_failed)"
                else:
                    logger.info(
                        "[pipeline] RMSD %s≥ threshold → GNINA best-affinity pose used for MD",
                        f"{rmsd:.2f} Å " if not failed else "",
                    )
        else:
            results["steps"]["alphafold3"] = {
                "skipped": True,
                "reason": "run_af3_validation=False or protein_sequence not provided",
            }

        # ── Step 6: GROMACS 10ns MD ───────────────────────────────────────────
        if req.run_md:
            task.progress = 75
            task.progress_msg = f"Step 6/6: GROMACS 10ns MD ({md_source})..."

            from routers.md_simulation import run_gromacs
            md_ref = await run_gromacs(GROMACSRequest(
                job_name=f"{req.job_name}_md",
                input_pdb=md_input_pdb,
                preset=GromacsPreset.PROTEIN_LIGAND,
                ligand_sdf=md_ligand_sdf,
                sim_time_ns=10.0,
            ))
            md_result = await _wait_for_task(md_ref.task_id, timeout=86400)
            md_result["md_source"] = md_source
            results["steps"]["md_simulation"] = md_result
        else:
            results["steps"]["md_simulation"] = {"skipped": True, "reason": "run_md=False"}

        task.progress = 100
        task.progress_msg = "Drug discovery pipeline complete!"
        results["summary"] = {
            "target_pdb": target_pdb,
            "ligand_smiles": ligand_smiles,
            "num_pockets_found": len(pockets),
            "best_affinity_kcal_mol": (
                best_affinity_pose.get("affinity_kcal_mol") if best_affinity_pose else None
            ),
            "best_cnnscore": best_cnn_pose.get("CNNscore") if best_cnn_pose else None,
            "md_source": md_source,
        }
        return results

    task = await task_manager.submit("drug_discovery_pipeline", req.model_dump(), _run)
    return TaskRef(task_id=task.task_id, status=task.status, tool="drug_discovery_pipeline",
                   poll_url=f"/api/v1/tasks/{task.task_id}")


@router.post("/binder-design", response_model=TaskRef,
             summary="Full binder design + ADC: RFdiffusion → MPNN → AF3 → FreeSASA → Linker → Payload → Conjugate")
async def binder_design_pipeline(req: BinderDesignPipelineRequest):
    """
    Complete protein binder design + ADC construction pipeline:
    1. RFdiffusion  → generate backbone scaffolds
    2. ProteinMPNN  → design sequences for each backbone
    3. AlphaFold3   → validate top sequences (predict complex, filter by ipTM)
    4. FreeSASA     → analyze nanobody surface, find Lys/Cys conjugation sites (SASA>40Å²)
    5. Linker Select → pick cleavable linker matching conjugation chemistry
    6. Fetch Payload → retrieve MMAE SMILES from PubChem
    7. RDKit Conjugate → build ADC: antibody + linker + payload → DAR=4
    """
    # Resolve num_designs → num_rfdiffusion_designs if user used the alias
    if req.num_designs != 10 and req.num_rfdiffusion_designs == 50:
        req.num_rfdiffusion_designs = req.num_designs

    # ── dry_run: return mock results without running any tool ──
    if req.dry_run:
        hotspot_str = ",".join(req.hotspot_residues) if req.hotspot_residues else None
        mock = {
            "job_name": req.job_name,
            "dry_run": True,
            "steps": {
                "fetch_pdb": {
                    "status": "skipped (dry_run)",
                    "pdb_id": req.pdb_id,
                    "target_pdb": req.target_pdb,
                },
                "rfdiffusion": {
                    "status": "skipped (dry_run)",
                    "target_pdb": req.pdb_id or req.target_pdb,
                    "hotspot_residues": hotspot_str,
                    "num_designs": req.num_rfdiffusion_designs,
                },
                "proteinmpnn": {"status": "skipped (dry_run)"},
                "af3_validation": {
                    "status": "skipped (dry_run)",
                    "validated_designs": [],
                    "total_tested": 0,
                    "passed": 0,
                    "best_iptm": None,
                    "best_sequence": None,
                },
                "freesasa": {"status": "skipped (dry_run)"},
                "linker_select": {"status": "skipped (dry_run)"},
                "fetch_payload": {"status": "skipped (dry_run)"},
                "rdkit_conjugate": {"status": "skipped (dry_run)"},
            },
            "adc_design": {
                "nanobody_sequence": None,
                "iptm": None,
                "conjugation_site": None,
                "linker": "MC-VC-PABC (mock)",
                "payload": "MMAE",
                "dar": 4,
                "adc_mw": None,
                "adc_smiles": None,
                "adc_structure_path": None,
                "dry_run": True,
            },
        }
        task = await task_manager.submit(
            "binder_design_pipeline", req.model_dump(),
            _make_dry_run_coroutine(mock),
        )
        return TaskRef(
            task_id=task.task_id, status=task.status,
            tool="binder_design_pipeline",
            poll_url=f"/api/v1/tasks/{task.task_id}",
        )

    async def _run(task):
        results = {"job_name": req.job_name, "steps": {}}
        hotspot_str = ",".join(req.hotspot_residues) if req.hotspot_residues else None

        # Step 0: Fetch target PDB if pdb_id given
        task.progress = 2
        task.progress_msg = "Fetching target structure..."
        if req.pdb_id:
            from routers.structure_prediction import fetch_pdb, FetchPDBRequest
            fetch_result = await fetch_pdb(FetchPDBRequest(pdb_id=req.pdb_id))
            target_pdb = fetch_result["output_pdb"]
            results["steps"]["fetch_pdb"] = fetch_result
            logger.info("[binder_pipeline] fetch_pdb(%s) → %s", req.pdb_id, target_pdb)
        elif req.target_pdb:
            target_pdb = req.target_pdb
            results["steps"]["fetch_pdb"] = {"skipped": True, "target_pdb": target_pdb}
        else:
            raise ValueError("Provide either pdb_id (RCSB fetch) or target_pdb (local path)")

        # Step 0.5: Auto-detect hotspots via Tier classification if none provided
        if not hotspot_str:
            _target_name = getattr(req, 'target_name', '') or req.job_name
            tier_info = _classify_target_tier(_target_name, req.pdb_id or "", {})
            if tier_info["tier"] == 1:
                # Tier1: extract interface from known complex PDB
                task.progress_msg = "Auto-detecting hotspots (Tier1: extract_interface)..."
                from routers.structure_prediction import (
                    fetch_pdb as _fetch, FetchPDBRequest as _FR,
                    extract_interface_residues, ExtractInterfaceRequest,
                )
                complex_pdb_result = await _fetch(_FR(pdb_id=tier_info["pdb"]))
                interface_result = await extract_interface_residues(ExtractInterfaceRequest(
                    job_name=f"{req.job_name}_interface",
                    complex_pdb=complex_pdb_result["output_pdb"],
                    receptor_chain=tier_info["receptor_chain"],
                    ligand_chains=tier_info["ligand_chains"],
                    cutoff_angstrom=5.0, top_n=6,
                ))
                hotspot_str = ",".join(interface_result["interface_residues"])
                results["steps"]["hotspot_discovery"] = {
                    "method": "tier1_extract_interface",
                    "source_pdb": tier_info["pdb"],
                    "source_antibody": tier_info["source"],
                    "hotspots": interface_result["interface_residues"],
                }
                logger.info("[binder_pipeline] Tier1 auto-hotspot from %s: %s",
                            tier_info["pdb"], hotspot_str)
            else:
                results["steps"]["hotspot_discovery"] = {
                    "method": f"tier{tier_info['tier']}_user_provided",
                    "note": "No hotspots provided and target not in KNOWN_COMPLEXES. "
                            "Consider running PeSTo for Tier3 targets.",
                }

        # Step 1: RFdiffusion backbone generation
        task.progress = 5
        task.progress_msg = "Step 1/7: RFdiffusion backbone generation..."
        from routers.protein_design import run_rfdiffusion
        rfd_req = RFdiffusionRequest(
            job_name=f"{req.job_name}_rfd",
            mode="binder_design",
            target_pdb=target_pdb,
            hotspot_residues=hotspot_str,
            num_designs=req.num_rfdiffusion_designs,
        )
        rfd_ref = await run_rfdiffusion(rfd_req)
        rfd_result = await _wait_for_task(rfd_ref.task_id, timeout=7200)
        results["steps"]["rfdiffusion"] = rfd_result
        backbone_pdbs = rfd_result.get("pdb_files", [])

        # Step 2: ProteinMPNN sequence design on each backbone
        task.progress = 50
        task.progress_msg = f"Step 2/7: ProteinMPNN on {len(backbone_pdbs)} backbones..."
        from routers.protein_design import run_proteinmpnn
        all_sequences = []
        # Detect binder chain: RFdiffusion binder_design outputs chain A=target, B=binder
        # MPNN must redesign the BINDER chain (shorter one), not the target
        _binder_chain = "B"  # RFdiffusion binder_design default
        try:
            import gemmi as _gemmi
            _st = _gemmi.read_structure(backbone_pdbs[0])
            _chains = [(c.name, len([r for r in c if r.entity_type == _gemmi.EntityType.Polymer])) for c in _st[0]]
            _chains.sort(key=lambda x: x[1])
            _binder_chain = _chains[0][0]  # shortest chain = binder
            logger.info("[binder_pipeline] Detected binder chain: %s (%d res)", _binder_chain, _chains[0][1])
        except Exception:
            pass

        for i, pdb in enumerate(backbone_pdbs[:20]):  # cap at 20 for time
            mpnn_req = ProteinMPNNRequest(
                job_name=f"{req.job_name}_mpnn_{i}",
                input_pdb=pdb,
                num_sequences=req.num_mpnn_sequences,
                chains_to_design=_binder_chain,
            )
            mpnn_ref = await run_proteinmpnn(mpnn_req)
            mpnn_result = await _wait_for_task(mpnn_ref.task_id, timeout=600)
            all_sequences.append(mpnn_result)
        results["steps"]["proteinmpnn"] = {"designs": all_sequences}

        # Step 3: AlphaFold3 validation of top designs
        if req.run_af3_validation:
            task.progress = 75
            task.progress_msg = "Step 3/7: AlphaFold3 validation of top binders..."

            # Extract target sequence from PDB for AF3 complex prediction
            target_seq = _extract_sequence_from_pdb(target_pdb)
            if not target_seq:
                logger.warning("[binder_pipeline] Cannot extract target sequence from %s", target_pdb)
                results["steps"]["af3_validation"] = {
                    "error": "cannot_extract_target_sequence",
                    "validated_designs": [],
                    "total_tested": 0,
                    "passed": 0,
                    "best_iptm": None,
                    "best_sequence": None,
                }
            else:
                # Collect all designed sequences from MPNN output
                # MPNN router returns {fasta_file, output_dir}; parse FASTA for sequences
                all_designs = []
                for step_idx, step_result in enumerate(all_sequences):
                    # Try structured sequences first, then fall back to FASTA parsing
                    seqs = step_result.get("sequences", [])
                    if not seqs and step_result.get("fasta_file"):
                        seqs = _parse_mpnn_fasta(step_result["fasta_file"])
                    for seq_info in seqs:
                        seq_info["_backbone_idx"] = step_idx
                        all_designs.append(seq_info)
                all_designs.sort(key=lambda x: x.get("score") if x.get("score") is not None else float("inf"))
                logger.info("[binder_pipeline] Collected %d MPNN designs from %d backbones", len(all_designs), len(all_sequences))
                top_n = min(5, len(all_designs))
                top_designs = all_designs[:top_n]

                from routers.structure_prediction import predict_alphafold3

                # Get antigen domain regions for AF3 validation
                _target_name = getattr(req, 'target_name', '') or req.job_name
                _hotspots = getattr(req, 'hotspot_residues', None) or []
                antigen_regions = _get_af3_antigen_regions(
                    target_name=_target_name,
                    target_seq=target_seq,
                    hotspot_residues=_hotspots,
                )
                logger.info("[binder_pipeline] AF3 antigen regions: %s",
                            [(r["domain_name"], len(r["sequence"])) for r in antigen_regions])

                validated_designs = []
                for i, design in enumerate(top_designs):
                    task.progress = 75 + int(20 * i / max(top_n, 1))
                    binder_seq = design.get("sequence", "")
                    if not binder_seq:
                        continue

                    # Validate against each antigen domain region
                    best_iptm_for_design = None
                    best_result_for_design = None
                    best_region_for_design = None

                    for region in antigen_regions:
                        task.progress_msg = (
                            f"Step 3/7: AF3 design {i+1}/{top_n} "
                            f"vs {region['domain_name']} ({len(region['sequence'])}aa)..."
                        )

                        af3_ref = await predict_alphafold3(AlphaFold3Request(
                            job_name=f"{req.job_name}_{region['domain_name']}_val_{i}",
                            chains=[
                                AF3Chain(type="protein", sequence=binder_seq),
                                AF3Chain(type="protein", sequence=region["sequence"]),
                            ],
                            num_seeds=3,
                        ))
                        try:
                            af3_result = await _wait_for_af3_task(af3_ref.task_id)
                        except Exception as e:
                            logger.warning("[binder_pipeline] AF3 failed for design %d/%s: %s",
                                           i, region["domain_name"], e)
                            continue

                        # Parse ipTM
                        iptm = None
                        for cf in af3_result.get("confidence_files", []):
                            try:
                                with open(cf) as f:
                                    conf = json.load(f)
                                iptm = conf.get("iptm", conf.get("ipTM"))
                                if iptm is not None:
                                    break
                            except Exception:
                                continue

                        if iptm is not None and (best_iptm_for_design is None or iptm > best_iptm_for_design):
                            best_iptm_for_design = iptm
                            best_result_for_design = af3_result
                            best_region_for_design = region

                        if len(antigen_regions) > 1 and iptm is not None:
                            await asyncio.sleep(3)

                    # Use best domain result for this design
                    if best_result_for_design is not None:
                        af3_result = best_result_for_design
                        iptm = best_iptm_for_design
                    else:
                        # All domains failed for this design
                        validated_designs.append({
                            "rank": i + 1,
                            "sequence": binder_seq,
                            "mpnn_score": design.get("score"),
                            "iptm": None,
                            "confidence": "failed",
                            "af3_structure_path": None,
                            "passed_validation": False,
                            "domain": best_region_for_design["domain_name"] if best_region_for_design else None,
                            "error": "all_domains_failed",
                        })
                        if i < top_n - 1:
                            await asyncio.sleep(5)
                        continue

                    # Determine structure output path from best domain result
                    cif_files = af3_result.get("cif_files", [])
                    af3_structure_path = os.path.dirname(cif_files[0]) if cif_files else None

                    # Classify confidence
                    if iptm is not None and iptm >= 0.75:
                        confidence = "high"
                        passed = True
                    elif iptm is not None and iptm >= 0.6:
                        confidence = "low_confidence"
                        passed = True
                    else:
                        confidence = "low" if iptm is not None else "unknown"
                        passed = False

                    validated_designs.append({
                        "rank": i + 1,
                        "sequence": binder_seq,
                        "mpnn_score": design.get("score"),
                        "iptm": round(iptm, 4) if iptm is not None else None,
                        "confidence": confidence,
                        "af3_structure_path": af3_structure_path,
                        "passed_validation": passed,
                        "domain": best_region_for_design["domain_name"] if best_region_for_design else None,
                        "antigen_offset": best_region_for_design["offset"] if best_region_for_design else 0,
                        "antigen_length": len(best_region_for_design["sequence"]) if best_region_for_design else len(target_seq),
                    })

                    # 5s delay before next submission to avoid GPU OOM
                    if i < top_n - 1:
                        await asyncio.sleep(5)

                # ── ipSAE verification (detect false positives) ──
                from routers.structure_prediction import ipsae_score as _ipsae, IpSAERequest as _IpSAEReq
                for d in validated_designs:
                    if d.get("af3_structure_path") and d.get("iptm") is not None:
                        try:
                            ipsae_result = await _ipsae(_IpSAEReq(
                                job_name=f"{req.job_name}_ipsae_{d['rank']}",
                                af3_output_dir=d["af3_structure_path"],
                            ))
                            # Take max ipSAE across chain pairs
                            pairs = ipsae_result.get("chain_pairs", [])
                            d["ipsae"] = max((p.get("ipSAE", 0) for p in pairs), default=0)
                            d["ipsae"] = round(d["ipsae"], 4)
                            logger.info("[binder_pipeline] Design %d: ipTM=%.3f, ipSAE=%.4f",
                                        d["rank"], d["iptm"], d["ipsae"])
                            # ipSAE=0 with decent ipTM → false positive
                            if d["ipsae"] < 0.01 and d.get("passed_validation"):
                                d["passed_validation"] = False
                                d["confidence"] = "false_positive"
                                logger.warning("[binder_pipeline] Design %d: ipSAE=0 → false positive!", d["rank"])
                        except Exception as e:
                            d["ipsae"] = None
                            logger.warning("[binder_pipeline] ipSAE failed for design %d: %s", d["rank"], e)

                # Sort validated designs by ipTM descending
                validated_designs.sort(
                    key=lambda x: x["iptm"] if x["iptm"] is not None else -1,
                    reverse=True,
                )
                # Re-assign rank after sorting
                for rank, d in enumerate(validated_designs, 1):
                    d["rank"] = rank

                passed_list = [d for d in validated_designs if d["passed_validation"]]
                best = validated_designs[0] if validated_designs else None

                results["steps"]["af3_validation"] = {
                    "validated_designs": validated_designs,
                    "total_tested": len(validated_designs),
                    "passed": len(passed_list),
                    "best_iptm": best["iptm"] if best else None,
                    "best_ipsae": best.get("ipsae") if best else None,
                    "best_sequence": best["sequence"] if best else None,
                }
        else:
            results["steps"]["af3_validation"] = {
                "skipped": True,
                "validated_designs": [],
                "total_tested": 0,
                "passed": 0,
                "best_iptm": None,
                "best_sequence": None,
            }

        # ── Step 4-7: ADC construction (freesasa → linker → payload → conjugate) ──
        adc_design = {}
        af3_val = results["steps"].get("af3_validation", {})
        best_design = None
        validated = af3_val.get("validated_designs", [])
        if validated:
            # Pick best ipTM design that has a structure path
            for d in validated:
                if d.get("af3_structure_path") and d.get("passed_validation"):
                    best_design = d
                    break
            if not best_design and validated:
                best_design = validated[0]  # fallback to top even if not passed

        if not best_design or not best_design.get("af3_structure_path"):
            adc_design = {
                "error": "No AF3 validated structure available for ADC construction",
                "partial": True,
            }
        else:
            nanobody_seq = best_design["sequence"]
            best_iptm = best_design.get("iptm")
            af3_dir = best_design["af3_structure_path"]

            # Find the CIF/PDB file in AF3 output dir for freesasa
            af3_pdb = None
            for ext in ("_protein.pdb", ".pdb"):
                candidate = os.path.join(af3_dir, os.listdir(af3_dir)[0].rsplit(".", 1)[0] + ext) if os.path.isdir(af3_dir) else None
                if candidate and os.path.exists(candidate):
                    af3_pdb = candidate
                    break
            if not af3_pdb:
                # Convert best CIF to PDB for freesasa
                cif_files = [f for f in os.listdir(af3_dir) if f.endswith(".cif")] if os.path.isdir(af3_dir) else []
                if cif_files:
                    af3_cif = os.path.join(af3_dir, cif_files[0])
                    af3_pdb = af3_cif.replace(".cif", "_for_sasa.pdb")
                    try:
                        convert_af3_cif_to_pdb(af3_cif, af3_pdb, protein_only=False)
                    except Exception as e:
                        logger.warning("[binder_pipeline] CIF→PDB for freesasa failed: %s", e)
                        af3_pdb = None

            # ── Step 4: FreeSASA conjugation site analysis ──────────────────
            try:
                task.progress = 80
                task.progress_msg = "Step 4/7: FreeSASA conjugation site analysis..."

                if not af3_pdb:
                    raise RuntimeError("No PDB available for FreeSASA analysis")

                from routers.protein_design import run_freesasa
                from schemas.models import FreeSASARequest
                sasa_ref = await run_freesasa(FreeSASARequest(
                    job_name=f"{req.job_name}_freesasa",
                    input_pdb=af3_pdb,
                ))
                sasa_result = await _wait_for_task(sasa_ref.task_id, timeout=300)
                results["steps"]["freesasa"] = sasa_result

                # Filter: binder chain only + SASA > 40 Å²
                # ADC payload must be conjugated to binder, NOT antigen
                all_sites = sasa_result.get("conjugation_sites", [])
                _binder_ch = _binder_chain  # detected at MPNN step (shortest chain in RFD output)
                filtered_sites = [
                    s for s in all_sites
                    if float(s.get("sasa", 0)) > 40.0 and s.get("chain", "") == _binder_ch
                ]
                filtered_sites.sort(key=lambda s: float(s.get("sasa", 0)), reverse=True)
                top_sites = filtered_sites[:3]
                adc_design["conjugation_sites"] = top_sites

                if not top_sites:
                    raise RuntimeError("No conjugation sites with SASA > 40 Å² found")

                top_site = top_sites[0]
                site_residue = top_site["residue"]
                site_type = top_site.get("type", "")

                logger.info(
                    "[binder_pipeline] FreeSASA top site: %s (SASA=%.1f, type=%s)",
                    site_residue, float(top_site.get("sasa", 0)), site_type,
                )
            except Exception as e:
                logger.warning("[binder_pipeline] Step 4 (freesasa) failed: %s", e)
                adc_design["error"] = f"step4_freesasa_failed: {e}"
                adc_design["partial"] = True
                site_residue = None
                site_type = ""

            # ── Step 5: Linker selection ────────────────────────────────────
            linker_name = None
            linker_smiles = None
            try:
                task.progress = 85
                task.progress_msg = "Step 5/7: Linker selection..."

                # Choose chemistry based on conjugation site type
                if site_type and "Cys" in site_type:
                    preferred_chem = "maleimide_thiol"
                else:
                    preferred_chem = "nhs_amine"  # default for Lys

                from routers.adc import linker_select
                from schemas.models import LinkerSelectRequest
                linker_result = await linker_select(LinkerSelectRequest(
                    cleavable=True,
                    reaction_type=preferred_chem,
                    compatible_payload="MMAE",
                    max_results=3,
                ))
                results["steps"]["linker_select"] = linker_result

                recommended = linker_result.get("recommended_linkers", [])
                if recommended:
                    top_linker = recommended[0]
                    linker_name = top_linker.get("name", top_linker.get("linker_name"))
                    linker_smiles = top_linker.get("smiles", top_linker.get("linker_smiles"))
                    adc_design["linker"] = linker_name
                    adc_design["linker_smiles"] = linker_smiles
                    adc_design["reaction_type"] = preferred_chem
                    logger.info("[binder_pipeline] Selected linker: %s (%s)", linker_name, preferred_chem)
                else:
                    raise RuntimeError("No matching linkers found")
            except Exception as e:
                logger.warning("[binder_pipeline] Step 5 (linker_select) failed: %s", e)
                if "error" not in adc_design:
                    adc_design["error"] = f"step5_linker_failed: {e}"
                    adc_design["partial"] = True

            # ── Step 6: Fetch payload (MMAE) ────────────────────────────────
            payload_smiles = None
            try:
                task.progress = 90
                task.progress_msg = "Step 6/7: Fetching MMAE payload..."

                from routers.structure_prediction import fetch_molecule, FetchMoleculeRequest
                mmae_result = await fetch_molecule(FetchMoleculeRequest(query="MMAE"))
                results["steps"]["fetch_payload"] = mmae_result

                payload_smiles = mmae_result.get("smiles")
                adc_design["payload"] = "MMAE"
                adc_design["payload_smiles"] = payload_smiles
                adc_design["payload_mw"] = mmae_result.get("molecular_weight")
                logger.info("[binder_pipeline] MMAE SMILES: %s", (payload_smiles or "")[:60])
            except Exception as e:
                logger.warning("[binder_pipeline] Step 6 (fetch_molecule MMAE) failed: %s", e)
                if "error" not in adc_design:
                    adc_design["error"] = f"step6_payload_failed: {e}"
                    adc_design["partial"] = True

            # ── Step 7: RDKit conjugation ───────────────────────────────────
            try:
                task.progress = 95
                task.progress_msg = "Step 7/7: RDKit ADC conjugation..."

                if not all([site_residue, linker_smiles, payload_smiles, af3_pdb]):
                    missing = []
                    if not site_residue: missing.append("conjugation_site")
                    if not linker_smiles: missing.append("linker_smiles")
                    if not payload_smiles: missing.append("payload_smiles")
                    if not af3_pdb: missing.append("antibody_pdb")
                    raise RuntimeError(f"Missing inputs: {', '.join(missing)}")

                from routers.adc import rdkit_conjugate
                from schemas.models import RDKitConjugateRequest
                conj_ref = await rdkit_conjugate(RDKitConjugateRequest(
                    job_name=f"{req.job_name}_conjugate",
                    antibody_pdb=af3_pdb,
                    conjugation_site=site_residue,
                    linker_smiles=linker_smiles,
                    payload_smiles=payload_smiles,
                    linker_name=linker_name or "",
                    reaction_type="auto",
                ))
                conj_result = await _wait_for_task(conj_ref.task_id, timeout=300)
                results["steps"]["rdkit_conjugate"] = conj_result

                adc_design["adc_smiles"] = conj_result.get("adc_smiles")
                adc_design["dar"] = 4  # standard ADC DAR
                adc_design["adc_structure_path"] = conj_result.get("output_sdf")
                adc_design["reaction_type_used"] = conj_result.get("reaction_type_used")
                adc_design["covalent"] = conj_result.get("covalent")
                logger.info("[binder_pipeline] ADC conjugation complete: %s", conj_result.get("reaction_type_used"))
            except Exception as e:
                logger.warning("[binder_pipeline] Step 7 (rdkit_conjugate) failed: %s", e)
                if "error" not in adc_design:
                    adc_design["error"] = f"step7_conjugate_failed: {e}"
                    adc_design["partial"] = True

            # Assemble final adc_design summary
            adc_design["nanobody_sequence"] = nanobody_seq
            adc_design["iptm"] = best_iptm
            if site_residue:
                adc_design["conjugation_site"] = site_residue

        results["adc_design"] = adc_design

        task.progress = 100
        task.progress_msg = (
            f"Binder design + ADC pipeline complete. {len(backbone_pdbs)} backbones, "
            f"{len(all_sequences)} sequence sets, "
            f"{af3_val.get('passed', 0)}/{af3_val.get('total_tested', 0)} AF3-validated."
        )
        return results

    task = await task_manager.submit("binder_design_pipeline", req.model_dump(), _run)
    return TaskRef(task_id=task.task_id, status=task.status, tool="binder_design_pipeline",
                   poll_url=f"/api/v1/tasks/{task.task_id}")


# ─── Target Tier Classification ───────────────────────────────────────────────

# Known antibody-antigen complex structures for Tier 1 hotspot extraction
# IMPORTANT: All chain IDs MUST be verified from actual PDB structure using gemmi.
# Never guess ligand_chains — always check: gemmi.read_structure(pdb) → list all chains + sizes.
# Lesson: 1YY9 EGFR had ligand_chains=["B"] (wrong), actual cetuximab is ["C","D"].
KNOWN_COMPLEXES = {
    "HER2":  {"pdb": "1N8Z", "receptor_chain": "C", "ligand_chains": ["A", "B"], "source": "trastuzumab"},        # C(581aa) vs A(214)+B(220) Fab
    "ERBB2": {"pdb": "1N8Z", "receptor_chain": "C", "ligand_chains": ["A", "B"], "source": "trastuzumab"},
    "PD-L1": {"pdb": "5XXY", "receptor_chain": "A", "ligand_chains": ["H", "L"], "source": "atezolizumab"},       # A(99aa PD-L1) vs H(208)+L(211) Fab
    "EGFR":  {"pdb": "1YY9", "receptor_chain": "A", "ligand_chains": ["C", "D"], "source": "cetuximab"},          # A(613aa) vs C(211)+D(220) Fab
    "VEGF":  {"pdb": "1BJ1", "receptor_chain": "V", "ligand_chains": ["H", "L"], "source": "bevacizumab"},        # V(94aa VEGF) vs H(218)+L(213) Fab; W/J/K are symmetry mates
    "TNF":   {"pdb": "3WD5", "receptor_chain": "A", "ligand_chains": ["H", "L"], "source": "adalimumab"},         # A(152aa TNF) vs H(214)+L(213) Fab
    # CD20 6Y4J removed — only 1 chain (A, 426aa), no antibody in this structure
}


def _classify_target_tier(target_name: str, pdb_id: str, rag_result: dict) -> dict:
    """
    Classify target into tiers for hotspot identification strategy.
    Tier 1: Known complex PDB exists → structural_database path (most reliable)
    Tier 2: Homologous structure found via RAG → rag_guided path
    Tier 3: Novel target → computational_prediction path (DiscoTope3 + IEDB + RAG)
    """
    if target_name:
        target_upper = target_name.upper().strip()
        for key, val in KNOWN_COMPLEXES.items():
            if key in target_upper:
                return {"tier": 1, "method": "structural_database", **val}

    # Check RAG for PDB mentions of antibody complexes
    rag_pdbs = rag_result.get("pdb_ids", [])
    if rag_pdbs:
        return {"tier": 2, "method": "rag_guided", "pdb": rag_pdbs[0]}

    return {"tier": 3, "method": "computational_prediction"}


# ─── Pocket-Guided Binder + ADC Pipeline ─────────────────────────────────────

@router.post("/pocket-guided-binder",
             response_model=TaskRef,
             summary="Pocket-guided binder design + ADC: 6D pocket scoring (P2Rank+SASA+conservation+RAG+electrostatics+DiscoTope3) → Qwen selection → RFdiffusion → MPNN → AF3 → ADC")
async def pocket_guided_binder_pipeline(req: PocketGuidedBinderPipelineRequest):
    """
    Automated pocket-guided binder design + ADC construction (16 steps):
    1. fetch_pdb              — download target from RCSB
    2. RAG literature search  — find binding site evidence, extract residues
    3. fpocket + P2Rank + DiscoTope3 — detect pockets + predict B-cell epitopes (parallel)
    4. FreeSASA per pocket    — surface exposure scoring
    5. Conservation + electrostatics — B-factor and charge analysis
    6. 6D composite scoring + Qwen pocket selection (P2Rank/SASA/conservation/RAG/electrostatics/epitope)
    7. DiffDock (optional)    — druggability reference on selected pocket
    8a. RFdiffusion           — binder backbone design using epitope-enriched hotspots
    8b. BindCraft (parallel)  — end-to-end binder design (if container available)
    9. ProteinMPNN            — sequence design on RFdiffusion backbones
    10. Merge candidates      — combine MPNN + BindCraft designs
    11. AlphaFold3            — validate top designs (ipTM filter)
    12–16. ADC construction   — FreeSASA conjugation → Linker → Payload → RDKit
    """

    if req.dry_run:
        mock = {
            "job_name": req.job_name, "dry_run": True,
            "steps": {k: {"status": "skipped (dry_run)"} for k in [
                "fetch_pdb", "rag_search", "fpocket", "p2rank", "discotope3",
                "pocket_sasa", "pocket_scoring", "qwen_selection",
                "diffdock", "rfdiffusion", "bindcraft", "proteinmpnn",
                "af3_validation", "freesasa", "linker_select",
                "fetch_payload", "rdkit_conjugate",
            ]},
            "pocket_scores": {},
            "selected_pocket": {"id": None, "center": None, "hotspots": [], "reason": "dry_run"},
            "diffdock_reference": None,
            "pocket_analysis": {"hotspot_residues": [], "top_pocket": None},
            "adc_design": {"dry_run": True, "dar": 4},
        }
        task = await task_manager.submit(
            "pocket_guided_binder_pipeline", req.model_dump(),
            _make_dry_run_coroutine(mock),
        )
        return TaskRef(task_id=task.task_id, status=task.status,
                       tool="pocket_guided_binder_pipeline",
                       poll_url=f"/api/v1/tasks/{task.task_id}")

    async def _run(task):
        results = {"job_name": req.job_name, "steps": {}}

        # ── Step 1: Fetch PDB ─────────────────────────────────────────────
        task.progress = 2
        task.progress_msg = "Step 1/16: Fetching target PDB..."
        from routers.structure_prediction import fetch_pdb, FetchPDBRequest
        fetch_result = await fetch_pdb(FetchPDBRequest(
            pdb_id=req.pdb_id, chains=req.chains,
        ))
        target_pdb = fetch_result["output_pdb"]
        results["steps"]["fetch_pdb"] = fetch_result
        logger.info("[pocket_guided] fetch_pdb(%s) → %s", req.pdb_id, target_pdb)

        # ── Step 0: Target tier classification ──────────────────────────
        # Determine hotspot strategy BEFORE doing any computation
        target_name = req.target_name or req.pdb_id

        # Quick RAG lookup for tier classification (also used in later steps)
        task.progress = 3
        task.progress_msg = "Step 2/16: RAG literature search + tier classification..."
        rag_context = await _rag_search_pocket_context(target_name, req.pdb_id)
        rag_residues_raw = rag_context.get("residues", [])
        results["steps"]["rag_search"] = {
            "n_papers": len(rag_context.get("text", "").split(". ")) if rag_context.get("text") else 0,
            "residues_found": rag_residues_raw,
            "domains": rag_context.get("domains", []),
            "kd_values": rag_context.get("kd_values", []),
        }
        logger.info("[pocket_guided] RAG: %d residues, %d domains found",
                     len(rag_residues_raw), len(rag_context.get("domains", [])))

        # Classify target tier
        tier_info = _classify_target_tier(target_name, req.pdb_id, rag_context)
        results["target_tier"] = tier_info["tier"]
        results["hotspot_method"] = tier_info["method"]
        logger.info("[pocket_guided] Target tier: %d (%s)", tier_info["tier"], tier_info["method"])

        # ── Tier 1: Extract hotspots from known complex ──────────────────
        if tier_info["tier"] == 1:
            task.progress = 5
            task.progress_msg = (
                f"Tier 1: Known complex {tier_info['pdb']} ({tier_info['source']}) detected — "
                f"extracting interface residues..."
            )

            # Fetch the known complex PDB
            complex_fetch = await fetch_pdb(FetchPDBRequest(pdb_id=tier_info["pdb"]))
            complex_pdb_path = complex_fetch["output_pdb"]

            # Extract interface residues
            from routers.structure_prediction import extract_interface_residues, ExtractInterfaceRequest
            interface_result = await extract_interface_residues(ExtractInterfaceRequest(
                job_name=f"{req.job_name}_interface",
                complex_pdb=complex_pdb_path,
                receptor_chain=tier_info["receptor_chain"],
                ligand_chains=tier_info["ligand_chains"],
                cutoff_angstrom=5.0,
                top_n=8,
            ))

            hotspot_residues = interface_result["interface_residues"]
            results["steps"]["interface_extraction"] = {
                "status": "completed",
                "source_pdb": tier_info["pdb"],
                "source_antibody": tier_info["source"],
                "interface_residues": hotspot_residues,
                "num_contacts": interface_result.get("num_contacts", {}),
                "total_interface": interface_result.get("total_interface", 0),
            }
            results["hotspot_source_pdb"] = tier_info["pdb"]

            # Remap chain IDs if the target PDB uses a different chain than the complex
            # e.g. 1N8Z uses chain C for HER2, but user might fetch 2A91 chain A
            antigen_chain = req.antigen_chain or req.chains or "A"
            if antigen_chain != tier_info["receptor_chain"]:
                hotspot_residues = [
                    f"{antigen_chain}{r.lstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"
                    for r in hotspot_residues
                ]
                logger.info("[pocket_guided] Remapped hotspots from chain %s → %s: %s",
                            tier_info["receptor_chain"], antigen_chain, hotspot_residues)

            # Skip steps 3-6: no need for pocket detection or epitope prediction
            results["epitope_override_triggered"] = False  # Not an override — it's tier 1
            results["steps"]["fpocket"] = {"status": "skipped (tier_1)"}
            results["steps"]["p2rank"] = {"status": "skipped (tier_1)"}
            results["steps"]["discotope3"] = {"status": "skipped (tier_1)"}
            results["steps"]["pocket_sasa"] = {"status": "skipped (tier_1)"}
            results["steps"]["pocket_scoring"] = {"status": "skipped (tier_1)"}
            results["steps"]["qwen_selection"] = {"status": "skipped (tier_1)"}
            results["pocket_scores"] = {}
            results["selected_pocket"] = {
                "id": None,
                "center": None,
                "hotspots": hotspot_residues,
                "epitope_enriched": hotspot_residues,
                "reason": (
                    f"tier_1_structural_database: {len(hotspot_residues)} interface residues "
                    f"from {tier_info['pdb']} ({tier_info['source']} complex)"
                ),
                "composite_score": None,
            }
            results["pocket_analysis"] = {
                "hotspot_residues": hotspot_residues,
                "top_pocket": None,
                "method": "structural_database",
                "epitope_residues_added": len(hotspot_residues),
            }
            results["hotspot_residues"] = hotspot_residues

            logger.info("[pocket_guided] Tier 1: %d interface hotspots from %s: %s",
                        len(hotspot_residues), tier_info["pdb"], hotspot_residues)
            task.progress = 15
            task.progress_msg = (
                f"Tier 1: {len(hotspot_residues)} interface residues from "
                f"{tier_info['pdb']} ({tier_info['source']}) → skip to binder design"
            )

        # ── Steps 3-6: Pocket detection + scoring (skip for Tier 1) ─────
        if tier_info["tier"] == 1:
            # Tier 1 already set hotspot_residues via interface extraction
            hotspot_residues = results["hotspot_residues"]
        else:
            # Tier 2/3: proceed with pocket detection + epitope scoring
            pass

        from routers.pocket_analysis import run_fpocket, run_p2rank
        from routers.immunology import run_discotope3

        # For tier 1, steps 3-6 are skipped (hotspots already set by interface extraction)
        dt3_epitopes = []

        if tier_info["tier"] != 1:
            # ── Step 3: fpocket + P2Rank pocket detection ────────────────────
            task.progress = 6
            task.progress_msg = "Step 3/16: Detecting pockets + predicting epitopes (fpocket + P2Rank + DiscoTope3)..."

            pdb_basename = os.path.basename(target_pdb).lower()
            dt3_struc_type = "alphafold" if ("af_" in pdb_basename or "alphafold" in pdb_basename) else "solved"

            # Launch fpocket, P2Rank, and DiscoTope3 in parallel
            fpocket_ref = await run_fpocket(FpocketRequest(
                job_name=f"{req.job_name}_fpocket", input_pdb=target_pdb,
            ))
            p2rank_ref = await run_p2rank(P2RankRequest(
                job_name=f"{req.job_name}_p2rank", input_pdb=target_pdb,
            ))
            dt3_ref = None
            try:
                dt3_ref = await run_discotope3(DiscoTope3Request(
                    job_name=f"{req.job_name}_discotope3",
                    input_pdb=target_pdb,
                    struc_type=dt3_struc_type,
                    calibrated_score_epi_threshold=0.90,
                ))
                logger.info("[pocket_guided] DiscoTope3 launched: %s", dt3_ref.task_id)
            except Exception as e:
                logger.warning("[pocket_guided] DiscoTope3 launch failed (non-fatal): %s", e)

            # Wait for all three in parallel
            async def _wait_fpocket():
                return await _wait_for_task(fpocket_ref.task_id, timeout=300)

            async def _wait_p2rank():
                return await _wait_for_task(p2rank_ref.task_id, timeout=300)

            async def _wait_dt3():
                if not dt3_ref:
                    return None
                try:
                    return await _wait_for_task(dt3_ref.task_id, timeout=600)
                except Exception as e:
                    logger.warning("[pocket_guided] DiscoTope3 failed (non-fatal): %s", e)
                    return None

            fpocket_result, p2rank_result, dt3_result = await asyncio.gather(
                _wait_fpocket(), _wait_p2rank(), _wait_dt3()
            )

            p2rank_pockets = p2rank_result.get("pockets", [])[:5]  # top 5
            results["steps"]["fpocket"] = {
                "n_pockets": len(fpocket_result.get("pockets", [])),
                "top3": fpocket_result.get("pockets", [])[:3],
            }
            results["steps"]["p2rank"] = {
                "n_pockets": len(p2rank_result.get("pockets", [])),
                "top5": p2rank_pockets,
            }

            # Parse DiscoTope3 epitope residues (non-blocking fallback)
            dt3_epitopes = []
            if dt3_result:
                dt3_epitopes = dt3_result.get("epitopes", [])
                results["steps"]["discotope3"] = {
                    "status": "completed",
                    "num_residues": dt3_result.get("num_residues", 0),
                    "num_epitope_residues": dt3_result.get("num_epitope_residues", 0),
                    "struc_type": dt3_struc_type,
                }
                logger.info("[pocket_guided] DiscoTope3: %d epitope residues from %d total",
                             dt3_result.get("num_epitope_residues", 0),
                             dt3_result.get("num_residues", 0))
            else:
                results["steps"]["discotope3"] = {"status": "failed_or_skipped", "epitope_fallback": 0.0}
                logger.info("[pocket_guided] DiscoTope3 unavailable, epitope_score=0.0 for all pockets")

            logger.info("[pocket_guided] fpocket=%d, P2Rank=%d pockets, DiscoTope3=%d epitopes",
                         len(fpocket_result.get("pockets", [])),
                         len(p2rank_result.get("pockets", [])),
                         len(dt3_epitopes))

            if not p2rank_pockets:
                raise RuntimeError("P2Rank produced no pockets")

            # Parse residues for each pocket
            pocket_data = []
            for pocket in p2rank_pockets:
                residue_ids_raw = (
                    pocket.get("residue_ids", "") or
                    pocket.get(" residue_ids", "") or ""
                )
                residues = _parse_p2rank_residues(residue_ids_raw)
                p2rank_prob = float(pocket.get("probability", pocket.get(" probability", 0)))
                p2rank_score = float(pocket.get("score", pocket.get(" score", 0)))
                center_x = float(pocket.get("center_x", pocket.get(" center_x", 0)))
                center_y = float(pocket.get("center_y", pocket.get(" center_y", 0)))
                center_z = float(pocket.get("center_z", pocket.get(" center_z", 0)))
                pocket_id = int(pocket.get("rank", pocket.get(" rank", 0)))
                pocket_data.append({
                    "pocket_id": pocket_id,
                    "residues": residues,
                    "p2rank_prob": p2rank_prob,
                    "p2rank_score": p2rank_score,
                    "center": [center_x, center_y, center_z],
                })

            # ── Known epitope override check ─────────────────────────────────
            # If DiscoTope3 high-confidence epitopes overlap with RAG literature
            # residues by >= 3, use them directly as hotspots — skip composite scoring.
            # DT3 raw score range is 0-0.5 typically; use top-20% percentile as threshold.
            # RAG residues may use UniProt numbering while PDB uses crystal numbering
            # (offset can be 0-30+), so use fuzzy matching with ±3 tolerance.
            epitope_override_triggered = False
            override_hotspots = []

            if dt3_epitopes and rag_residues_raw:
                def _to_num(r: str) -> str:
                    return r.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_:")

                # Adaptive threshold: top 20% of DT3 scores
                all_dt3_scores = []
                for e in dt3_epitopes:
                    try:
                        all_dt3_scores.append(float(e.get("DiscoTope-3.0_score", 0)))
                    except (ValueError, KeyError):
                        continue
                all_dt3_scores.sort(reverse=True)
                dt3_threshold = all_dt3_scores[max(0, len(all_dt3_scores) // 5)] if all_dt3_scores else 0.15
                dt3_threshold = max(dt3_threshold, 0.10)  # floor at 0.10
                logger.info("[pocket_guided] DT3 adaptive threshold: %.3f (top-20%% of %d residues)",
                             dt3_threshold, len(all_dt3_scores))

                dt3_high = {}  # resnum_int → chain+resid key
                for e in dt3_epitopes:
                    try:
                        if float(e.get("DiscoTope-3.0_score", 0)) >= dt3_threshold:
                            key = f"{e['chain']}{e['res_id']}"
                            num_str = _to_num(key)
                            if num_str.isdigit():
                                dt3_high[int(num_str)] = key
                    except (ValueError, KeyError):
                        continue

                rag_nums_int = set()
                for r in rag_residues_raw:
                    n = _to_num(r)
                    if n.isdigit():
                        rag_nums_int.add(int(n))

                # Fuzzy match: DT3 residue N matches RAG residue M if |N-M| <= 3
                # This handles PDB-vs-UniProt numbering offsets up to ±3
                # For larger offsets, also try common HER2 offset (~22)
                offsets_to_try = [0, -22, -23, 22, 23]  # PDB→UniProt common offsets
                best_overlap_keys = []
                best_overlap_count = 0
                for offset in offsets_to_try:
                    matched_keys = []
                    for dt3_num, dt3_key in dt3_high.items():
                        adjusted = dt3_num + offset
                        # Check if any RAG residue is within ±3 of adjusted position
                        if any(abs(adjusted - rn) <= 3 for rn in rag_nums_int):
                            matched_keys.append(dt3_key)
                    if len(matched_keys) > best_overlap_count:
                        best_overlap_count = len(matched_keys)
                        best_overlap_keys = matched_keys
                        logger.info("[pocket_guided] Override offset=%d: %d matches", offset, len(matched_keys))

                if best_overlap_count >= 2:
                    epitope_override_triggered = True
                    # Sort overlapping residues by DiscoTope3 score (highest first)
                    dt3_score_map = {}
                    for e in dt3_epitopes:
                        try:
                            dt3_score_map[f"{e['chain']}{e['res_id']}"] = float(e.get("DiscoTope-3.0_score", 0))
                        except (ValueError, KeyError):
                            continue
                    best_overlap_keys.sort(key=lambda k: dt3_score_map.get(k, 0), reverse=True)
                    # Spatial clustering: keep tightest cluster within 15Å, max 5 residues
                    override_hotspots = _cluster_hotspots(
                        best_overlap_keys, target_pdb, max_dist=15.0, max_n=5
                    )

                    logger.info("[pocket_guided] known_epitope_override triggered: %s "
                                "(DT3 ∩ RAG = %d total, clustered to %d, threshold=%.3f)",
                                override_hotspots, best_overlap_count, len(override_hotspots), dt3_threshold)

            results["epitope_override_triggered"] = epitope_override_triggered

            if epitope_override_triggered:
                # Skip steps 4-6: directly use literature-validated epitope sites
                hotspot_residues = override_hotspots
                results["steps"]["pocket_sasa"] = {"status": "skipped (epitope_override)"}
                results["steps"]["pocket_scoring"] = {"status": "skipped (epitope_override)"}
                results["steps"]["qwen_selection"] = {"status": "skipped (epitope_override)"}
                results["pocket_scores"] = {}
                results["selected_pocket"] = {
                    "id": None,
                    "center": None,
                    "hotspots": hotspot_residues,
                    "epitope_enriched": hotspot_residues,
                    "reason": f"known_epitope_override: {len(override_hotspots)} residues validated by both DiscoTope3 (score>{dt3_threshold:.2f}) and literature (RAG, fuzzy ±3)",
                    "composite_score": None,
                }
                results["pocket_analysis"] = {
                    "hotspot_residues": hotspot_residues,
                    "top_pocket": None,
                    "method": "known_epitope_override",
                    "epitope_residues_added": len(hotspot_residues),
                }
                logger.info("[pocket_guided] Override: skipping steps 4-6, hotspots=%s", hotspot_residues)
                task.progress = 15
                task.progress_msg = f"Epitope override: {len(hotspot_residues)} literature-validated epitope residues → skip to binder design"

            else:
                # ── Step 4: FreeSASA per pocket ──────────────────────────────
                task.progress = 10
                task.progress_msg = "Step 4/16: Computing surface exposure (FreeSASA)..."
                sasa_per_residue = _compute_freesasa_per_residue(target_pdb)
                results["steps"]["pocket_sasa"] = {
                    "total_residues_computed": len(sasa_per_residue),
                }
                logger.info("[pocket_guided] FreeSASA: %d residues computed", len(sasa_per_residue))

                # ── Step 5: Conservation + Electrostatics ────────────────────
                task.progress = 12
                task.progress_msg = "Step 5/16: Computing conservation & electrostatics..."

                # RAG residues are plain numbers like ['42', '310']
                # _compute_rag_score normalizes both sides to plain numbers for comparison
                rag_residues = rag_residues_raw

                # ── Step 6: Composite scoring + Qwen selection ───────────────
                task.progress = 14
                task.progress_msg = "Step 6/16: Scoring pockets and consulting Qwen..."

                # Run PeSTo once for the whole protein (cached for all pockets)
                _pesto_cache = {}
                pesto_whole = _compute_pesto_score_for_pocket(target_pdb, [], _pesto_cache)

                scored_pockets = []
                for pd in pocket_data:
                    p2rank_score_norm = min(pd["p2rank_prob"], 1.0)
                    sasa_score = _compute_sasa_score_for_pocket(sasa_per_residue, pd["residues"])
                    conservation = _compute_bfactor_conservation(target_pdb, pd["residues"])
                    rag_score = _compute_rag_score(rag_residues, pd["residues"])
                    electrostatics = _compute_electrostatics_from_pdb(target_pdb, pd["residues"])
                    pesto_score = _compute_pesto_score_for_pocket(target_pdb, pd["residues"], _pesto_cache)

                    # PPI-optimized composite (replaces old 6D):
                    # rag(0.30) + pesto_ppi(0.25) + conservation(0.20) + sasa(0.10) + electrostatics(0.15)
                    composite = (
                        rag_score * 0.30 +
                        pesto_score * 0.25 +
                        conservation * 0.20 +
                        sasa_score * 0.10 +
                        electrostatics * 0.15
                    )

                    scored_pockets.append({
                        "pocket_id": pd["pocket_id"],
                        "center": pd["center"],
                        "residues": pd["residues"][:15],
                        "n_residues": len(pd["residues"]),
                        "scores": {
                            "rag": rag_score,
                            "pesto_ppi": pesto_score,
                            "conservation": conservation,
                            "sasa": sasa_score,
                            "electrostatics": electrostatics,
                            "p2rank": round(p2rank_score_norm, 4),  # kept for reference
                            "composite": round(composite, 4),
                        },
                    })

                scored_pockets.sort(key=lambda x: x["scores"]["composite"], reverse=True)
                results["pocket_scores"] = {
                    f"pocket{sp['pocket_id']}": sp["scores"] for sp in scored_pockets
                }

                # Ask Qwen to select pocket
                qwen_selection = await _qwen_select_pocket(
                    scored_pockets, rag_context.get("text", ""), target_name, req.pdb_id
                )
                results["steps"]["qwen_selection"] = qwen_selection

                # Determine selected pocket and hotspots
                selected_id = qwen_selection.get("selected_pocket_id")
                qwen_hotspots = qwen_selection.get("hotspot_residues", [])
                selection_reason = qwen_selection.get("selection_reason", "")

                # Find the selected pocket data; fall back to highest composite if Qwen fails
                selected_pocket_data = None
                for sp in scored_pockets:
                    if sp["pocket_id"] == selected_id:
                        selected_pocket_data = sp
                        break
                if not selected_pocket_data:
                    selected_pocket_data = scored_pockets[0]
                    selected_id = selected_pocket_data["pocket_id"]
                    selection_reason = "Fallback: highest composite score (Qwen selection unavailable)"
                    logger.warning("[pocket_guided] Qwen selection failed, using pocket %d (composite=%.3f)",
                                   selected_id, selected_pocket_data["scores"]["composite"])

                # Use Qwen's hotspots if valid, otherwise take top 6 from selected pocket
                if qwen_hotspots and len(qwen_hotspots) >= 3:
                    hotspot_residues = qwen_hotspots[:6]
                else:
                    # Find full residue list for selected pocket
                    for pd in pocket_data:
                        if pd["pocket_id"] == selected_id:
                            hotspot_residues = pd["residues"][:6]
                            break
                    else:
                        hotspot_residues = selected_pocket_data.get("residues", [])[:6]

                # Enrich hotspots with top DiscoTope3 epitope residues from the selected pocket
                epitope_enriched = []
                if dt3_epitopes:
                    # Get top epitope residues near selected pocket (within 8 Å)
                    selected_residue_set = set()
                    for pd in pocket_data:
                        if pd["pocket_id"] == selected_id:
                            selected_residue_set = set(pd["residues"])
                            break

                    # Parse CA coords for distance check
                    ca_coords = {}
                    try:
                        with open(target_pdb) as f:
                            for line in f:
                                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                                    chain = line[21]
                                    resnum = line[22:27].strip()
                                    ca_coords[f"{chain}{resnum}"] = (
                                        float(line[30:38]), float(line[38:46]), float(line[46:54])
                                    )
                    except Exception:
                        pass

                    cutoff_sq = 8.0 ** 2
                    pocket_coords = [ca_coords[r] for r in selected_residue_set if r in ca_coords]

                    # Rank epitope residues by score, filter those near pocket
                    epi_sorted = sorted(
                        [e for e in dt3_epitopes if float(e.get("DiscoTope-3.0_score", 0)) >= 0.5],
                        key=lambda e: float(e.get("DiscoTope-3.0_score", 0)),
                        reverse=True,
                    )
                    for e in epi_sorted:
                        epi_key = f"{e['chain']}{e['res_id']}"
                        if epi_key in ca_coords and pocket_coords:
                            ex, ey, ez = ca_coords[epi_key]
                            near = any(
                                (ex - px) ** 2 + (ey - py) ** 2 + (ez - pz) ** 2 <= cutoff_sq
                                for px, py, pz in pocket_coords
                            )
                            if near and epi_key not in set(hotspot_residues):
                                epitope_enriched.append(epi_key)
                        if len(epitope_enriched) >= 3:
                            break

                # Merge: original hotspots + up to 3 epitope-enriched residues (cap at 8 total)
                if epitope_enriched:
                    hotspot_residues = (hotspot_residues + epitope_enriched)[:8]
                    logger.info("[pocket_guided] Epitope-enriched hotspots: +%d residues → %s",
                                 len(epitope_enriched), hotspot_residues)

                results["selected_pocket"] = {
                    "id": selected_id,
                    "center": selected_pocket_data["center"],
                    "hotspots": hotspot_residues,
                    "epitope_enriched": epitope_enriched,
                    "reason": selection_reason,
                    "composite_score": selected_pocket_data["scores"]["composite"],
                }
                results["pocket_analysis"] = {
                    "hotspot_residues": hotspot_residues,
                    "top_pocket": selected_pocket_data,
                    "method": "6D_multi_criteria_qwen_epitope",
                    "epitope_residues_added": len(epitope_enriched),
                }
                logger.info("[pocket_guided] Selected pocket %d (composite=%.3f), hotspots: %s",
                             selected_id, selected_pocket_data["scores"]["composite"], hotspot_residues)

        if not hotspot_residues:
            raise RuntimeError("No usable hotspot residues from pocket scoring")

        # Step 7 (DiffDock) removed — small molecule druggability reference
        # is irrelevant for protein binder/ADC design pipelines.
        results["steps"]["diffdock"] = {"status": "removed"}
        results["diffdock_reference"] = None

        # ── Step 8a+8b: RFdiffusion + BindCraft (with auto-clustering) ────
        task.progress = 22
        from routers.protein_design import run_rfdiffusion, run_proteinmpnn

        # Auto-cluster hotspots by spatial proximity
        hotspot_clusters = _multi_cluster_hotspots(hotspot_residues, target_pdb, distance_threshold=15.0)
        results["hotspot_clusters"] = hotspot_clusters
        n_clusters = len(hotspot_clusters)

        if n_clusters > 1:
            logger.info("[pocket_guided] Hotspots split into %d clusters: %s", n_clusters, hotspot_clusters)
            task.progress_msg = f"Step 8/16: {n_clusters} epitope clusters → designing separately..."
        else:
            hotspot_str = ",".join(hotspot_residues)
            task.progress_msg = f"Step 8/16: RFdiffusion + BindCraft parallel (hotspots: {hotspot_str})..."

        # 8a: Launch RFdiffusion per cluster
        rfd_refs = []
        designs_per_cluster = max(3, req.num_rfdiffusion_designs // n_clusters)
        for ci, cluster in enumerate(hotspot_clusters):
            cluster_str = ",".join(cluster)
            suffix = f"_c{ci}" if n_clusters > 1 else ""
            rfd_ref = await run_rfdiffusion(RFdiffusionRequest(
                job_name=f"{req.job_name}_rfd{suffix}",
                mode="binder_design",
                target_pdb=target_pdb,
                hotspot_residues=cluster_str,
                num_designs=designs_per_cluster,
            ))
            rfd_refs.append((ci, cluster_str, rfd_ref))
            logger.info("[pocket_guided] RFdiffusion cluster %d: %s (%d designs)",
                        ci, cluster_str, designs_per_cluster)

        # 8b: Launch BindCraft on first (largest) cluster
        bindcraft_ref = None
        bindcraft_result = None
        try:
            from routers.protein_design import run_bindcraft
            from schemas.models import BindCraftRequest
            bc_hotspots = ",".join(hotspot_clusters[0])
            bindcraft_ref = await run_bindcraft(BindCraftRequest(
                job_name=f"{req.job_name}_bindcraft",
                target_pdb=target_pdb,
                target_hotspots=bc_hotspots,
                num_designs=10,
            ))
            logger.info("[pocket_guided] BindCraft launched: %s (cluster 0)", bindcraft_ref.task_id)
        except Exception as e:
            logger.warning("[pocket_guided] BindCraft launch failed (non-fatal): %s", e)
            results["steps"]["bindcraft"] = {"status": "failed", "error": str(e)}

        # Wait for all RFdiffusion tasks — collect backbones from all clusters
        backbone_pdbs = []
        rfd_results_all = []
        for ci, cluster_str, rfd_ref in rfd_refs:
            try:
                rfd_result = await _wait_for_task(rfd_ref.task_id, timeout=7200)
                rfd_results_all.append(rfd_result)
                backbone_pdbs.extend(rfd_result.get("pdb_files", []))
            except Exception as rfd_err:
                logger.warning("[pocket_guided] RFdiffusion cluster %d failed: %s", ci, rfd_err)
                from pathlib import Path
                suffix = f"_c{ci}" if n_clusters > 1 else ""
                rfd_output_dir = os.path.join(settings.OUTPUT_DIR, f"{req.job_name}_rfd{suffix}", "rfdiffusion")
                partial = [str(p) for p in Path(rfd_output_dir).glob("*.pdb")] if os.path.isdir(rfd_output_dir) else []
                backbone_pdbs.extend(partial)
                rfd_results_all.append({"status": "partial", "error": str(rfd_err), "pdb_files": partial})

        results["steps"]["rfdiffusion"] = {
            "status": "completed" if backbone_pdbs else "failed",
            "pdb_files": backbone_pdbs,
            "n_clusters": n_clusters,
            "cluster_results": rfd_results_all,
        }
        if not backbone_pdbs:
            raise RuntimeError("RFdiffusion failed: 0 designs across all clusters")
        logger.info("[pocket_guided] RFdiffusion: %d backbones from %d clusters", len(backbone_pdbs), n_clusters)

        # ── Step 9: ProteinMPNN on RFdiffusion backbones ─────────────────
        task.progress = 40
        task.progress_msg = f"Step 9/16: ProteinMPNN on {len(backbone_pdbs)} backbones..."
        # Detect binder chain from RFdiffusion backbone
        _binder_chain2 = "B"
        try:
            import gemmi as _gemmi
            _st2 = _gemmi.read_structure(backbone_pdbs[0])
            _ch2 = [(c.name, len([r for r in c if r.entity_type == _gemmi.EntityType.Polymer])) for c in _st2[0]]
            _ch2.sort(key=lambda x: x[1])
            _binder_chain2 = _ch2[0][0]
            logger.info("[pocket_guided] Binder chain: %s (%d res)", _binder_chain2, _ch2[0][1])
        except Exception:
            pass

        all_sequences = []
        for i, pdb in enumerate(backbone_pdbs[:20]):
            mpnn_ref = await run_proteinmpnn(ProteinMPNNRequest(
                job_name=f"{req.job_name}_mpnn_{i}",
                input_pdb=pdb,
                num_sequences=req.num_mpnn_sequences,
                chains_to_design=_binder_chain2,
            ))
            mpnn_result = await _wait_for_task(mpnn_ref.task_id, timeout=600)
            all_sequences.append(mpnn_result)
        results["steps"]["proteinmpnn"] = {"designs": len(all_sequences)}

        # Collect BindCraft results (if launched)
        bindcraft_designs = []
        if bindcraft_ref:
            try:
                bindcraft_result = await _wait_for_task(bindcraft_ref.task_id, timeout=7200)
                bc_pdbs = bindcraft_result.get("pdb_files", [])
                results["steps"]["bindcraft"] = {
                    "status": "completed",
                    "n_designs": len(bc_pdbs),
                    "output_dir": bindcraft_result.get("output_dir"),
                }
                # Extract sequences from BindCraft PDBs for AF3 validation
                for bc_pdb in bc_pdbs[:10]:
                    seq = _extract_sequence_from_pdb(bc_pdb)
                    if seq:
                        bindcraft_designs.append({
                            "sequence": seq, "score": None,
                            "source": "bindcraft", "pdb": bc_pdb,
                        })
                logger.info("[pocket_guided] BindCraft: %d designs, %d sequences extracted",
                            len(bc_pdbs), len(bindcraft_designs))
            except Exception as e:
                logger.warning("[pocket_guided] BindCraft failed (non-fatal): %s", e)
                results["steps"]["bindcraft"] = {"status": "failed", "error": str(e)}
        elif "bindcraft" not in results.get("steps", {}):
            results["steps"]["bindcraft"] = {"status": "skipped"}

        # ── Step 10: Collect all candidates ──────────────────────────────
        af3_val = {"validated_designs": [], "total_tested": 0, "passed": 0,
                   "best_iptm": None, "best_sequence": None}
        if req.run_af3_validation:
            task.progress = 50
            task.progress_msg = "Step 10/16: Collecting candidates for filtering..."
            target_seq = _extract_sequence_from_pdb(target_pdb)

            if target_seq:
                all_designs = []
                # Collect RFdiffusion+MPNN designs
                for step_idx, step_result in enumerate(all_sequences):
                    seqs = step_result.get("sequences", [])
                    if not seqs and step_result.get("fasta_file"):
                        seqs = _parse_mpnn_fasta(step_result["fasta_file"])
                    for seq_info in seqs:
                        seq_info["_backbone_idx"] = step_idx
                        all_designs.append(seq_info)
                # Merge BindCraft designs (score=None sorts to end; take at most 2)
                for bc_design in bindcraft_designs[:2]:
                    bc_design["_backbone_idx"] = "bindcraft"
                    all_designs.append(bc_design)

                all_designs.sort(key=lambda x: x.get("score") if x.get("score") is not None else float("inf"))
                n_total = len(all_designs)
                n_mpnn = n_total - len(bindcraft_designs)
                logger.info("[pocket_guided] Collected %d candidates (%d MPNN + %d BindCraft)",
                            n_total, n_mpnn, len(bindcraft_designs))

                # ── Step 10a: ESM2 pseudo-perplexity filter ──────────────
                esm_filtered = all_designs  # default: no filter
                try:
                    task.progress = 52
                    task.progress_msg = f"Step 10a/16: ESM2 scoring {min(50, len(all_designs))} candidates..."
                    from routers.ml_tools import run_esm_score
                    from schemas.models import ESMScoreRequest

                    # Score top 50 sequences by MPNN score
                    esm_candidates = all_designs[:50]
                    esm_seqs = [d.get("sequence", "") for d in esm_candidates if d.get("sequence")]
                    if esm_seqs:
                        esm_ref = await run_esm_score(ESMScoreRequest(
                            job_name=f"{req.job_name}_esm_filter",
                            sequences=esm_seqs,
                        ))
                        esm_result = await _wait_for_task(esm_ref.task_id, timeout=1800)
                        esm_scores = esm_result.get("scores", [])

                        # Attach PPL scores and filter PPL < 15
                        for i, score_info in enumerate(esm_scores):
                            if i < len(esm_candidates):
                                esm_candidates[i]["esm_ppl"] = score_info.get("pseudo_perplexity", 999)

                        esm_filtered = [d for d in esm_candidates if d.get("esm_ppl", 999) < 15]
                        # Keep BindCraft designs regardless of PPL
                        for d in all_designs:
                            if d.get("source") == "bindcraft" and d not in esm_filtered:
                                esm_filtered.append(d)
                        esm_filtered.sort(key=lambda x: x.get("esm_ppl", 999))
                        results["steps"]["esm_filter"] = {
                            "status": "completed",
                            "input": len(esm_seqs),
                            "passed_ppl15": len(esm_filtered),
                            "best_ppl": min(d.get("esm_ppl", 999) for d in esm_filtered) if esm_filtered else None,
                        }
                        logger.info("[pocket_guided] ESM2 filter: %d → %d (PPL<15)",
                                    len(esm_seqs), len(esm_filtered))
                except Exception as e:
                    logger.warning("[pocket_guided] ESM2 filter failed (non-fatal): %s", e)
                    results["steps"]["esm_filter"] = {"status": "failed", "error": str(e)}

                # ── Step 10b: IgFold structure filter ────────────────────
                # IgFold uses AntiBERTy — only valid for antibody/nanobody sequences.
                # For de novo binders (RFdiffusion), skip IgFold and use ESM2 PPL only.
                _use_igfold = getattr(req, 'binder_type', 'de_novo') in ('nanobody', 'antibody')
                igfold_filtered = esm_filtered  # default: no filter

                if not _use_igfold:
                    # De novo binder: skip IgFold, take top 10 by ESM2 PPL
                    igfold_filtered = esm_filtered[:10]
                    results["steps"]["igfold_filter"] = {
                        "status": "skipped (de_novo binder — IgFold only valid for antibody sequences)",
                        "input": len(esm_filtered),
                        "top_n": len(igfold_filtered),
                    }
                    logger.info("[pocket_guided] IgFold skipped (binder_type=%s): top %d by ESM2 PPL",
                                getattr(req, 'binder_type', 'de_novo'), len(igfold_filtered))
                else:
                    try:
                        task.progress = 55
                        n_igfold = min(30, len(esm_filtered))
                        task.progress_msg = f"Step 10b/16: IgFold structure prediction on {n_igfold} candidates..."
                        from routers.immunology import run_igfold
                        from schemas.models import IgFoldRequest

                        igfold_results = []
                        for i, design in enumerate(esm_filtered[:n_igfold]):
                            seq = design.get("sequence", "")
                            if not seq:
                                continue
                            try:
                                ig_ref = await run_igfold(IgFoldRequest(
                                    job_name=f"{req.job_name}_igfold_{i}",
                                    sequences={"H": seq},
                                ))
                                ig_result = await _wait_for_task(ig_ref.task_id, timeout=120)
                                plddt = ig_result.get("mean_plddt") or 0
                                design["igfold_plddt"] = plddt
                                design["igfold_pdb"] = ig_result.get("output_pdb")
                                igfold_results.append(design)
                            except Exception as e:
                                logger.warning("[pocket_guided] IgFold %d failed: %s", i, e)
                                design["igfold_plddt"] = 0
                                igfold_results.append(design)

                        # Filter pLDDT > 70, keep top 10
                        igfold_passed = [d for d in igfold_results if d.get("igfold_plddt", 0) > 70]
                        # Keep BindCraft designs regardless
                        for d in igfold_results:
                            if d.get("source") == "bindcraft" and d not in igfold_passed:
                                igfold_passed.append(d)
                        igfold_passed.sort(key=lambda x: x.get("igfold_plddt", 0), reverse=True)
                        igfold_filtered = igfold_passed[:10]
                        results["steps"]["igfold_filter"] = {
                            "status": "completed",
                            "input": n_igfold,
                            "passed_plddt70": len(igfold_passed),
                            "top_n": len(igfold_filtered),
                            "best_plddt": max((d.get("igfold_plddt", 0) for d in igfold_filtered), default=0),
                        }
                        logger.info("[pocket_guided] IgFold filter: %d → %d (pLDDT>70) → top %d",
                                    n_igfold, len(igfold_passed), len(igfold_filtered))
                    except Exception as e:
                        logger.warning("[pocket_guided] IgFold filter failed (non-fatal): %s", e)
                        results["steps"]["igfold_filter"] = {"status": "failed", "error": str(e)}

                # ── Step 11: AF3 validation on filtered candidates ───────
                top_n = min(10, len(igfold_filtered))
                top_designs = igfold_filtered[:top_n]
                logger.info("[pocket_guided] AF3 candidates: %d (after ESM2+IgFold funnel from %d total)",
                            top_n, n_total)

                from routers.structure_prediction import predict_alphafold3
                # Get antigen domain regions for AF3 (domain truncation)
                _tgt_name = req.target_name or req.pdb_id or req.job_name
                _hs_nums = []
                for h in hotspot_residues:
                    try: _hs_nums.append(int(re.sub(r'[A-Za-z_:]', '', str(h))))
                    except: pass
                antigen_regions = _get_af3_antigen_regions(_tgt_name, target_seq, _hs_nums if _hs_nums else None)
                # Use first (best) region for all designs
                af3_antigen_seq = antigen_regions[0]["sequence"]
                af3_domain = antigen_regions[0]["domain_name"]
                logger.info("[pocket_guided] AF3 antigen: %s (%daa, full=%daa)",
                            af3_domain, len(af3_antigen_seq), len(target_seq))

                validated = []
                for i, design in enumerate(top_designs):
                    task.progress = 55 + int(15 * i / max(top_n, 1))
                    task.progress_msg = f"Step 11/16: AF3 {i+1}/{top_n} vs {af3_domain} ({len(af3_antigen_seq)}aa)..."
                    binder_seq = design.get("sequence", "")
                    if not binder_seq:
                        continue
                    af3_ref = await predict_alphafold3(AlphaFold3Request(
                        job_name=f"{req.job_name}_af3_val_{i}",
                        chains=[
                            AF3Chain(type="protein", sequence=binder_seq),
                            AF3Chain(type="protein", sequence=af3_antigen_seq),
                        ],
                        num_seeds=3,
                    ))
                    try:
                        af3_result = await _wait_for_af3_task(af3_ref.task_id)
                    except Exception as e:
                        validated.append({"rank": i+1, "sequence": binder_seq,
                                          "iptm": None, "passed": False, "error": str(e),
                                          "domain": af3_domain})
                        if i < top_n - 1:
                            await asyncio.sleep(5)
                        continue

                    iptm = None
                    for cf in af3_result.get("confidence_files", []):
                        try:
                            with open(cf) as f:
                                conf = json.load(f)
                            iptm = conf.get("iptm", conf.get("ipTM"))
                            if iptm is not None:
                                break
                        except Exception:
                            continue

                    passed = iptm is not None and iptm >= 0.6
                    validated.append({
                        "rank": i+1, "sequence": binder_seq,
                        "mpnn_score": design.get("score"),
                        "iptm": round(iptm, 4) if iptm is not None else None,
                        "domain": af3_domain,
                        "antigen_length": len(af3_antigen_seq),
                        "passed": passed,
                        "af3_dir": os.path.dirname(af3_result.get("cif_files", [""])[0]) if af3_result.get("cif_files") else None,
                    })
                    if i < top_n - 1:
                        await asyncio.sleep(5)

                # ── ipSAE verification (detect false positives) ──
                from routers.structure_prediction import ipsae_score as _ipsae2, IpSAERequest as _IpSAEReq2
                for d in validated:
                    if d.get("af3_dir") and d.get("iptm") is not None:
                        try:
                            ipsae_r = await _ipsae2(_IpSAEReq2(
                                job_name=f"{req.job_name}_ipsae_{d['rank']}",
                                af3_output_dir=d["af3_dir"],
                            ))
                            pairs = ipsae_r.get("chain_pairs", [])
                            d["ipsae"] = round(max((p.get("ipSAE", 0) for p in pairs), default=0), 4)
                            logger.info("[pocket_guided] Design %d: ipTM=%.3f, ipSAE=%.4f",
                                        d["rank"], d["iptm"], d["ipsae"])
                            if d["ipsae"] < 0.01 and d.get("passed"):
                                d["passed"] = False
                                d["confidence"] = "false_positive"
                                logger.warning("[pocket_guided] Design %d: ipSAE=0 → false positive!", d["rank"])
                        except Exception as e:
                            d["ipsae"] = None
                            logger.warning("[pocket_guided] ipSAE failed for design %d: %s", d["rank"], e)

                validated.sort(key=lambda x: x["iptm"] if x["iptm"] is not None else -1, reverse=True)
                passed_list = [d for d in validated if d.get("passed")]
                best = validated[0] if validated else None
                af3_val = {
                    "validated_designs": validated,
                    "total_tested": len(validated),
                    "passed": len(passed_list),
                    "best_iptm": best["iptm"] if best else None,
                    "best_ipsae": best.get("ipsae") if best else None,
                    "best_sequence": best["sequence"] if best else None,
                }
            else:
                af3_val["error"] = "cannot_extract_target_sequence"
        else:
            af3_val["skipped"] = True

        results["steps"]["af3_validation"] = af3_val

        # ── Steps 12–15: ADC construction ─────────────────────────────────
        adc_design = {}
        best_design = None
        for d in af3_val.get("validated_designs", []):
            if d.get("af3_dir") and d.get("passed"):
                best_design = d
                break
        if not best_design:
            vd = af3_val.get("validated_designs", [])
            best_design = vd[0] if vd else None

        if not best_design or not best_design.get("af3_dir"):
            adc_design = {"error": "No AF3 structure for ADC steps", "partial": True}
        else:
            af3_dir = best_design["af3_dir"]
            # Resolve a PDB file from AF3 output for FreeSASA / ADC steps
            af3_pdb = None

            # Strategy 1: Convert *_model.cif → PDB (preferred, has all atoms)
            if os.path.isdir(af3_dir):
                cif_files = sorted(
                    [f for f in os.listdir(af3_dir) if f.endswith(".cif")],
                    key=lambda f: ("_model" not in f, f),  # prefer *_model.cif
                )
                if cif_files:
                    af3_cif = os.path.join(af3_dir, cif_files[0])
                    af3_pdb = af3_cif.replace(".cif", "_for_sasa.pdb")
                    # Skip re-conversion if output already exists and is non-empty
                    if os.path.exists(af3_pdb) and os.path.getsize(af3_pdb) > 100:
                        logger.info("[pocket_guided] Reusing existing PDB: %s", af3_pdb)
                    else:
                        try:
                            convert_af3_cif_to_pdb(af3_cif, af3_pdb, protein_only=False)
                        except Exception as e:
                            logger.warning("[pocket_guided] CIF→PDB (pdbfixer) failed: %s", e)
                            # Fallback: bare gemmi conversion without pdbfixer
                            try:
                                import gemmi
                                st = gemmi.read_structure(af3_cif)
                                st.write_pdb(af3_pdb)
                                logger.info("[pocket_guided] CIF→PDB (gemmi-only fallback) → %s", af3_pdb)
                            except Exception as e2:
                                logger.warning("[pocket_guided] CIF→PDB gemmi fallback also failed: %s", e2)
                                af3_pdb = None

            # Strategy 2: Look for existing PDB in AF3 dir or seed subdirs
            if not af3_pdb and os.path.isdir(af3_dir):
                for candidate in os.listdir(af3_dir):
                    full = os.path.join(af3_dir, candidate)
                    if candidate.endswith(".pdb") and os.path.getsize(full) > 100:
                        af3_pdb = full
                        logger.info("[pocket_guided] Found existing PDB in AF3 dir: %s", af3_pdb)
                        break
                # Check seed subdirectories
                if not af3_pdb:
                    for subdir in sorted(os.listdir(af3_dir)):
                        subpath = os.path.join(af3_dir, subdir)
                        if os.path.isdir(subpath):
                            for f in os.listdir(subpath):
                                if f.endswith(".pdb") and os.path.getsize(os.path.join(subpath, f)) > 100:
                                    af3_pdb = os.path.join(subpath, f)
                                    logger.info("[pocket_guided] Found PDB in AF3 seed dir: %s", af3_pdb)
                                    break
                            if af3_pdb:
                                break

            if af3_pdb:
                logger.info("[pocket_guided] AF3 PDB for ADC: %s (%.1f KB)",
                            af3_pdb, os.path.getsize(af3_pdb) / 1024)
            else:
                logger.warning("[pocket_guided] No PDB could be resolved from AF3 dir: %s", af3_dir)

            # Step 12: FreeSASA conjugation sites
            site_residue = None
            site_type = ""
            try:
                task.progress = 75
                task.progress_msg = "Step 12/15: FreeSASA conjugation sites..."
                if not af3_pdb:
                    raise RuntimeError("No PDB for FreeSASA")
                from routers.protein_design import run_freesasa
                sasa_ref = await run_freesasa(FreeSASARequest(
                    job_name=f"{req.job_name}_freesasa", input_pdb=af3_pdb,
                ))
                sasa_result = await _wait_for_task(sasa_ref.task_id, timeout=300)
                results["steps"]["freesasa"] = sasa_result
                # Filter: binder chain only + SASA > 40 Å² (ADC payload on binder, not antigen)
                _bc = _binder_chain2  # detected at MPNN step
                filtered = [s for s in sasa_result.get("conjugation_sites", [])
                            if float(s.get("sasa", 0)) > 40.0 and s.get("chain", "") == _bc]
                filtered.sort(key=lambda s: float(s.get("sasa", 0)), reverse=True)
                top_sites = filtered[:3]
                adc_design["conjugation_sites"] = top_sites
                if top_sites:
                    site_residue = top_sites[0]["residue"]
                    site_type = top_sites[0].get("type", "")
                else:
                    raise RuntimeError("No sites with SASA > 40 Å²")
            except Exception as e:
                logger.warning("[pocket_guided] Step 11 failed: %s", e)
                adc_design.setdefault("error", f"freesasa: {e}")
                adc_design["partial"] = True

            # Step 13: Linker Select
            linker_name = linker_smiles = None
            try:
                task.progress = 82
                task.progress_msg = "Step 13/15: Linker selection..."
                chem = "maleimide_thiol" if site_type and "Cys" in site_type else "nhs_amine"
                from routers.adc import linker_select
                linker_result = await linker_select(LinkerSelectRequest(
                    cleavable=True, reaction_type=chem,
                    compatible_payload="MMAE", max_results=3,
                ))
                results["steps"]["linker_select"] = linker_result
                rec = linker_result.get("recommended_linkers", [])
                if rec:
                    linker_name = rec[0].get("name", rec[0].get("linker_name"))
                    linker_smiles = rec[0].get("smiles", rec[0].get("linker_smiles"))
                    adc_design["linker"] = linker_name
                    adc_design["linker_smiles"] = linker_smiles
                    adc_design["reaction_type"] = chem
            except Exception as e:
                logger.warning("[pocket_guided] Step 12 failed: %s", e)
                if "error" not in adc_design:
                    adc_design["error"] = f"linker: {e}"
                    adc_design["partial"] = True

            # Step 14: Fetch MMAE
            payload_smiles = None
            try:
                task.progress = 88
                task.progress_msg = "Step 14/15: Fetching MMAE payload..."
                from routers.structure_prediction import fetch_molecule, FetchMoleculeRequest
                mmae = await fetch_molecule(FetchMoleculeRequest(query="MMAE"))
                results["steps"]["fetch_payload"] = mmae
                payload_smiles = mmae.get("smiles")
                adc_design["payload"] = "MMAE"
                adc_design["payload_smiles"] = payload_smiles
            except Exception as e:
                logger.warning("[pocket_guided] Step 13 failed: %s", e)
                if "error" not in adc_design:
                    adc_design["error"] = f"payload: {e}"
                    adc_design["partial"] = True

            # Step 15: RDKit Conjugation
            try:
                task.progress = 94
                task.progress_msg = "Step 15/15: RDKit ADC conjugation (DAR=4)..."
                if not all([site_residue, linker_smiles, payload_smiles, af3_pdb]):
                    missing = [k for k, v in {"site": site_residue, "linker": linker_smiles,
                                               "payload": payload_smiles, "pdb": af3_pdb}.items() if not v]
                    raise RuntimeError(f"Missing: {missing}")
                from routers.adc import rdkit_conjugate
                conj_ref = await rdkit_conjugate(RDKitConjugateRequest(
                    job_name=f"{req.job_name}_conjugate",
                    antibody_pdb=af3_pdb, conjugation_site=site_residue,
                    linker_smiles=linker_smiles, payload_smiles=payload_smiles,
                    linker_name=linker_name or "", reaction_type="auto",
                ))
                conj_result = await _wait_for_task(conj_ref.task_id, timeout=300)
                results["steps"]["rdkit_conjugate"] = conj_result
                adc_design["adc_smiles"] = conj_result.get("adc_smiles")
                adc_design["dar"] = 4
                adc_design["adc_structure_path"] = conj_result.get("output_sdf")
                adc_design["reaction_type_used"] = conj_result.get("reaction_type_used")
            except Exception as e:
                logger.warning("[pocket_guided] Step 11 failed: %s", e)
                if "error" not in adc_design:
                    adc_design["error"] = f"conjugate: {e}"
                    adc_design["partial"] = True

            adc_design["nanobody_sequence"] = best_design.get("sequence")
            adc_design["iptm"] = best_design.get("iptm")
            if site_residue:
                adc_design["conjugation_site"] = site_residue

        results["adc_design"] = adc_design

        task.progress = 100
        sel = results.get("selected_pocket", {})
        epi_added = results.get("pocket_analysis", {}).get("epitope_residues_added", 0)
        task.progress_msg = (
            f"Pocket-guided binder + ADC complete (6D scoring). "
            f"Selected pocket {sel.get('id')} (composite={(sel.get('composite_score') or 0):.3f}), "
            f"hotspots: {hotspot_residues}"
            f"{f' (+{epi_added} epitope-enriched)' if epi_added else ''}, "
            f"{len(backbone_pdbs)} backbones, "
            f"{af3_val.get('passed', 0)}/{af3_val.get('total_tested', 0)} AF3-validated."
        )
        return results

    task = await task_manager.submit("pocket_guided_binder_pipeline", req.model_dump(), _run)
    return TaskRef(task_id=task.task_id, status=task.status,
                   tool="pocket_guided_binder_pipeline",
                   poll_url=f"/api/v1/tasks/{task.task_id}")
