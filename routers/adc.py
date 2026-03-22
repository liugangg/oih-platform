"""
ADC (Antibody-Drug Conjugate) Router
Tools: linker_select, rdkit_conjugate

Supports 7 conjugation chemistries:
  maleimide_thiol, nhs_amine, hydrazone, oxime, disulfide, dbco_azide, transglutaminase
"""
# --- SYNC_NOTES (auto-generated from CLAUDE.md, do not edit) ---
# ADC 注意事项（来自 CLAUDE.md，勿手动编辑）：
#   - Step 3: AF3 验证 — top5 MPNN → AF3 复合物 → ipTM 分级（≥0.75 high / ≥0.6 low_confidence）
#   - 每步 try/except，单步失败标记 `partial: true` 不阻塞后续
#   - AF3 任务间隔 5 秒避免 GPU OOM，timeout 1800s
#   - Step 1-3: RFdiffusion → ProteinMPNN → AF3 验证（ipTM ≥0.75 high / ≥0.6 low_confidence）
#   - 每步 try/except，单步失败标记 partial，不阻塞后续
# --- /SYNC_NOTES ---

import os
import logging
from fastapi import APIRouter, HTTPException
from schemas.models import (
    LinkerSelectRequest, LinkerSelectResult,
    RDKitConjugateRequest, RDKitConjugateResult,
    TaskRef,
)
from core.task_manager import task_manager
from core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1: REACTION_REGISTRY — 7 mainstream ADC conjugation chemistries
# ═══════════════════════════════════════════════════════════════════════════════

REACTION_REGISTRY = {

    # ── Type 1: Maleimide-thiol (>60% clinical ADCs) ─────────────────────────
    "maleimide_thiol": {
        "smarts_chain": [
            "[C:1]1=[C:2]C(=O)[NH1]C1=O.[SH1:3]>>[C:1]1([S:3])[C:2]C(=O)[NH1]C1=O",
            "[C:1]=[C:2]C(=O)[N:4].[SH1:3]>>[C:1]([S:3])[C:2]C(=O)[N:4]",
            "[C:1]=[C:2]C(=O).[SH1:3]>>[C:1]([S:3])[C:2]C(=O)",
        ],
        "chemistry_cn": "马来酰亚胺-巯基偶联 (Michael addition)",
        "linker_end": "maleimide",
        "payload_end": "thiol/cysteine",
        "examples": ["SMCC", "maleimide-PEG4", "MC-VC-PABC"],
        "dar_range": "2-8",
        "stability": "pH sensitive, may retro-Michael in plasma",
        "note": "Kadcyla (T-DM1), Adcetris (brentuximab vedotin)",
    },

    # ── Type 2: NHS ester / acid-amine (lysine conjugation) ──────────────────
    "nhs_amine": {
        "smarts_chain": [
            # NHS ester + primary amine → amide
            "[C:1](=O)ON1C(=O)CCC1=O.[NH2:2]>>[C:1](=O)[NH1:2]",
            # Carboxylic acid + primary amine → amide
            "[C:1](=O)[OH1].[NH2:2]>>[C:1](=O)[NH1:2]",
            # NHS ester + secondary amine (N-methyl, e.g. MMAE) → amide
            "[C:1](=O)ON1C(=O)CCC1=O.[NH1:2]>>[C:1](=O)[N:2]",
            # Carboxylic acid + secondary amine → amide
            "[C:1](=O)[OH1].[NH1:2]>>[C:1](=O)[N:2]",
            # Amide coupling with N(C) (tertiary-like in MMAE context)
            "[C:1](=O)ON1C(=O)CCC1=O.[NH0:2]([CH3])[#6:3]>>[C:1](=O)[N:2]([CH3])[#6:3]",
            # Acid + hydroxyl → ester (fallback)
            "[C:1](=O)[OH1].[OH1:2][#6:3]>>[C:1](=O)[O:2][#6:3]",
            "[C:1](=O)ON1C(=O)CCC1=O.[OH1:2]>>[C:1](=O)[O:2]",
        ],
        "chemistry_cn": "NHS酯/羧酸-胺/醇缩合 (酰胺键/酯键)",
        "linker_end": "NHS_ester / carboxylic_acid",
        "payload_end": "primary_amine / lysine / hydroxyl",
        "examples": ["VC-PABC", "SMCC", "SPDB"],
        "dar_range": "3-4",
        "stability": "stable amide/ester bond",
        "note": "Non-site-specific, high DAR heterogeneity",
    },

    # ── Type 3: Hydrazone (pH-responsive cleavable) ──────────────────────────
    "hydrazone": {
        "smarts_chain": [
            "[C:1](=[O:4]).[NH2:2][NH1:3]>>[C:1](=[N:2][NH1:3]).[O:4]",
            "[CH1:1]=O.[NH2:2][N:3]>>[CH1:1]=[N:2][N:3]",
            "[C:1](=O)[#6:4].[NH2:2][N:3]>>[C:1](=[N:2][N:3])[#6:4]",
        ],
        "chemistry_cn": "腙键形成 (pH响应性可裂解)",
        "linker_end": "hydrazide",
        "payload_end": "ketone / aldehyde",
        "examples": ["Hydrazone", "BMPH", "EMCH"],
        "dar_range": "4-8",
        "stability": "pH-cleavable (lysosome pH~5, stable at pH 7.4)",
        "note": "Mylotarg (gemtuzumab ozogamicin) early version",
    },

    # ── Type 4: Oxime (stable cleavable alternative) ─────────────────────────
    "oxime": {
        "smarts_chain": [
            "[CH1:1]=O.[NH2:2][OH1]>>[CH1:1]=[N:2]O",
            "[C:1](=[O:3]).[NH2:2]O>>[C:1](=[N:2]O).[O:3]",
        ],
        "chemistry_cn": "肟键形成 (稳定可裂解)",
        "linker_end": "hydroxylamine / aminooxy",
        "payload_end": "aldehyde / ketone",
        "examples": ["aminooxy-PEG"],
        "dar_range": "2-4",
        "stability": "more stable than hydrazone at physiological pH",
        "note": "Site-specific, homogeneous DAR",
    },

    # ── Type 5: Disulfide (reductive cleavable) ──────────────────────────────
    "disulfide": {
        "smarts_chain": [
            "[S:1][S:2].[SH1:3]>>[S:1][S:3]",
            "[SH1:1].[SH1:2]>>[S:1][S:2]",
        ],
        "chemistry_cn": "二硫键形成 (还原性可裂解)",
        "linker_end": "pyridyldithiol / activated disulfide",
        "payload_end": "thiol",
        "examples": ["SPDP", "DTNB", "CL2"],
        "dar_range": "2-4",
        "stability": "GSH-cleavable in cytoplasm (1-10 mM GSH)",
        "note": "Intracellular glutathione cleaves and releases payload",
    },

    # ── Type 6: Click chemistry DBCO-azide (bioorthogonal, site-specific) ────
    "dbco_azide": {
        "smarts_chain": [
            "[N-:1]=[N+:2]=[N:3].[C:4]#[C:5]>>[n:1]1[n:2]=[n:3][C:4]=[C:5]1",
        ],
        "chemistry_cn": "SPAAC点击化学 (DBCO-叠氮, 生物正交)",
        "linker_end": "DBCO / BCN (strained alkyne)",
        "payload_end": "azide",
        "examples": ["DBCO-PEG4", "BCN-NHS"],
        "dar_range": "2 (site-specific)",
        "stability": "highly stable triazole, bioorthogonal",
        "note": "Next-gen site-specific ADC, precise DAR=2",
    },

    # ── Type 7: Transglutaminase (enzymatic, site-specific) ──────────────────
    "transglutaminase": {
        "smarts_chain": [],
        "chemistry_cn": "转谷氨酰胺酶催化定点偶联 (酶促)",
        "linker_end": "amine-PEG / cadaverine",
        "payload_end": "glutamine_tag (Q295)",
        "examples": ["cadaverine-payload"],
        "dar_range": "2 (site-specific)",
        "stability": "extremely stable isopeptide bond",
        "note": "Requires engineered antibody with Q295 tag; no in-silico reaction",
    },
}

# Generic fallback reactions (tried when primary type fails)
_GENERIC_SMARTS = [
    ("amide",  "[C:1](=O)[OH1].[NH2:2]>>[C:1](=O)[NH1:2]"),
    ("amide_sec", "[C:1](=O)[OH1].[NH1:2]>>[C:1](=O)[N:2]"),
    ("nhs_amide", "[C:1](=O)ON1C(=O)CCC1=O.[NH1:2]>>[C:1](=O)[N:2]"),
    ("ester",  "[C:1](=O)[OH1].[OH1:2][#6:3]>>[C:1](=O)[O:2][#6:3]"),
    ("ester",  "[C:1](=O)Cl.[OH1:2]>>[C:1](=O)[O:2]"),
    ("anhydride_open", "[C:1](=O)[O:4][C:2](=O).[OH1:3]>>[C:1](=O)[O:3].[C:2](=O)[OH1]"),
    ("anhydride_open", "[C:1](=O)[O:4][C:2](=O).[NH2:3]>>[C:1](=O)[NH1:3].[C:2](=O)[OH1]"),
]

# ─── Linker name → reaction type mapping ─────────────────────────────────────

LINKER_REACTION_MAP = {
    # Maleimide-thiol
    "MC-VC-PABC":      "maleimide_thiol",
    "MC-VA-PABC":      "maleimide_thiol",
    "SMCC":            "maleimide_thiol",
    "MCC":             "maleimide_thiol",
    "MC":              "maleimide_thiol",
    "Maleimide-PEG4":  "maleimide_thiol",
    "maleimide-PEG4":  "maleimide_thiol",
    "GGFG":            "maleimide_thiol",
    "PEG2-VC-PABC":    "maleimide_thiol",
    "MD-Linker":       "maleimide_thiol",
    "cBu-Cit-PABC":    "maleimide_thiol",
    # NHS-amine
    "VC-PABC":         "nhs_amine",
    # Disulfide
    "SPDB":            "disulfide",
    "Sulfo-SPDB":      "disulfide",
    "SPDP":            "disulfide",
    "CL2":             "disulfide",
    "CL2A":            "disulfide",
    # Hydrazone
    "Hydrazone":       "hydrazone",
    "AcBut-Hydrazone": "hydrazone",
    "BMPH":            "hydrazone",
    "EMCH":            "hydrazone",
    # Click chemistry
    "DBCO-PEG4":       "dbco_azide",
    "DBCO-PEG4-NHS":   "dbco_azide",
    "BCN-PEG3-NHS":    "dbco_azide",
    "BCN-NHS":         "dbco_azide",
    # Oxime
    "Aminooxy-PEG4":   "oxime",
    # Transglutaminase
    "TGase-Cadaverine": "transglutaminase",
    # Thioether
    "Bromoacetamide":  "maleimide_thiol",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2: Auto-detect reaction type from functional groups
# ═══════════════════════════════════════════════════════════════════════════════

# SMARTS patterns for functional group detection on linker
_FG_PATTERNS = {
    "maleimide":     "[C]1=[C]C(=O)[N]C1=O",          # maleimide ring
    "maleimide_open": "C=CC(=O)N",                     # open-chain maleimide-like
    "nhs_ester":     "C(=O)ON1C(=O)CCC1=O",            # NHS ester
    "carboxylic":    "[CX3](=O)[OX2H1]",               # -COOH
    "hydrazide":     "[NH2][NH]C(=O)",                  # -NHNH2 or -CONHNH2
    "hydrazine":     "[NH2][NH]",                       # -NHNH2
    "hydroxylamine": "[NH2]O",                          # -ONH2
    "pyridyl_dithio": "SSc1ccccn1",                     # pyridyldithiol
    "disulfide":     "[#16][#16]",                      # S-S bond
    "thiol":         "[SH1]",                           # free thiol
    "azide":         "[N-]=[N+]=N",                     # azide
    "alkyne":        "C#C",                             # terminal alkyne
    "primary_amine": "[NH2;!$(NC=O)]",                  # free primary amine
    "aldehyde":      "[CH1]=O",                         # aldehyde
    "ketone":        "[#6][CX3](=O)[#6]",              # ketone
}


def detect_reaction_type(linker_smiles: str, payload_smiles: str,
                         linker_name: str | None = None) -> dict:
    """
    Determine the best reaction type for conjugation.

    Priority:
      1. linker_name in LINKER_REACTION_MAP → direct lookup
      2. Functional group detection on linker + payload matching
      3. Fallback → nhs_amine (most permissive)

    Returns: {reaction_type, confidence, reasoning, detection_method}
    """
    from rdkit import Chem

    # ── Priority 1: linker_name lookup ────────────────────────────────────────
    if linker_name and linker_name in LINKER_REACTION_MAP:
        rtype = LINKER_REACTION_MAP[linker_name]
        return {
            "reaction_type": rtype,
            "confidence": "high",
            "reasoning": f"Linker '{linker_name}' mapped to {rtype}",
            "detection_method": "linker_name",
        }

    # ── Priority 2: functional group detection ────────────────────────────────
    linker_mol = Chem.MolFromSmiles(linker_smiles)
    payload_mol = Chem.MolFromSmiles(payload_smiles)
    if linker_mol is None or payload_mol is None:
        return {
            "reaction_type": "nhs_amine",
            "confidence": "low",
            "reasoning": "Could not parse SMILES, defaulting to nhs_amine",
            "detection_method": "fallback",
        }

    def _has(mol, smarts_key):
        pat = Chem.MolFromSmarts(_FG_PATTERNS[smarts_key])
        return pat is not None and mol.HasSubstructMatch(pat)

    # Check linker functional groups and match with payload
    # Maleimide on linker + thiol on payload
    if (_has(linker_mol, "maleimide") or _has(linker_mol, "maleimide_open")) and _has(payload_mol, "thiol"):
        return {"reaction_type": "maleimide_thiol", "confidence": "high",
                "reasoning": "Linker has maleimide, payload has thiol (-SH)",
                "detection_method": "auto"}

    # Pyridyldithiol / disulfide on linker + thiol on payload
    if (_has(linker_mol, "pyridyl_dithio") or _has(linker_mol, "disulfide")) and _has(payload_mol, "thiol"):
        return {"reaction_type": "disulfide", "confidence": "high",
                "reasoning": "Linker has activated disulfide (S-S), payload has thiol",
                "detection_method": "auto"}

    # Hydrazide on linker + aldehyde/ketone on payload (or vice versa)
    if (_has(linker_mol, "hydrazide") or _has(linker_mol, "hydrazine")):
        if _has(payload_mol, "aldehyde") or _has(payload_mol, "ketone"):
            return {"reaction_type": "hydrazone", "confidence": "high",
                    "reasoning": "Linker has hydrazide, payload has ketone/aldehyde",
                    "detection_method": "auto"}
    if (_has(payload_mol, "hydrazide") or _has(payload_mol, "hydrazine")):
        if _has(linker_mol, "aldehyde") or _has(linker_mol, "ketone"):
            return {"reaction_type": "hydrazone", "confidence": "high",
                    "reasoning": "Payload has hydrazide, linker has ketone/aldehyde",
                    "detection_method": "auto"}

    # Hydroxylamine + aldehyde/ketone → oxime
    if _has(linker_mol, "hydroxylamine") and (_has(payload_mol, "aldehyde") or _has(payload_mol, "ketone")):
        return {"reaction_type": "oxime", "confidence": "high",
                "reasoning": "Linker has hydroxylamine, payload has aldehyde/ketone",
                "detection_method": "auto"}

    # Azide + alkyne → click chemistry
    if (_has(linker_mol, "azide") and _has(payload_mol, "alkyne")) or \
       (_has(linker_mol, "alkyne") and _has(payload_mol, "azide")):
        return {"reaction_type": "dbco_azide", "confidence": "medium",
                "reasoning": "Azide + alkyne detected → SPAAC click chemistry",
                "detection_method": "auto"}

    # Maleimide on linker (without thiol on payload) — still likely maleimide_thiol
    # (the Cys thiol comes from the antibody, not the payload)
    if _has(linker_mol, "maleimide") or _has(linker_mol, "maleimide_open"):
        return {"reaction_type": "maleimide_thiol", "confidence": "medium",
                "reasoning": "Linker has maleimide (thiol from antibody Cys, not payload)",
                "detection_method": "auto"}

    # NHS ester or carboxylic acid on linker + amine on payload
    if (_has(linker_mol, "nhs_ester") or _has(linker_mol, "carboxylic")) and _has(payload_mol, "primary_amine"):
        return {"reaction_type": "nhs_amine", "confidence": "high",
                "reasoning": "Linker has NHS/COOH, payload has primary amine",
                "detection_method": "auto"}

    # Carboxylic on linker, amine on payload (broader match)
    if _has(linker_mol, "carboxylic") or _has(linker_mol, "nhs_ester"):
        return {"reaction_type": "nhs_amine", "confidence": "medium",
                "reasoning": "Linker has carboxylic acid/NHS ester",
                "detection_method": "auto"}

    # Amine on payload + any electrophile
    if _has(payload_mol, "primary_amine"):
        return {"reaction_type": "nhs_amine", "confidence": "low",
                "reasoning": "Payload has amine, attempting amide bond formation",
                "detection_method": "auto"}

    # Default fallback
    return {
        "reaction_type": "nhs_amine",
        "confidence": "low",
        "reasoning": "No specific functional group pair detected, defaulting to amide coupling",
        "detection_method": "fallback",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Part 3: Execute conjugation reaction
# ═══════════════════════════════════════════════════════════════════════════════

def _try_smarts_list(smarts_list, linker_mol, payload_mol, label_prefix=""):
    """
    Try a list of SMARTS reactions in order, both forward and reverse.
    Returns (product_mol, smarts_used) or (None, None).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    for i, smarts in enumerate(smarts_list):
        try:
            rxn = AllChem.ReactionFromSmarts(smarts)
        except Exception:
            continue

        for reactants, direction in [
            ((linker_mol, payload_mol), "fwd"),
            ((payload_mol, linker_mol), "rev"),
        ]:
            try:
                products_sets = rxn.RunReactants(reactants)
            except Exception:
                continue

            for products in products_sets:
                for product in products:
                    try:
                        Chem.SanitizeMol(product)
                        smi = Chem.MolToSmiles(product)
                        if smi and len(smi) > 5:
                            return product, f"{label_prefix}smarts[{i}]_{direction}"
                    except Exception:
                        continue

    return None, None


def run_conjugation(linker_mol, payload_mol, reaction_type: str):
    """
    Attempt covalent conjugation using SMARTS from REACTION_REGISTRY.

    Strategy:
      1. Try all smarts_chain entries for the specified reaction_type
      2. If all fail, try _GENERIC_SMARTS (amide, ester, anhydride opening)
      3. If still no product, return None (caller does dot-disconnected fallback)

    Returns: (product_mol | None, covalent: bool, actual_bond_type: str, warnings: list[str])
    """
    entry = REACTION_REGISTRY.get(reaction_type)
    if not entry:
        return None, False, "none", [f"Unknown reaction_type: {reaction_type}"]

    warnings = []
    smarts_chain = entry.get("smarts_chain", [])

    if not smarts_chain:
        warnings.append(f"{reaction_type}: no SMARTS available (enzymatic/non-chemical)")
        return None, False, "none", warnings

    # ── Step 1: Try primary reaction type's SMARTS chain ─────────────────────
    product, used = _try_smarts_list(smarts_chain, linker_mol, payload_mol, f"{reaction_type}_")
    if product:
        return product, True, reaction_type, warnings

    warnings.append(f"Primary {reaction_type} SMARTS chain ({len(smarts_chain)} entries) produced no valid product")

    # ── Step 2: Try generic fallback reactions ────────────────────────────────
    generic_smarts = [s for _, s in _GENERIC_SMARTS]
    product, used = _try_smarts_list(generic_smarts, linker_mol, payload_mol, "generic_")
    if product:
        # Identify which generic reaction worked
        generic_idx = int(used.split("[")[1].split("]")[0])
        bond_name = _GENERIC_SMARTS[generic_idx][0]
        warnings.append(f"Used generic fallback: {bond_name} bond formation ({used})")
        return product, True, f"{reaction_type}(generic_{bond_name})", warnings

    warnings.append("Generic fallback SMARTS also failed")
    return None, False, "none", warnings


# ═══════════════════════════════════════════════════════════════════════════════
# Linker Library (loaded from data/linker_library.json)
# ═══════════════════════════════════════════════════════════════════════════════

import json

_LINKER_LIBRARY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "linker_library.json")

def _load_linker_library() -> list:
    with open(_LINKER_LIBRARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["linkers"]


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/linker_select", response_model=LinkerSelectResult,
             summary="Search clinical ADC linker library (20 entries)")
async def linker_select(req: LinkerSelectRequest):
    """
    Filter the 20-entry clinical linker library by cleavable, reaction_type,
    compatible_payload, and clinical_status. Returns top matches.
    """
    linkers = _load_linker_library()

    # ── Resolve legacy linker_type → cleavable bool ───────────────────────────
    cleavable = req.cleavable
    if cleavable is None and req.linker_type is not None:
        cleavable = (req.linker_type == "cleavable")

    # ── Apply filters ─────────────────────────────────────────────────────────
    results = []
    for entry in linkers:
        if cleavable is not None and entry.get("cleavable") != cleavable:
            continue
        if req.reaction_type and entry.get("reaction_type") != req.reaction_type:
            continue
        if req.clinical_status and entry.get("clinical_status") != req.clinical_status:
            continue
        if req.compatible_payload:
            query = req.compatible_payload.lower()
            payloads = [p.lower() for p in entry.get("compatible_payloads", [])]
            if not any(query in p or p in query for p in payloads):
                continue
        results.append(entry)

    # ── Sort: approved > clinical > research (when no explicit filter) ────────
    _STATUS_ORDER = {"approved": 0, "clinical": 1, "research": 2}
    if not req.clinical_status:
        results.sort(key=lambda e: _STATUS_ORDER.get(e.get("clinical_status", "research"), 9))

    total = len(results)
    results = results[: req.max_results]

    # ── Build recommendation reason for top result ────────────────────────────
    recommendation = ""
    if results:
        top = results[0]
        parts = []
        adcs = top.get("approved_adcs", [])
        if adcs:
            parts.append(f"已获批ADC使用: {', '.join(adcs[:2])}")
        parts.append(f"反应类型: {top['reaction_type']}, DAR: {top['dar_range']}")
        if top.get("stability_plasma") in ("high", "very_high"):
            parts.append(f"血浆稳定性: {top['stability_plasma']}")
        recommendation = f"推荐 {top['name']} — {'；'.join(parts)}"

    return {
        "recommended_linkers": [
            {
                "id": r["id"],
                "name": r["name"],
                "smiles": r["smiles"],
                "reaction_type": r["reaction_type"],
                "dar_range": r["dar_range"],
                "approved_adcs": r.get("approved_adcs", []),
                "stability_plasma": r.get("stability_plasma", ""),
                "cleavage_mechanism": r.get("cleavage_mechanism"),
                "notes": r.get("notes", ""),
            }
            for r in results
        ],
        "total_matched": total,
        "recommendation": recommendation,
    }


@router.post("/rdkit_conjugate", response_model=TaskRef,
             summary="Build ADC payload-linker conjugate with RDKit (7 reaction types)")
async def rdkit_conjugate(req: RDKitConjugateRequest):
    """
    Build ADC payload-linker conjugate via covalent reaction.
    Supports 7 reaction chemistries; auto-detects from functional groups when
    reaction_type='auto'.
    """
    output_dir = os.path.join(settings.OUTPUT_DIR, req.job_name, "adc")
    os.makedirs(output_dir, exist_ok=True)

    async def _run(task):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        task.progress = 5
        task.progress_msg = "Parsing molecules..."

        linker_mol = Chem.MolFromSmiles(req.linker_smiles)
        if linker_mol is None:
            raise RuntimeError(f"Invalid linker SMILES: {req.linker_smiles}")

        payload_mol = Chem.MolFromSmiles(req.payload_smiles)
        if payload_mol is None:
            raise RuntimeError(f"Invalid payload SMILES: {req.payload_smiles}")

        # ── Step 1: Determine reaction type ───────────────────────────────────
        task.progress = 15
        task.progress_msg = "Detecting reaction type..."

        if req.reaction_type and req.reaction_type != "auto":
            detection = {
                "reaction_type": req.reaction_type,
                "confidence": "manual",
                "reasoning": f"User specified reaction_type={req.reaction_type}",
                "detection_method": "manual",
            }
        else:
            detection = detect_reaction_type(
                req.linker_smiles, req.payload_smiles, req.linker_name
            )

        rtype = detection["reaction_type"]
        entry = REACTION_REGISTRY.get(rtype, REACTION_REGISTRY["nhs_amine"])

        task.progress = 30
        task.progress_msg = f"Running {rtype} conjugation..."

        # ── Step 2: Attempt covalent conjugation ──────────────────────────────
        product_mol, covalent, bond_type, warnings = run_conjugation(linker_mol, payload_mol, rtype)

        if product_mol and covalent:
            canonical = Chem.MolToSmiles(product_mol)
            final_mol = product_mol
        else:
            # Fallback: dot-disconnected complex
            combined = Chem.MolFromSmiles(f"{req.linker_smiles}.{req.payload_smiles}")
            if combined is None:
                raise RuntimeError("Failed to build even dot-disconnected molecule")
            canonical = Chem.MolToSmiles(combined)
            final_mol = combined
            covalent = False
            warnings.append("Covalent reaction failed; using dot-disconnected (non-covalent) representation")

        # ── Step 3: Generate 3D and write SDF ─────────────────────────────────
        task.progress = 60
        task.progress_msg = "Generating 3D coordinates..."

        mol_3d = Chem.AddHs(final_mol)
        embedding_status = "3d"
        try:
            params = AllChem.ETKDGv3()
            embed_result = AllChem.EmbedMolecule(mol_3d, params)
            if embed_result != 0:
                # Retry with random coordinates (for large/complex molecules)
                params.useRandomCoords = True
                embed_result = AllChem.EmbedMolecule(mol_3d, params)
            if embed_result == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=500)
                except Exception:
                    warnings.append("MMFF optimization failed; using unoptimized 3D coords")
            else:
                embedding_status = "2d_only"
                warnings.append("3D embedding failed (large ADC molecule); returning 2D SMILES only")
        except Exception as embed_err:
            embedding_status = "2d_only"
            warnings.append(f"3D embedding error: {embed_err}; returning 2D SMILES only")

        mol_3d = Chem.RemoveHs(mol_3d)
        atom_count = mol_3d.GetNumAtoms()

        sdf_path = os.path.join(output_dir, "payload_linker.sdf")
        writer = Chem.SDWriter(sdf_path)
        mol_3d.SetProp("conjugation_site", req.conjugation_site)
        mol_3d.SetProp("antibody_pdb", req.antibody_pdb)
        mol_3d.SetProp("reaction_type", rtype)
        mol_3d.SetProp("covalent", str(covalent))
        writer.write(mol_3d)
        writer.close()

        task.progress = 100
        cov_str = "covalent" if covalent else "non-covalent"
        task.progress_msg = (
            f"Done ({cov_str}, {rtype}). "
            f"SMILES: {canonical[:60]}... | {atom_count} atoms"
        )

        return {
            "adc_smiles": canonical,
            "covalent": covalent,
            "reaction_type_used": rtype,
            "reaction_chemistry": entry["chemistry_cn"],
            "detection_method": detection["detection_method"],
            "dar_range": entry["dar_range"],
            "stability_note": entry["stability"],
            "linker_info": {
                "linker_end": entry["linker_end"],
                "payload_end": entry["payload_end"],
                "examples": entry["examples"],
                "note": entry["note"],
            },
            "output_sdf": sdf_path,
            "embedding_status": embedding_status,
            "atom_count": atom_count,
            "warnings": warnings,
        }

    task = await task_manager.submit("rdkit_conjugate", req.model_dump(), _run)
    return TaskRef(
        task_id=task.task_id, status=task.status, tool="rdkit_conjugate",
        poll_url=f"/api/v1/tasks/{task.task_id}",
    )
