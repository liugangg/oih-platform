import os
from typing import List, Tuple

SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")

SKILL_MAP = {
    "autodock":    "AUTODOCK_GPU_WORKFLOW.md",
    "gromacs":     "GROMACS_WORKFLOW.md",
    "diffdock":    "DIFFDOCK_WORKFLOW.md",
    "alphafold3":  "ALPHAFOLD3_WORKFLOW.md",
    "rfdiffusion": "RFDIFFUSION_WORKFLOW.md",
    "proteinmpnn": "PROTEINMPNN_WORKFLOW.md",
    "bindcraft":   "BINDCRAFT_WORKFLOW.md",
    "chemprop":    "CHEMPROP_WORKFLOW.md",
    "esm":         "ESM_WORKFLOW.md",
    "fpocket":     "FPOCKET_P2RANK_WORKFLOW.md",
    "vina":        "VINA_GPU_GNINA_WORKFLOW.md",
    "gnina":       "VINA_GPU_GNINA_WORKFLOW.md",
    "overview":    "OIH_PLATFORM_OVERVIEW.md",
    "ptm_upload":  "PTM_UPLOAD_PARADIGM.md",
    "adc":         "ADC_WORKFLOW.md",
    "self_diagnosis": "SELF_DIAGNOSIS_WORKFLOW.md",
    "discotope3":  "DISCOTOPE3_WORKFLOW.md",
    "igfold":      "IGFOLD_WORKFLOW.md",
    "tool_scope":  "TOOL_SCOPE_RULES.md",
    "binder_design": "BINDER_DESIGN_WORKFLOW.md",
    "conversation_rules": "CONVERSATION_RULES.md",
    "customer_service": "CUSTOMER_SERVICE.md",
    "domain_knowledge": "DOMAIN_KNOWLEDGE.md",
    "matplotlib_figures": "MATPLOTLIB_FIGURES.md",
    "pymol_rendering": "PYMOL_RENDERING.md",
    "rag": "RAG_WORKFLOW.md",
    "paper_metrics": "PAPER_METRICS_WORKFLOW.md",
}

KEYWORD_SKILL_MAP = {
    "autodock":    ["autodock", "dock", "docking", "vina", "autogrid", "gpf", "pdbqt", "ligand", "smiles", "molecular docking"],
    "gromacs":     ["gromacs", "gmx", "molecular dynamics", "md simulation", "trajectory", "nvt", "npt", "simulation"],
    "diffdock":    ["diffdock", "blind dock", "blind docking"],
    "alphafold3":  ["alphafold", "af3", "structure predict", "fold", "folding", "structure prediction"],
    "rfdiffusion": ["rfdiffusion", "diffusion", "protein design", "scaffold", "antibody", "nanobody", "binder design", "binder"],
    "proteinmpnn": ["proteinmpnn", "mpnn", "sequence design", "inverse folding"],
    "bindcraft":   ["bindcraft", "binder design", "binder", "binding protein design"],
    "esm":         ["esm", "embedding", "protein language", "protein language model", "esm2", "protein embedding"],
    "chemprop":    ["chemprop", "admet", "property predict", "molecular property", "toxicity", "solubility", "bioactivity", "qsar"],
    "fpocket":     ["fpocket", "p2rank", "pocket", "binding site", "cavity", "druggable"],
    "vina":        ["vina", "vina-gpu", "vina_gpu", "autodockvina", "fast docking"],
    "gnina":       ["gnina", "cnn score", "cnn docking", "deep learning docking"],
    "overview":    ["what tools", "available tools", "what can you do", "capabilities", "help", "platform overview"],
    "ptm_upload":  ["upload", "pdb file", "fasta", "smiles", "glycosylation", "phosphorylation", "disulfide bond", "ptm", "modification"],
    "adc":         ["adc", "ADC", "antibody drug conjugate", "antibody conjugate", "conjugate drug", "linker", "payload", "MMAE", "DM1", "maleimide", "conjugate", "DAR", "drug-antibody ratio"],
    "self_diagnosis": ["failure", "error", "failed", "oom", "timeout", "fix", "diagnose", "retry", "crash", "debug"],
    "discotope3":  ["discotope", "epitope", "b-cell", "b cell", "antigenic epitope", "immunogenicity", "epitope prediction", "epitope mapping"],
    "igfold":      ["igfold", "antibody structure", "nanobody fold", "antibody structure prediction", "nanobody folding", "antibody folding"],
    "tool_scope":  ["tool scope", "when to use", "prohibited", "misuse", "binder_type", "tier"],
    "binder_design": ["binder design", "hotspot", "binding site", "where to bind", "target analysis",
                       "interface", "de novo binder", "pocket scoring", "tier classification",
                       "domain truncation", "6D"],
    "domain_knowledge": ["what is", "principle", "explain", "knowledge", "concept", "how does",
                          "mechanism", "ipTM", "pLDDT", "RMSD"],
    "matplotlib_figures": ["matplotlib", "chart", "plot", "figure", "draw chart", "bar chart",
                           "scatter plot", "heatmap", "visualization"],
    "pymol_rendering": ["pymol", "render", "rendering", "structure image", "protein image", "cartoon", "surface",
                        "structure visualization"],
    "rag": ["literature", "pubmed", "paper", "search literature", "search paper",
            "bioRxiv", "knowledge base"],
    "paper_metrics": ["metrics", "paper data", "tsv", "metric extraction", "binder_len", "antigen_len",
                      "ipSAE", "pDockQ", "extract metrics", "validation results", "all_designs"],
}

def load_skill(skill_name: str) -> str:
    filename = SKILL_MAP.get(skill_name)
    if not filename:
        return ""
    path = os.path.join(SKILLS_DIR, filename)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def detect_skills(user_message: str) -> List[str]:
    msg = user_message.lower()
    return [skill for skill, keywords in KEYWORD_SKILL_MAP.items()
            if any(kw in msg for kw in keywords)]

# When a pipeline-level task is detected, auto-inject all related skills
# so Qwen has complete decision context (not just keyword-matched subset)
PIPELINE_SKILL_BUNDLES = {
    # binder/ADC design → binder_design workflow already covers RFdiffusion/MPNN/AF3/ADC rules
    # Only add tool_scope (small, complementary) — no need to duplicate individual tool workflows
    "binder_design": {
        "triggers": ["binder", "nanobody", "antibody", "binding protein",
                      "adc", "ADC", "conjugate", "drug conjugate",
                      "design.*binder"],
        "skills": ["binder_design", "tool_scope"],
    },
    # drug discovery → binder_design has the full pipeline; add fpocket for pocket analysis
    "drug_discovery": {
        "triggers": ["drug discovery", "target.*design", "pipeline",
                      "end.to.end"],
        "skills": ["binder_design", "tool_scope", "fpocket"],
    },
}


def build_dynamic_system_prompt(base_prompt: str, user_message: str) -> Tuple[str, List[str]]:
    detected = detect_skills(user_message)

    # Pipeline-aware injection: if a pipeline task is detected,
    # add all related skills that Qwen needs for correct decision-making
    # Pipeline-aware injection: if a pipeline task is detected,
    # replace individually-matched tool skills with the consolidated workflow.
    # e.g. binder_design workflow already covers rfdiffusion/proteinmpnn/af3/adc rules,
    # so injecting those individually is wasteful token duplication.
    msg_lower = user_message.lower()
    _COVERED_BY_BUNDLE = {
        "binder_design": {"rfdiffusion", "proteinmpnn", "alphafold3", "adc",
                          "bindcraft", "discotope3", "igfold"},
        "drug_discovery": {"rfdiffusion", "proteinmpnn", "alphafold3", "adc",
                           "discotope3", "esm"},
    }
    for bundle_name, bundle in PIPELINE_SKILL_BUNDLES.items():
        import re as _re
        if any(_re.search(t, msg_lower) for t in bundle["triggers"]):
            # Remove individual skills that the bundle's workflow already covers
            covered = _COVERED_BY_BUNDLE.get(bundle_name, set())
            detected = [s for s in detected if s not in covered]
            for skill in bundle["skills"]:
                if skill not in detected:
                    detected.append(skill)

    # Load skills sorted by size (smallest first = most skills fit), enforce char budget
    # vLLM context=32768 tokens; base_prompt~10K chars + tools~40K chars → ~20K chars left for skills
    MAX_SKILL_CHARS = 15000

    loaded = []
    for name in detected:
        content = load_skill(name)
        if content:
            loaded.append((name, content))

    # Sort by file size ascending — pack more small skills before hitting budget
    loaded.sort(key=lambda x: len(x[1]))

    sections = []
    total_chars = 0
    injected = []
    for name, content in loaded:
        if total_chars + len(content) > MAX_SKILL_CHARS:
            continue
        sections.append(f"## [{name.upper()} OPERATING PROCEDURE]\n\n{content}")
        total_chars += len(content)
        injected.append(name)

    if sections:
        skill_block = "\n\n---\n\n".join(sections)
        augmented = (
            base_prompt
            + "\n\n" + "=" * 60
            + "\n# CURRENT TASK OPERATING PROCEDURES (MUST FOLLOW STRICTLY)\n"
            + "=" * 60
            + "\n\n" + skill_block
        )
        if len(detected) > len(injected):
            skipped = [n for n in detected if n not in injected]
            import logging
            logging.getLogger("oih").info(
                "[SkillsLoader] Budget %d/%d chars, injected %d/%d skills, skipped: %s",
                total_chars, MAX_SKILL_CHARS, len(injected), len(detected), skipped)
        return augmented, detected

    return base_prompt, detected
