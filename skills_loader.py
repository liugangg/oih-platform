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
}

KEYWORD_SKILL_MAP = {
    "autodock":    ["autodock", "dock", "docking", "vina", "autogrid", "gpf", "pdbqt", "ligand", "smiles", "对接", "分子对接", "配体"],
    "gromacs":     ["gromacs", "gmx", "molecular dynamics", "md simulation", "trajectory", "nvt", "npt", "分子动力学", "模拟"],
    "diffdock":    ["diffdock", "blind dock", "blind docking", "盲对接"],
    "alphafold3":  ["alphafold", "af3", "structure predict", "fold", "折叠", "结构预测"],
    "rfdiffusion": ["rfdiffusion", "diffusion", "protein design", "scaffold", "蛋白设计", "antibody", "抗体", "nanobody", "纳米抗体", "binder design", "binder"],
    "proteinmpnn": ["proteinmpnn", "mpnn", "sequence design", "inverse folding", "序列设计"],
    "bindcraft":   ["bindcraft", "binder design", "binder", "结合蛋白设计"],
    "esm":         ["esm", "embedding", "protein language", "蛋白语言模型", "esm2", "protein embedding"],
    "chemprop":    ["chemprop", "admet", "property predict", "molecular property", "分子性质", "toxicity", "solubility", "bioactivity", "qsar"],
    "fpocket":     ["fpocket", "p2rank", "pocket", "binding site", "cavity", "druggable", "口袋", "结合位点"],
    "vina":        ["vina", "vina-gpu", "vina_gpu", "autodockvina", "快速对接"],
    "gnina":       ["gnina", "cnn score", "cnn docking", "深度学习对接"],
    "overview":    ["what tools", "available tools", "what can you do", "capabilities", "help", "能做什么", "有哪些工具", "平台介绍"],
    "ptm_upload":  ["上传", "upload", "pdb文件", "fasta", "smiles", "糖基化", "磷酸化", "二硫键", "ptm", "修饰", "modification"],
    "adc":         ["adc", "ADC", "antibody drug conjugate", "抗体偶联", "偶联药物", "linker", "payload", "MMAE", "DM1", "maleimide", "conjugate", "DAR", "药抗比"],
    "self_diagnosis": ["失败", "错误", "error", "failed", "oom", "超时", "timeout", "修复", "诊断", "重试", "retry", "crash", "崩溃", "debug"],
    "discotope3":  ["discotope", "epitope", "表位", "b-cell", "b cell", "抗原表位", "免疫原性", "immunogenicity", "epitope prediction", "epitope mapping", "表位预测", "表位映射"],
    "igfold":      ["igfold", "antibody structure", "nanobody fold", "抗体结构预测", "纳米抗体折叠", "antibody folding"],
    "tool_scope":  ["适用范围", "tool scope", "when to use", "禁止", "误用", "binder_type", "tier"],
    "binder_design": ["binder design", "hotspot", "binding site", "where to bind", "target analysis",
                       "interface", "de novo binder", "pocket scoring", "tier classification",
                       "domain truncation", "6D", "热点", "结合位点", "靶点分析", "域截取"],
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
    # binder/ADC design → needs ALL of these to make correct decisions
    "binder_design": {
        "triggers": ["binder", "nanobody", "antibody", "抗体", "纳米抗体",
                      "结合蛋白", "adc", "ADC", "偶联", "drug conjugate",
                      "设计.*binder", "design.*binder"],
        "skills": ["binder_design", "tool_scope", "alphafold3", "rfdiffusion", "proteinmpnn",
                   "discotope3", "igfold", "adc", "bindcraft"],
    },
    # drug discovery → full stack
    "drug_discovery": {
        "triggers": ["drug discovery", "药物发现", "靶点.*设计", "pipeline",
                      "端到端", "end.to.end"],
        "skills": ["tool_scope", "alphafold3", "rfdiffusion", "proteinmpnn",
                   "discotope3", "adc", "fpocket", "esm"],
    },
}


def build_dynamic_system_prompt(base_prompt: str, user_message: str) -> Tuple[str, List[str]]:
    detected = detect_skills(user_message)

    # Pipeline-aware injection: if a pipeline task is detected,
    # add all related skills that Qwen needs for correct decision-making
    msg_lower = user_message.lower()
    for bundle_name, bundle in PIPELINE_SKILL_BUNDLES.items():
        import re as _re
        if any(_re.search(t, msg_lower) for t in bundle["triggers"]):
            for skill in bundle["skills"]:
                if skill not in detected:
                    detected.append(skill)

    sections = []
    for name in detected:
        content = load_skill(name)
        if content:
            sections.append(f"## [{name.upper()} 操作规程]\n\n{content}")

    if sections:
        skill_block = "\n\n---\n\n".join(sections)
        augmented = (
            base_prompt
            + "\n\n" + "=" * 60
            + "\n# 当前任务操作规程（必须严格遵循）\n"
            + "=" * 60
            + "\n\n" + skill_block
        )
        return augmented, detected

    return base_prompt, detected
