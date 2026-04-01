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
    "domain_knowledge": ["什么是", "原理", "解释", "科普", "知识", "概念", "what is", "explain",
                          "how does", "mechanism", "principle", "ipTM", "pLDDT", "RMSD"],
    "matplotlib_figures": ["matplotlib", "图表", "plot", "figure", "chart", "画图", "柱状图",
                           "散点图", "热图", "heatmap", "可视化", "visualization"],
    "pymol_rendering": ["pymol", "渲染", "render", "结构图", "蛋白图", "cartoon", "surface",
                        "structure image", "结构可视化"],
    "rag": ["文献", "论文", "literature", "pubmed", "paper", "检索", "搜索文献", "search paper",
            "bioRxiv", "知识库", "knowledge base"],
    "paper_metrics": ["metrics", "paper data", "tsv", "论文数据", "指标提取", "binder_len", "antigen_len",
                      "ipSAE", "pDockQ", "extract metrics", "验证结果", "all_designs"],
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
        "triggers": ["binder", "nanobody", "antibody", "抗体", "纳米抗体",
                      "结合蛋白", "adc", "ADC", "偶联", "drug conjugate",
                      "设计.*binder", "design.*binder"],
        "skills": ["binder_design", "tool_scope"],
    },
    # drug discovery → binder_design has the full pipeline; add fpocket for pocket analysis
    "drug_discovery": {
        "triggers": ["drug discovery", "药物发现", "靶点.*设计", "pipeline",
                      "端到端", "end.to.end"],
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
        sections.append(f"## [{name.upper()} 操作规程]\n\n{content}")
        total_chars += len(content)
        injected.append(name)

    if sections:
        skill_block = "\n\n---\n\n".join(sections)
        augmented = (
            base_prompt
            + "\n\n" + "=" * 60
            + "\n# 当前任务操作规程（必须严格遵循）\n"
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
