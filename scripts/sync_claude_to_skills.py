#!/usr/bin/env python3
"""
sync_claude_to_skills.py
========================
CLAUDE.md → skills/*.md + routers/*.py + qwen_tools.py 单向同步

用法：
    python scripts/sync_claude_to_skills.py          # dry-run（只看变更）
    python scripts/sync_claude_to_skills.py --apply   # 真正写入

同步目标（5 处）：
  1. skills/*.md          → Qwen 技能文档（keyword 注入 system prompt）
  2. qwen_tools.py        → QWEN_SYSTEM_PROMPT 尾部注意事项
  3. routers/*.py          → 每个 router 文件头部 # SYNC_NOTES 区块
  4. CLAUDE.md             → 唯一编辑入口（只读不写）
  5. qwen_agent.py         → 通过 skills_loader 间接同步（skill 文件更新即生效）

踩了新坑 → 写 CLAUDE.md → python sync_claude_to_skills.py --apply → 重启 API → 全部同步完毕
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ─── 路径 ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent          # /data/oih/oih-api
CLAUDE_MD    = ROOT / "CLAUDE.md"
SKILLS_DIR   = ROOT / "skills"
ROUTERS_DIR  = ROOT / "routers"
QWEN_TOOLS   = ROOT / "tool_definitions" / "qwen_tools.py"

# ─── 工具名 → skill 文件 ─────────────────────────────────────────────────────

TOOL_SKILL_MAP: Dict[str, str] = {
    "chemprop":    "CHEMPROP_WORKFLOW.md",
    "gromacs":     "GROMACS_WORKFLOW.md",
    "alphafold3":  "ALPHAFOLD3_WORKFLOW.md",
    "autodock":    "AUTODOCK_GPU_WORKFLOW.md",
    "rfdiffusion": "RFDIFFUSION_WORKFLOW.md",
    "proteinmpnn": "PROTEINMPNN_WORKFLOW.md",
    "bindcraft":   "BINDCRAFT_WORKFLOW.md",
    "diffdock":    "DIFFDOCK_WORKFLOW.md",
    "esm":         "ESM_WORKFLOW.md",
    "fpocket":     "FPOCKET_P2RANK_WORKFLOW.md",
    "gnina":       "VINA_GPU_GNINA_WORKFLOW.md",
    "vina":        "VINA_GPU_GNINA_WORKFLOW.md",
    "adc":         "ADC_WORKFLOW.md",
    "rag":         "RAG_WORKFLOW.md",
}

# ─── 工具名 → router 文件 ────────────────────────────────────────────────────

TOOL_ROUTER_MAP: Dict[str, str] = {
    "chemprop":    "ml_tools.py",
    "gromacs":     "md_simulation.py",
    "alphafold3":  "structure_prediction.py",
    "autodock":    "molecular_docking.py",
    "rfdiffusion": "protein_design.py",
    "proteinmpnn": "protein_design.py",
    "bindcraft":   "protein_design.py",
    "diffdock":    "molecular_docking.py",
    "esm":         "ml_tools.py",
    "fpocket":     "pocket_analysis.py",
    "gnina":       "molecular_docking.py",
    "vina":        "molecular_docking.py",
    "adc":         "adc.py",
}

# ─── 标题关键词 → 工具名 ─────────────────────────────────────────────────────

HEADER_KEYWORDS: Dict[str, List[str]] = {
    "chemprop":    ["chemprop", "admet"],
    "gromacs":     ["gromacs", "md workflow", "md simulation"],
    "alphafold3":  ["alphafold", "af3"],
    "autodock":    ["autodock"],
    "rfdiffusion": ["rfdiffusion"],
    "proteinmpnn": ["proteinmpnn"],
    "bindcraft":   ["bindcraft"],
    "diffdock":    ["diffdock"],
    "esm":         ["esm2", "esm "],
    "fpocket":     ["fpocket", "p2rank", "pocket"],
    "gnina":       ["gnina"],
    "vina":        ["vina-gpu", "vina_gpu"],
    "adc":         ["adc", "antibody drug conjugate", "偶联"],
    "rag":         ["rag", "文献检索", "retrieval"],
}

GPU_TOOLS = {"gromacs", "alphafold3", "autodock", "rfdiffusion", "proteinmpnn",
             "bindcraft", "diffdock", "esm", "gnina"}

# ─── 标记 ────────────────────────────────────────────────────────────────────

SYNC_MARKER_START = "<!-- AUTO_SYNC_FROM_CLAUDE_MD -->"
SYNC_MARKER_END   = "<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->"
SYNC_SECTION_TITLE = "## ⚠️ 注意事项（自动同步自 CLAUDE.md）"

QWEN_MARKER_START = "# === 工具注意事项（自动同步自 CLAUDE.md） ==="
QWEN_MARKER_END   = "# === /工具注意事项 ==="

ROUTER_MARKER_START = "# --- SYNC_NOTES (auto-generated from CLAUDE.md, do not edit) ---"
ROUTER_MARKER_END   = "# --- /SYNC_NOTES ---"


# ─── Step 1: 解析 CLAUDE.md ──────────────────────────────────────────────────

def parse_claude_md(path: Path) -> Tuple[Dict[str, List[str]], List[str]]:
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")

    tool_notes: Dict[str, List[str]] = {}
    global_notes: List[str] = []

    current_tool = None
    current_lines: List[str] = []
    in_relevant_section = False

    for line in lines:
        header_match = re.match(r"^(#{2,3})\s+(.+)$", line)
        if header_match:
            if in_relevant_section and current_lines:
                content = _clean_lines(current_lines)
                if content:
                    if current_tool:
                        tool_notes.setdefault(current_tool, []).extend(content)
                    else:
                        global_notes.extend(content)

            title = header_match.group(2).lower()
            current_tool = _match_tool(title)
            current_lines = []
            in_relevant_section = _is_notes_section(title)
            continue

        if in_relevant_section:
            current_lines.append(line)

    if in_relevant_section and current_lines:
        content = _clean_lines(current_lines)
        if content:
            if current_tool:
                tool_notes.setdefault(current_tool, []).extend(content)
            else:
                global_notes.extend(content)

    return tool_notes, global_notes


def _match_tool(title: str) -> str | None:
    for tool, keywords in HEADER_KEYWORDS.items():
        if any(kw in title for kw in keywords):
            return tool
    return None


def _is_notes_section(title: str) -> bool:
    indicators = [
        "注意", "规则", "坑", "修复", "critical", "pitfall", "bug",
        "fix", "warning", "必须", "血泪", "workflow", "pipeline",
        "preprocessing", "layout", "admet", "队列", "queue",
    ]
    return any(ind in title for ind in indicators)


def _clean_lines(lines: List[str]) -> List[str]:
    result = []
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("|") or stripped.startswith("---") or stripped == "---":
            continue
        if stripped.startswith("#") or stripped.startswith("```"):
            continue
        actionable_keywords = [
            "必须", "不要", "不能", "永远", "禁止", "避免", "切勿",
            "always", "never", "must", "critical", "warning", "avoid",
            "注意", "重要", "danger", "device=0", "gpu_id",
            "accelerator", "symlink", "python3.11", "noproxy",
            "不要动", "不要混", "不要搞混", "kill", "restart",
            "oom", "失败", "崩溃", "丢失", "broken", "tc-grps",
            "nvt.gro", "npt.gro", "em.gro", "md.xtc", "验证",
            "静默", "make_ndx", "protein_lig",
        ]
        has_keyword = any(k in stripped.lower() for k in actionable_keywords)
        if has_keyword:
            result.append(stripped)
    return result


# ─── Step 2: 写入 skills/*.md ────────────────────────────────────────────────

def update_skill_file(skill_path: Path, notes: List[str], dry_run: bool) -> bool:
    if not skill_path.exists():
        return False

    text = skill_path.read_text(encoding="utf-8")

    block_lines = [
        "",
        SYNC_MARKER_START,
        SYNC_SECTION_TITLE,
        "",
    ]
    for note in notes:
        if not note.startswith("-"):
            note = f"- {note}"
        block_lines.append(note)
    block_lines.append("")
    block_lines.append(SYNC_MARKER_END)
    new_block = "\n".join(block_lines)

    if SYNC_MARKER_START in text:
        pattern = re.compile(
            re.escape(SYNC_MARKER_START) + r".*?" + re.escape(SYNC_MARKER_END),
            re.DOTALL,
        )
        new_text = pattern.sub(new_block.strip(), text)
    else:
        new_text = text.rstrip() + "\n" + new_block + "\n"

    if new_text == text:
        return False
    if not dry_run:
        skill_path.write_text(new_text, encoding="utf-8")
    return True


# ─── Step 3: 写入 routers/*.py ───────────────────────────────────────────────

def update_router_file(router_path: Path, tool_name: str,
                       notes: List[str], dry_run: bool) -> bool:
    """在 router .py 文件顶部（import 之前）插入/替换 SYNC_NOTES 注释块。"""
    if not router_path.exists():
        return False

    text = router_path.read_text(encoding="utf-8")

    # 构建注释块
    comment_lines = [ROUTER_MARKER_START]
    comment_lines.append(f"# {tool_name.upper()} 注意事项（来自 CLAUDE.md，勿手动编辑）：")
    for note in notes:
        # 去掉 markdown 格式，转成纯注释；确保单行安全
        clean = note.lstrip("-*> ").strip()
        # 替换换行符、反斜杠n、反引号等可能破坏 Python 语法的字符
        clean = clean.replace("\\n", " ").replace("\n", " ")
        clean = clean.replace("'", "'").replace('"', '"')
        clean = re.sub(r"\s+", " ", clean)  # 合并空白
        if clean:
            if len(clean) > 95:
                clean = clean[:92] + "..."
            comment_lines.append(f"#   - {clean}")
    comment_lines.append(ROUTER_MARKER_END)
    new_block = "\n".join(comment_lines)

    if ROUTER_MARKER_START in text:
        pattern = re.compile(
            re.escape(ROUTER_MARKER_START) + r".*?" + re.escape(ROUTER_MARKER_END),
            re.DOTALL,
        )
        new_text = pattern.sub(new_block, text)
    else:
        # 插入在文件 docstring 之后、第一个 import 之前
        import_match = re.search(r"^(import |from )", text, re.MULTILINE)
        if import_match:
            pos = import_match.start()
            new_text = text[:pos] + new_block + "\n\n" + text[pos:]
        else:
            new_text = new_block + "\n\n" + text

    if new_text == text:
        return False
    if not dry_run:
        router_path.write_text(new_text, encoding="utf-8")
    return True


# ─── Step 4: 更新 QWEN_SYSTEM_PROMPT ─────────────────────────────────────────

def update_qwen_system_prompt(
    tool_notes: Dict[str, List[str]],
    global_notes: List[str],
    dry_run: bool,
) -> bool:
    text = QWEN_TOOLS.read_text(encoding="utf-8")

    warning_lines = [QWEN_MARKER_START]

    if global_notes:
        warning_lines.append("\n## 通用规则")
        for n in global_notes:
            warning_lines.append(f"- {n}" if not n.startswith("-") else n)

    for tool, notes in sorted(tool_notes.items()):
        if notes:
            warning_lines.append(f"\n## {tool.upper()} 注意事项")
            for n in notes:
                warning_lines.append(f"- {n}" if not n.startswith("-") else n)

    warning_lines.append(QWEN_MARKER_END)
    warning_block = "\n".join(warning_lines)

    if QWEN_MARKER_START in text:
        pattern = re.compile(
            re.escape(QWEN_MARKER_START) + r".*?" + re.escape(QWEN_MARKER_END),
            re.DOTALL,
        )
        new_text = pattern.sub(warning_block, text)
    else:
        idx = text.rfind('"""')
        if idx == -1:
            print("[WARN] 找不到 QWEN_SYSTEM_PROMPT 结束位置", file=sys.stderr)
            return False
        new_text = text[:idx] + "\n\n" + warning_block + "\n" + text[idx:]

    if new_text == text:
        return False
    if not dry_run:
        QWEN_TOOLS.write_text(new_text, encoding="utf-8")
    return True


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CLAUDE.md → skills + routers + QWEN_SYSTEM_PROMPT 单向同步"
    )
    parser.add_argument("--apply", action="store_true",
                        help="实际写入文件（默认 dry-run 只显示变更）")
    args = parser.parse_args()
    dry_run = not args.apply

    if dry_run:
        print("=== DRY RUN（不写入文件，加 --apply 生效）===\n")

    if not CLAUDE_MD.exists():
        print(f"[ERROR] {CLAUDE_MD} 不存在", file=sys.stderr)
        sys.exit(1)

    # 1. 解析
    tool_notes, global_notes = parse_claude_md(CLAUDE_MD)

    print(f"[解析] 从 CLAUDE.md 提取到：")
    print(f"  通用注意事项: {len(global_notes)} 条")
    for tool, notes in sorted(tool_notes.items()):
        print(f"  {tool}: {len(notes)} 条")

    # 2. 为 GPU 工具追加通用 GPU 规则
    gpu_note = "容器内 NVIDIA_VISIBLE_DEVICES=1，永远用 device=0 / gpu_id=0（不要用1）"
    for tool in GPU_TOOLS:
        tool_notes.setdefault(tool, [])
        if not any("gpu_id" in n.lower() or "device=0" in n.lower()
                    for n in tool_notes[tool]):
            tool_notes[tool].insert(0, gpu_note)

    # 3. 写入 skills
    print(f"\n[同步 skills]")
    skills_changed = 0
    for tool, notes in sorted(tool_notes.items()):
        skill_file = TOOL_SKILL_MAP.get(tool)
        if not skill_file:
            continue
        skill_path = SKILLS_DIR / skill_file
        if not skill_path.exists():
            continue
        changed = update_skill_file(skill_path, notes, dry_run)
        status = "更新" if changed else "无变化"
        print(f"  {tool} → {skill_file}: {status} ({len(notes)} 条)")
        if changed:
            skills_changed += 1

    # 4. 写入 routers
    print(f"\n[同步 routers]")
    routers_changed = 0
    # 按 router 文件聚合多个工具的 notes
    router_aggregated: Dict[str, List[str]] = {}
    router_tools: Dict[str, List[str]] = {}
    for tool, notes in sorted(tool_notes.items()):
        router_file = TOOL_ROUTER_MAP.get(tool)
        if not router_file:
            continue
        router_aggregated.setdefault(router_file, [])
        router_tools.setdefault(router_file, [])
        router_tools[router_file].append(tool)
        for n in notes:
            if n not in router_aggregated[router_file]:  # 去重
                router_aggregated[router_file].append(n)

    for router_file, notes in sorted(router_aggregated.items()):
        router_path = ROUTERS_DIR / router_file
        tools_str = "+".join(router_tools.get(router_file, []))
        changed = update_router_file(router_path, tools_str, notes, dry_run)
        status = "更新" if changed else "无变化"
        print(f"  {tools_str} → {router_file}: {status} ({len(notes)} 条)")
        if changed:
            routers_changed += 1

    # 5. 写入 QWEN_SYSTEM_PROMPT
    print(f"\n[同步 QWEN_SYSTEM_PROMPT]")
    qwen_changed = update_qwen_system_prompt(tool_notes, global_notes, dry_run)
    print(f"  qwen_tools.py: {'更新' if qwen_changed else '无变化'}")

    # 6. 汇总
    total = skills_changed + routers_changed + (1 if qwen_changed else 0)
    print(f"\n{'=' * 40}")
    if dry_run:
        print(f"[DRY RUN] {total} 个文件将被修改。加 --apply 实际写入。")
    else:
        print(f"[DONE] {total} 个文件已更新。重启 API 生效：")
        print(f"  kill $(pgrep -f 'uvicorn main:app') && \\")
        print(f"  cd /data/oih/oih-api && \\")
        print(f"  NO_PROXY='*' no_proxy='*' nohup /data/oih/miniconda/bin/python \\")
        print(f"    -m uvicorn main:app --host 0.0.0.0 --port 8080 --workers 1 \\")
        print(f"    > /tmp/fastapi.log 2>&1 &")


if __name__ == "__main__":
    main()
