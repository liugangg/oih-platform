#!/usr/bin/env python3
"""
sync_claude_to_skills.py
========================
CLAUDE.md -> skills/*.md + routers/*.py + qwen_tools.py one-way sync

Usage:
    python scripts/sync_claude_to_skills.py          # dry-run (preview changes)
    python scripts/sync_claude_to_skills.py --apply   # actually write files

Sync targets (5 locations):
  1. skills/*.md          -> Qwen skill docs (keyword-injected into system prompt)
  2. qwen_tools.py        -> QWEN_SYSTEM_PROMPT appended notes
  3. routers/*.py          -> # SYNC_NOTES block at top of each router file
  4. CLAUDE.md             -> single source of truth (read-only)
  5. qwen_agent.py         -> indirectly synced via skills_loader (skill file updates take effect)

New pitfall found -> edit CLAUDE.md -> python sync_claude_to_skills.py --apply -> restart API -> all synced
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ─── Paths ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent          # /data/oih/oih-api
CLAUDE_MD    = ROOT / "CLAUDE.md"
SKILLS_DIR   = ROOT / "skills"
ROUTERS_DIR  = ROOT / "routers"
QWEN_TOOLS   = ROOT / "tool_definitions" / "qwen_tools.py"

# ─── Tool name -> skill file ──────────────────────────────────────────────────

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

# ─── Tool name -> router file ─────────────────────────────────────────────────

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

# ─── Header keywords -> tool name ─────────────────────────────────────────────

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
    "adc":         ["adc", "antibody drug conjugate", "conjugate"],
    "rag":         ["rag", "literature search", "retrieval"],
}

GPU_TOOLS = {"gromacs", "alphafold3", "autodock", "rfdiffusion", "proteinmpnn",
             "bindcraft", "diffdock", "esm", "gnina"}

# ─── Markers ─────────────────────────────────────────────────────────────────

SYNC_MARKER_START = "<!-- AUTO_SYNC_FROM_CLAUDE_MD -->"
SYNC_MARKER_END   = "<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->"
SYNC_SECTION_TITLE = "## Notes (auto-synced from CLAUDE.md)"

QWEN_MARKER_START = "# === Tool notes (auto-synced from CLAUDE.md) ==="
QWEN_MARKER_END   = "# === /Tool notes ==="

ROUTER_MARKER_START = "# --- SYNC_NOTES (auto-generated from CLAUDE.md, do not edit) ---"
ROUTER_MARKER_END   = "# --- /SYNC_NOTES ---"


# ─── Step 1: Parse CLAUDE.md ──────────────────────────────────────────────────

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
        "note", "rule", "pitfall", "fix", "critical", "pitfall", "bug",
        "fix", "warning", "must", "lesson", "workflow", "pipeline",
        "preprocessing", "layout", "admet", "queue", "queue",
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
            "must", "do not", "cannot", "always", "never", "avoid", "forbidden",
            "always", "never", "must", "critical", "warning", "avoid",
            "note", "important", "danger", "device=0", "gpu_id",
            "accelerator", "symlink", "python3.11", "noproxy",
            "do not touch", "do not mix", "do not confuse", "kill", "restart",
            "oom", "fail", "crash", "lost", "broken", "tc-grps",
            "nvt.gro", "npt.gro", "em.gro", "md.xtc", "verify",
            "silent", "make_ndx", "protein_lig",
        ]
        has_keyword = any(k in stripped.lower() for k in actionable_keywords)
        if has_keyword:
            result.append(stripped)
    return result


# ─── Step 2: Write to skills/*.md ─────────────────────────────────────────────

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


# ─── Step 3: Write to routers/*.py ────────────────────────────────────────────

def update_router_file(router_path: Path, tool_name: str,
                       notes: List[str], dry_run: bool) -> bool:
    """Insert/replace SYNC_NOTES comment block at top of router .py file (before imports)."""
    if not router_path.exists():
        return False

    text = router_path.read_text(encoding="utf-8")

    # Build comment block
    comment_lines = [ROUTER_MARKER_START]
    comment_lines.append(f"# {tool_name.upper()} notes (from CLAUDE.md, do not edit manually):")
    for note in notes:
        # Strip markdown formatting, convert to plain comment; ensure single-line safe
        clean = note.lstrip("-*> ").strip()
        # Replace newlines, backslash-n, backticks that could break Python syntax
        clean = clean.replace("\\n", " ").replace("\n", " ")
        clean = clean.replace("'", "'").replace('"', '"')
        clean = re.sub(r"\s+", " ", clean)  # collapse whitespace
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
        # Insert after docstring, before first import
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


# ─── Step 4: Update QWEN_SYSTEM_PROMPT ────────────────────────────────────────

def update_qwen_system_prompt(
    tool_notes: Dict[str, List[str]],
    global_notes: List[str],
    dry_run: bool,
) -> bool:
    text = QWEN_TOOLS.read_text(encoding="utf-8")

    warning_lines = [QWEN_MARKER_START]

    if global_notes:
        warning_lines.append("\n## General rules")
        for n in global_notes:
            warning_lines.append(f"- {n}" if not n.startswith("-") else n)

    for tool, notes in sorted(tool_notes.items()):
        if notes:
            warning_lines.append(f"\n## {tool.upper()} notes")
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
            print("[WARN] Cannot find QWEN_SYSTEM_PROMPT closing position", file=sys.stderr)
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
        description="CLAUDE.md → skills + routers + QWEN_SYSTEM_PROMPT one-way sync"
    )
    parser.add_argument("--apply", action="store_true",
                        help="Actually write files (default is dry-run, showing changes only)")
    args = parser.parse_args()
    dry_run = not args.apply

    if dry_run:
        print("=== DRY RUN (no files written; add --apply to execute) ===\n")

    if not CLAUDE_MD.exists():
        print(f"[ERROR] {CLAUDE_MD} does not exist", file=sys.stderr)
        sys.exit(1)

    # 1. Parse
    tool_notes, global_notes = parse_claude_md(CLAUDE_MD)

    print(f"[Parse] Extracted from CLAUDE.md:")
    print(f"  General notes: {len(global_notes)} items")
    for tool, notes in sorted(tool_notes.items()):
        print(f"  {tool}: {len(notes)} items")

    # 2. Append common GPU rules for GPU tools
    gpu_note = "Inside the container NVIDIA_VISIBLE_DEVICES=1; always use device=0 / gpu_id=0 (do not use 1)"
    for tool in GPU_TOOLS:
        tool_notes.setdefault(tool, [])
        if not any("gpu_id" in n.lower() or "device=0" in n.lower()
                    for n in tool_notes[tool]):
            tool_notes[tool].insert(0, gpu_note)

    # 3. Write to skills
    print(f"\n[Sync skills]")
    skills_changed = 0
    for tool, notes in sorted(tool_notes.items()):
        skill_file = TOOL_SKILL_MAP.get(tool)
        if not skill_file:
            continue
        skill_path = SKILLS_DIR / skill_file
        if not skill_path.exists():
            continue
        changed = update_skill_file(skill_path, notes, dry_run)
        status = "updated" if changed else "no change"
        print(f"  {tool} → {skill_file}: {status} ({len(notes)} items)")
        if changed:
            skills_changed += 1

    # 4. Write to routers
    print(f"\n[Sync routers]")
    routers_changed = 0
    # Aggregate notes from multiple tools per router file
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
            if n not in router_aggregated[router_file]:  # deduplicate
                router_aggregated[router_file].append(n)

    for router_file, notes in sorted(router_aggregated.items()):
        router_path = ROUTERS_DIR / router_file
        tools_str = "+".join(router_tools.get(router_file, []))
        changed = update_router_file(router_path, tools_str, notes, dry_run)
        status = "updated" if changed else "no change"
        print(f"  {tools_str} → {router_file}: {status} ({len(notes)} entries)")
        if changed:
            routers_changed += 1

    # 5. Write to QWEN_SYSTEM_PROMPT
    print(f"\n[Sync QWEN_SYSTEM_PROMPT]")
    qwen_changed = update_qwen_system_prompt(tool_notes, global_notes, dry_run)
    print(f"  qwen_tools.py: {'updated' if qwen_changed else 'no change'}")

    # 6. Summary
    total = skills_changed + routers_changed + (1 if qwen_changed else 0)
    print(f"\n{'=' * 40}")
    if dry_run:
        print(f"[DRY RUN] {total} file(s) would be modified. Add --apply to write.")
    else:
        print(f"[DONE] {total} file(s) updated. Restart API to take effect:")
        print(f"  kill $(pgrep -f 'uvicorn main:app') && \\")
        print(f"  cd /data/oih/oih-api && \\")
        print(f"  NO_PROXY='*' no_proxy='*' nohup /data/oih/miniconda/bin/python \\")
        print(f"    -m uvicorn main:app --host 0.0.0.0 --port 8080 --workers 1 \\")
        print(f"    > /tmp/fastapi.log 2>&1 &")


if __name__ == "__main__":
    main()
