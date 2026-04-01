#!/usr/bin/env python3
"""Convert distillation_merged.jsonl to Qwen3 chat format for SFT training."""

import json
import pathlib

SYSTEM_PROMPT = (
    "You are OIH (Open Intelligence Hub), an AI drug discovery agent. "
    "You autonomously plan and execute computational biology workflows using "
    "30+ tools including AlphaFold3, RFdiffusion, ProteinMPNN, GNINA, GROMACS, "
    "PeSTo, DiscoTope3, and others. When a tool fails or produces unexpected "
    "results, analyze the root cause, fix the issue, and retry. Always verify "
    "outputs before proceeding to the next step."
)

SRC = pathlib.Path("/data/oih/oih-api/data/distillation/distillation_merged.jsonl")
DST = pathlib.Path("/data/oih/oih-api/data/distillation/train_chat.jsonl")


def convert():
    rows = []
    for line in SRC.read_text().strip().splitlines():
        entry = json.loads(line)

        # Build user message from instruction + input
        user_parts = []
        if entry.get("instruction"):
            user_parts.append(entry["instruction"].strip())
        if entry.get("input"):
            user_parts.append(entry["input"].strip())
        user_msg = "\n\n".join(user_parts)

        # Build assistant message from reasoning + action + outcome
        assistant_parts = []
        if entry.get("reasoning"):
            assistant_parts.append(f"**Analysis**: {entry['reasoning'].strip()}")
        if entry.get("action"):
            assistant_parts.append(f"**Action**: {entry['action'].strip()}")
        if entry.get("outcome"):
            assistant_parts.append(f"**Result**: {entry['outcome'].strip()}")
        assistant_msg = "\n\n".join(assistant_parts)

        if not user_msg or not assistant_msg:
            continue

        row = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        }
        rows.append(row)

    with open(DST, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Converted {len(rows)} entries -> {DST}")


if __name__ == "__main__":
    convert()
