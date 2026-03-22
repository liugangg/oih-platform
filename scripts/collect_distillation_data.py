"""
Collect distillation training cases from task history.

Scans data/tasks/*.json for failed and successful tasks,
generates training samples in JSONL format for Qwen3 fine-tuning.

Usage:
    python scripts/collect_distillation_data.py
"""
import json
import os
import glob
from datetime import datetime


def extract_cases(tasks_dir: str, output_file: str):
    cases = []
    for f in sorted(glob.glob(f"{tasks_dir}/*.json")):
        try:
            with open(f) as fp:
                task = json.load(fp)
        except Exception:
            continue

        tool = task.get("tool", "unknown")
        status = task.get("status", "")
        error = task.get("error", "")
        result = task.get("result")

        if error:
            cases.append({
                "input": f"工具={tool} 错误={error[:200]}",
                "reasoning": "",  # 待人工填写
                "action": "",     # 待人工填写
                "outcome": status,
            })
        elif result and status == "completed":
            result_summary = json.dumps(result, ensure_ascii=False)[:300]
            cases.append({
                "input": f"工具={tool} 成功",
                "reasoning": "",
                "action": result_summary,
                "outcome": "success",
            })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"收集了 {len(cases)} 个案例 → {output_file}")


if __name__ == "__main__":
    extract_cases(
        "/data/oih/oih-api/data/tasks/",
        f"/data/oih/oih-api/data/distillation/auto_cases_{datetime.now().strftime('%Y%m%d')}.jsonl",
    )
