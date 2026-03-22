"""
Analysis Router
Tools: execute_python (sandboxed code execution), read_results_file (CSV/JSON reader)
"""

import os
import re
import json
import csv
import subprocess
import tempfile
import logging
from io import StringIO
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

OUTPUTS_ROOT = "/data/oih/outputs"
PLOTS_DIR = os.path.join(OUTPUTS_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Allowed imports whitelist — block os.system, subprocess, etc.
_BANNED_PATTERNS = [
    r"\bimport\s+os\b", r"\bfrom\s+os\b",
    r"\bimport\s+subprocess\b", r"\bfrom\s+subprocess\b",
    r"\bimport\s+shutil\b", r"\bfrom\s+shutil\b",
    r"\b__import__\b", r"\beval\s*\(", r"\bexec\s*\(",
    r"\bopen\s*\(.*(w|a|x)", r"\bos\.system\b", r"\bos\.popen\b",
    r"\bos\.remove\b", r"\bos\.unlink\b", r"\bos\.rmdir\b",
]


# ─── execute_python ──────────────────────────────────────────────────────────

class ExecutePythonRequest(BaseModel):
    python_code: str = Field(..., description="Python code to execute")
    timeout: int = Field(30, ge=5, le=120, description="Max execution time in seconds")


@router.post("/execute_python", summary="Execute Python code with scientific libraries")
async def execute_python(req: ExecutePythonRequest):
    """
    Run user-provided Python code in a sandboxed subprocess.
    Available: matplotlib, pandas, numpy, scipy, rdkit, json, csv, math.
    plt.savefig() outputs go to /data/oih/outputs/plots/.
    """
    code = req.python_code

    # Basic safety check
    for pattern in _BANNED_PATTERNS:
        if re.search(pattern, code):
            raise HTTPException(400, f"Blocked pattern detected: {pattern}")

    # Generate unique plot filename
    import uuid
    plot_id = uuid.uuid4().hex[:8]
    plot_path = os.path.join(PLOTS_DIR, f"plot_{plot_id}.png")

    # Wrap code: inject matplotlib backend + savefig redirect
    wrapper = f"""
import sys, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Override plt.savefig to capture output path
_original_savefig = plt.savefig
_plot_saved = False
_saved_path = None
def _patched_savefig(*args, **kwargs):
    global _plot_saved, _saved_path
    if not args and 'fname' not in kwargs:
        args = ("{plot_path}",)
    elif args:
        import os
        fname = str(args[0])
        if not os.path.isabs(fname):
            fname = os.path.join("{PLOTS_DIR}", fname)
            args = (fname,) + args[1:]
    _original_savefig(*args, **kwargs)
    _plot_saved = True
    _saved_path = str(args[0]) if args else kwargs.get('fname', '{plot_path}')
plt.savefig = _patched_savefig
plt.show = lambda *a, **k: plt.savefig("{plot_path}")

# --- User code ---
{code}
# --- End user code ---

# Auto-save if figure has content but wasn't saved
if plt.gcf().get_axes() and not _plot_saved:
    plt.savefig("{plot_path}")
    _plot_saved = True

if _plot_saved:
    _final_path = _saved_path or "{plot_path}"
    print(f"\\n__PLOT_PATH__={{_final_path}}")
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(wrapper)
        script_path = f.name

    try:
        result = subprocess.run(
            ["/data/oih/miniconda/bin/python", script_path],
            capture_output=True, text=True,
            timeout=req.timeout,
            cwd=OUTPUTS_ROOT,
            env={
                "PATH": "/data/oih/miniconda/bin:/usr/bin:/bin",
                "HOME": "/tmp",
                "MPLCONFIGDIR": "/tmp/matplotlib",
            },
        )

        stdout = result.stdout
        stderr = result.stderr

        # Extract plot path if saved
        detected_plot = None
        if "__PLOT_PATH__=" in stdout:
            for line in stdout.split("\n"):
                if line.startswith("__PLOT_PATH__="):
                    candidate = line.split("=", 1)[1].strip()
                    if os.path.exists(candidate):
                        detected_plot = candidate
            # Clean marker from output
            stdout = "\n".join(
                l for l in stdout.split("\n") if not l.startswith("__PLOT_PATH__=")
            ).strip()

        if result.returncode != 0:
            return {
                "status": "error",
                "output": stdout[-5000:] if stdout else "",
                "error": stderr[-2000:] if stderr else f"Exit code {result.returncode}",
                "plot_path": detected_plot,
            }

        return {
            "status": "ok",
            "output": stdout[-10000:] if stdout else "(no output)",
            "plot_path": detected_plot,
        }

    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"Execution timed out after {req.timeout}s", "output": "", "plot_path": None}
    finally:
        os.unlink(script_path)


# ─── read_results_file ───────────────────────────────────────────────────────

class ReadResultsFileRequest(BaseModel):
    file_path: str = Field(..., description="Path to CSV/JSON/TXT file under /data/oih/outputs/")
    max_chars: int = Field(10000, ge=100, le=50000, description="Max characters to return")


@router.post("/read_results_file", summary="Read a CSV/JSON/TXT results file")
async def read_results_file(req: ReadResultsFileRequest):
    """
    Read a results file from /data/oih/outputs/. Supports CSV, JSON, TXT, LOG, SDF.
    Returns file contents as string (truncated to max_chars).
    """
    path = req.file_path

    # Security: must be under allowed directories
    real = os.path.realpath(path)
    allowed_roots = ["/data/oih/outputs", "/data/oih/inputs", "/data/af3/output"]
    if not any(real.startswith(r) for r in allowed_roots):
        raise HTTPException(403, f"Access denied: path must be under {allowed_roots}")

    if not os.path.exists(path):
        raise HTTPException(404, f"File not found: {path}")

    if os.path.isdir(path):
        # List directory contents
        entries = sorted(os.listdir(path))
        return {
            "file_path": path,
            "type": "directory",
            "contents": entries[:200],
            "total_entries": len(entries),
        }

    size = os.path.getsize(path)
    ext = os.path.splitext(path)[1].lower()

    try:
        if ext == ".json":
            with open(path) as f:
                data = json.load(f)
            text = json.dumps(data, indent=2, ensure_ascii=False)
        elif ext == ".csv":
            with open(path) as f:
                reader = csv.reader(f)
                rows = list(reader)
            # Format as readable table
            if rows:
                header = rows[0]
                text = " | ".join(header) + "\n" + "-" * 40 + "\n"
                for row in rows[1:101]:  # cap at 100 data rows
                    text += " | ".join(row) + "\n"
                if len(rows) > 101:
                    text += f"\n... ({len(rows) - 1} total rows, showing first 100)"
            else:
                text = "(empty CSV)"
        else:
            with open(path, errors="replace") as f:
                text = f.read()

        truncated = len(text) > req.max_chars
        return {
            "file_path": path,
            "type": ext.lstrip(".") or "txt",
            "size_bytes": size,
            "contents": text[:req.max_chars],
            "truncated": truncated,
        }

    except UnicodeDecodeError:
        return {
            "file_path": path,
            "type": "binary",
            "size_bytes": size,
            "contents": f"(binary file, {size} bytes, cannot display as text)",
            "truncated": False,
        }
    except Exception as e:
        raise HTTPException(500, f"Error reading file: {e}")
