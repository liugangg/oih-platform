#!/usr/bin/env python3
"""
Organize /data/oih/outputs/ into typed subdirectories with metadata.

Usage:
    python3 scripts/organize_outputs.py              # dry-run (preview only)
    python3 scripts/organize_outputs.py --apply       # actually move directories
    python3 scripts/organize_outputs.py --archive-test  # move test_* to archive/

Rules:
    md_simulations/   — GROMACS outputs (largest)
    binder_design/    — RFdiffusion/BindCraft/ProteinMPNN/AF3 binder validation
    drug_discovery/   — GNINA/AutoDock/DiffDock/Vina docking
    structure/        — fetch_pdb, AF3 single-chain, ESM
    adc/              — FreeSASA/linker/conjugate
    analysis/         — plots, python analysis
    test/             — all test_* directories (archivable)
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

OUTPUTS_DIR = "/data/oih/outputs"
TASKS_DIR = "/data/oih/oih-api/data/tasks"

# ── Classification rules: regex pattern → category ───────────────────────────
# Order matters — first match wins
CLASSIFY_RULES = [
    # Test directories → test/
    (r"^test_", "test"),

    # GROMACS / MD
    (r"_md$|_md_|gromacs|_100ns|_10ns|_nvt|_npt|_em$", "md_simulations"),

    # ADC assembly
    (r"_freesasa|_sasa|_conjugat|_linker|_adc", "adc"),

    # Binder design pipeline outputs
    (r"_rfd$|_rfd_|rfdiffusion|_bindcraft|_mpnn|_proteinmpnn|_igfold|"
     r"binder_design|_af3_val|tier1|tier3|_6d_|override|clustered|"
     r"pocket_guided|_esm_filter", "binder_design"),

    # Drug discovery / docking
    (r"_dock|_gnina|_autodock|_vina|_diffdock|drug_discovery|_admet",
     "drug_discovery"),

    # Structure prediction / fetch
    (r"fetch_pdb|fetch_molecule|_alphafold|_af3$|_esm$|_esm_|"
     r"discotope|_p2rank|_fpocket|paper_", "structure"),

    # Analysis / plots
    (r"plot|figure|analysis|paper_data", "analysis"),
]


def classify_directory(dirname: str) -> str:
    """Classify an output directory name into a category."""
    name_lower = dirname.lower()
    for pattern, category in CLASSIFY_RULES:
        if re.search(pattern, name_lower):
            return category
    return "uncategorized"


def get_dir_size_mb(path: str) -> float:
    """Get directory size in MB."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024 * 1024)


def find_task_for_output(dirname: str) -> dict | None:
    """Find the task record that produced this output directory."""
    if not os.path.isdir(TASKS_DIR):
        return None
    for task_file in glob.glob(os.path.join(TASKS_DIR, "*.json")):
        try:
            with open(task_file) as f:
                task = json.load(f)
            result = task.get("result") or {}
            output_dir = result.get("output_dir", "")
            if dirname in output_dir:
                return {
                    "task_id": task.get("task_id", ""),
                    "tool": task.get("tool", ""),
                    "status": task.get("status", ""),
                    "created_at": task.get("created_at", ""),
                }
        except (json.JSONDecodeError, OSError):
            pass
    return None


def extract_result_summary(dirpath: str) -> dict:
    """Extract key result metrics from output files."""
    summary = {}

    # Check for ipTM in confidence files
    for cf in glob.glob(os.path.join(dirpath, "**/*summary_confidences*.json"), recursive=True):
        try:
            with open(cf) as f:
                conf = json.load(f)
            iptm = conf.get("iptm", conf.get("ipTM"))
            if iptm is not None:
                if "best_iptm" not in summary or iptm > summary["best_iptm"]:
                    summary["best_iptm"] = round(iptm, 4)
        except (json.JSONDecodeError, OSError):
            pass

    # Check for MPNN scores in FASTA headers
    for fa in glob.glob(os.path.join(dirpath, "**/*.fa"), recursive=True):
        try:
            with open(fa) as f:
                for line in f:
                    if line.startswith(">") and "score=" in line:
                        m = re.search(r"score=([\d.]+)", line)
                        if m:
                            score = float(m.group(1))
                            if "best_mpnn_score" not in summary or score < summary["best_mpnn_score"]:
                                summary["best_mpnn_score"] = round(score, 4)
        except OSError:
            pass

    # Count output files by type
    cif_count = len(glob.glob(os.path.join(dirpath, "**/*.cif"), recursive=True))
    pdb_count = len(glob.glob(os.path.join(dirpath, "**/*.pdb"), recursive=True))
    sdf_count = len(glob.glob(os.path.join(dirpath, "**/*.sdf"), recursive=True))
    csv_count = len(glob.glob(os.path.join(dirpath, "**/*.csv"), recursive=True))

    if cif_count:
        summary["cif_files"] = cif_count
    if pdb_count:
        summary["pdb_files"] = pdb_count
    if sdf_count:
        summary["sdf_files"] = sdf_count
    if csv_count:
        summary["csv_files"] = csv_count

    return summary


def create_metadata(dirpath: str, dirname: str, category: str) -> dict:
    """Create metadata.json for an output directory."""
    stat = os.stat(dirpath)
    size_mb = get_dir_size_mb(dirpath)
    task_info = find_task_for_output(dirname)
    result_summary = extract_result_summary(dirpath)

    metadata = {
        "directory": dirname,
        "category": category,
        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "size_mb": round(size_mb, 1),
        "result_summary": result_summary,
    }
    if task_info:
        metadata["task"] = task_info

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Organize output directories")
    parser.add_argument("--apply", action="store_true",
                        help="Actually move directories (default: dry-run)")
    parser.add_argument("--archive-test", action="store_true",
                        help="Move test_* directories to archive/")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Only generate metadata.json files, don't move anything")
    args = parser.parse_args()

    if not os.path.isdir(OUTPUTS_DIR):
        print(f"Error: {OUTPUTS_DIR} does not exist")
        sys.exit(1)

    # Scan all top-level directories
    entries = sorted(os.listdir(OUTPUTS_DIR))
    dirs = [e for e in entries if os.path.isdir(os.path.join(OUTPUTS_DIR, e))]

    # Classify
    categories = {}
    for dirname in dirs:
        cat = classify_directory(dirname)
        categories.setdefault(cat, []).append(dirname)

    # Print summary
    print(f"{'Category':<20s} {'Count':>6s} {'Size':>10s}")
    print("-" * 40)
    total_size = 0
    for cat in sorted(categories.keys()):
        cat_dirs = categories[cat]
        cat_size = sum(get_dir_size_mb(os.path.join(OUTPUTS_DIR, d)) for d in cat_dirs)
        total_size += cat_size
        size_str = f"{cat_size:.0f} MB" if cat_size < 1024 else f"{cat_size/1024:.1f} GB"
        print(f"  {cat:<18s} {len(cat_dirs):>6d} {size_str:>10s}")
    total_str = f"{total_size:.0f} MB" if total_size < 1024 else f"{total_size/1024:.1f} GB"
    print(f"  {'TOTAL':<18s} {len(dirs):>6d} {total_str:>10s}")

    if args.metadata_only:
        print(f"\n--- Generating metadata.json for {len(dirs)} directories ---")
        for dirname in dirs:
            dirpath = os.path.join(OUTPUTS_DIR, dirname)
            cat = classify_directory(dirname)
            metadata = create_metadata(dirpath, dirname, cat)
            meta_path = os.path.join(dirpath, "metadata.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        print("Done.")
        return

    if not args.apply and not args.archive_test:
        # Dry run — show what would happen
        print("\n--- Dry run (use --apply to execute) ---\n")
        for cat in sorted(categories.keys()):
            print(f"\n{cat}/")
            for dirname in categories[cat][:10]:
                size = get_dir_size_mb(os.path.join(OUTPUTS_DIR, dirname))
                size_str = f"{size:.0f}MB" if size < 1024 else f"{size/1024:.1f}GB"
                print(f"  ← {dirname} ({size_str})")
            if len(categories[cat]) > 10:
                print(f"  ... and {len(categories[cat]) - 10} more")
        return

    if args.archive_test:
        archive_dir = os.path.join(OUTPUTS_DIR, "archive")
        os.makedirs(archive_dir, exist_ok=True)
        test_dirs = categories.get("test", [])
        print(f"\n--- Archiving {len(test_dirs)} test directories ---")
        for dirname in test_dirs:
            src = os.path.join(OUTPUTS_DIR, dirname)
            dst = os.path.join(archive_dir, dirname)
            print(f"  {dirname} → archive/{dirname}")
            shutil.move(src, dst)
        print("Done.")
        return

    if args.apply:
        print(f"\n--- Moving {len(dirs)} directories ---")
        for cat in sorted(categories.keys()):
            cat_dir = os.path.join(OUTPUTS_DIR, cat)
            os.makedirs(cat_dir, exist_ok=True)
            for dirname in categories[cat]:
                src = os.path.join(OUTPUTS_DIR, dirname)
                dst = os.path.join(cat_dir, dirname)
                if os.path.exists(dst):
                    print(f"  SKIP {dirname} (already in {cat}/)")
                    continue
                # Generate metadata before moving
                metadata = create_metadata(src, dirname, cat)
                meta_path = os.path.join(src, "metadata.json")
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                print(f"  {dirname} → {cat}/")
                shutil.move(src, dst)
        print("Done.")


if __name__ == "__main__":
    main()
