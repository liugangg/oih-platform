#!/usr/bin/env python3
"""
OIH Autopilot — Background automatic monitoring script
=======================================================
Scans task status every 60s and automatically advances subsequent steps:

1. Pipeline completed → collect AF3 results
2. AF3 ipTM >= 0.4 → automatically run ipSAE
3. ipSAE > 0.15 → automatically CIF→PDB → FreeSASA → ADC conjugation
4. Detect GPU zombie processes → auto cleanup
5. All results written to autopilot_results.jsonl

Usage: nohup python3 scripts/autopilot.py &
"""
import json
import os
import sys
import time
import glob
import subprocess
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────
API_BASE = "http://localhost:8080/api/v1"
OUTPUTS_DIR = "/data/oih/outputs"
RESULTS_LOG = os.path.join(OUTPUTS_DIR, "autopilot_results.jsonl")
POLL_INTERVAL = 60  # seconds
IPTM_THRESHOLD = 0.4       # minimum ipTM to proceed with ipSAE
IPSAE_THRESHOLD = 0.15     # minimum ipSAE to proceed with ADC
GPU_ZOMBIE_TIMEOUT = 7200  # seconds — kill AF3 processes running longer than this

# NHS-PEG4 linker (Kadcyla-class)
NHS_PEG4_SMILES = "O=C(CCOCCOCCOCCOCCCNC=O)ON1C(=O)CCC1=O"
# MMAE payload
MMAE_SMILES = "CCC(C)C(C(=O)NC(CC1=CC=CC=C1)C(=O)OC)NC(=O)C(NC(=O)C(NC(=O)C1=CC=CC=C1O)C(C)C)C(C)C"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [autopilot] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/autopilot.log"),
    ]
)
log = logging.getLogger("autopilot")

# Track what we've already processed (persist across restarts)
STATE_FILE = "/tmp/autopilot_state.json"


def load_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {"processed_ipsae": [], "processed_adc": [], "processed_cleanup": []}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def log_result(entry: dict):
    """Append a result to the JSONL log."""
    entry["timestamp"] = datetime.now().isoformat()
    with open(RESULTS_LOG, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info("Result logged: %s", json.dumps(entry, ensure_ascii=False)[:200])


def api_get(path):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning("API GET %s failed: %s", path, e)
        return None


def api_post(path, data):
    try:
        r = requests.post(f"{API_BASE}{path}", json=data, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning("API POST %s failed: %s", path, e)
        return None


def poll_task(task_id, timeout=600, interval=15):
    """Poll until task completes or times out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        t = api_get(f"/tasks/{task_id}")
        if not t:
            time.sleep(interval)
            continue
        if t.get("status") == "completed":
            return t.get("result", {})
        if t.get("status") in ("failed", "cancelled"):
            log.warning("Task %s %s: %s", task_id[:8], t["status"], t.get("error", "")[:100])
            return None
        time.sleep(interval)
    log.warning("Task %s poll timeout (%ds)", task_id[:8], timeout)
    return None


# ─── Step 1: Find completed AF3 results ─────────────────────

def find_af3_results():
    """Scan output dirs for AF3 results with ipTM scores."""
    results = []
    for conf_file in glob.glob(f"{OUTPUTS_DIR}/**/alphafold3/**/*_summary_confidences.json", recursive=True):
        # Skip per-seed files, only top-level
        parts = conf_file.split("/")
        # pattern: .../job_name/alphafold3/job_name/job_name_summary_confidences.json
        parent_dir = os.path.dirname(conf_file)
        if "seed-" in os.path.basename(parent_dir):
            continue
        try:
            d = json.load(open(conf_file))
            iptm = d.get("iptm")
            if iptm is None:
                continue
            job_name = os.path.basename(parent_dir)
            cif_file = os.path.join(parent_dir, f"{job_name}_model.cif")
            conf_full = os.path.join(parent_dir, f"{job_name}_confidences.json")
            results.append({
                "job_name": job_name,
                "iptm": iptm,
                "ptm": d.get("ptm"),
                "cif_file": cif_file if os.path.exists(cif_file) else None,
                "confidences_file": conf_full if os.path.exists(conf_full) else None,
                "summary_file": conf_file,
                "dir": parent_dir,
            })
        except Exception:
            continue
    return results


# ─── Step 2: Run ipSAE ──────────────────────────────────────

def run_ipsae(af3_result: dict) -> dict | None:
    """Run ipSAE on an AF3 result. Returns scores or None."""
    conf = af3_result.get("confidences_file")
    cif = af3_result.get("cif_file")
    if not conf or not cif or not os.path.exists(conf) or not os.path.exists(cif):
        log.warning("ipSAE skip %s: missing files", af3_result["job_name"])
        return None

    # Fix permissions (AF3 outputs are root-owned)
    subprocess.run(
        ["docker", "exec", "oih-alphafold3", "chmod", "-R", "777", af3_result["dir"]],
        capture_output=True, timeout=10,
    )

    try:
        from ipsae import IpsaeCalculator
        calc = IpsaeCalculator(pae_cutoff=10.0, dist_cutoff=10.0)
        calc.calculate(conf, cif)

        # Read the output txt file
        txt_file = cif.replace("_model.cif", "_model_10_10.txt")
        if not os.path.exists(txt_file):
            log.warning("ipSAE output not found: %s", txt_file)
            return None

        # Parse the "max" line
        for line in open(txt_file):
            parts = line.split()
            if len(parts) > 12 and parts[4] == "max":
                return {
                    "ipsae": float(parts[5]),
                    "ipsae_d0chn": float(parts[6]),
                    "ipsae_d0dom": float(parts[7]),
                    "iptm_af": float(parts[8]),
                    "pdockq": float(parts[10]),
                    "pdockq2": float(parts[11]),
                    "lis": float(parts[12]),
                    "txt_file": txt_file,
                }
    except Exception as e:
        log.error("ipSAE failed for %s: %s", af3_result["job_name"], e)
    return None


# ─── Step 3: CIF→PDB + FreeSASA + ADC ───────────────────────

def run_adc_pipeline(af3_result: dict, ipsae_scores: dict) -> dict | None:
    """Convert CIF→PDB, run FreeSASA, then ADC conjugation."""
    job = af3_result["job_name"]
    cif = af3_result["cif_file"]
    pdb = cif.replace("_model.cif", "_for_sasa.pdb")

    # Step 3a: CIF → PDB
    if not os.path.exists(pdb):
        try:
            sys.path.insert(0, "/data/oih/oih-api")
            from routers.pipeline import convert_af3_cif_to_pdb
            convert_af3_cif_to_pdb(cif, pdb, protein_only=False)
            log.info("CIF→PDB: %s", pdb)
        except Exception as e:
            log.error("CIF→PDB failed for %s: %s", job, e)
            return None

    # Step 3b: FreeSASA
    sasa_ref = api_post("/design/freesasa", {
        "job_name": f"{job}_autopilot_sasa",
        "input_pdb": pdb,
    })
    if not sasa_ref:
        return None

    sasa_result = poll_task(sasa_ref["task_id"], timeout=120)
    if not sasa_result:
        return None

    sites = sasa_result.get("conjugation_sites", [])
    if not sites:
        log.warning("No conjugation sites for %s", job)
        return None

    # Pick best Lys site
    lys_sites = [s for s in sites if s.get("residue", "").startswith("K")]
    if not lys_sites:
        log.warning("No Lys sites for %s, using best overall", job)
        best_site = sites[0]
    else:
        best_site = lys_sites[0]

    site_residue = best_site["residue"]
    site_chain = best_site.get("chain", "A")
    site_sasa = best_site.get("sasa", 0)
    log.info("FreeSASA %s: best site %s:%s (%.1f Å²)", job, site_chain, site_residue, site_sasa)

    # Step 3c: ADC Conjugation
    adc_ref = api_post("/adc/rdkit_conjugate", {
        "job_name": f"{job}_autopilot_adc",
        "antibody_pdb": pdb,
        "conjugation_site": site_residue,
        "conjugation_chain": site_chain,
        "linker_smiles": NHS_PEG4_SMILES,
        "payload_smiles": MMAE_SMILES,
        "linker_name": "NHS-PEG4",
        "payload_name": "MMAE",
        "reaction_type": "nhs_amine",
    })
    if not adc_ref:
        return None

    adc_result = poll_task(adc_ref["task_id"], timeout=120)
    if not adc_result:
        return None

    return {
        "conjugation_site": f"{site_chain}:{site_residue}",
        "sasa": site_sasa,
        "covalent": adc_result.get("covalent"),
        "reaction": adc_result.get("reaction_type_used"),
        "linker": "NHS-PEG4",
        "payload": "MMAE",
        "adc_sdf": adc_result.get("output_sdf"),
    }


# ─── Step 4: GPU Zombie Cleanup ─────────────────────────────

def cleanup_gpu_zombies():
    """Kill AF3 processes running longer than GPU_ZOMBIE_TIMEOUT."""
    try:
        result = subprocess.run(
            ["docker", "exec", "oih-alphafold3", "ps", "aux"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if "run_alphafold.py" not in line:
                continue
            parts = line.split()
            if len(parts) < 11:
                continue
            pid = parts[1]
            etime = parts[9]  # ELAPSED time
            # Parse elapsed time (formats: MM:SS, HH:MM:SS, D-HH:MM:SS)
            try:
                if "-" in etime:
                    days, rest = etime.split("-")
                    h, m, s = rest.split(":")
                    total_secs = int(days) * 86400 + int(h) * 3600 + int(m) * 60 + int(s)
                elif etime.count(":") == 2:
                    h, m, s = etime.split(":")
                    total_secs = int(h) * 3600 + int(m) * 60 + int(s)
                else:
                    m, s = etime.split(":")
                    total_secs = int(m) * 60 + int(s)
            except ValueError:
                continue

            if total_secs > GPU_ZOMBIE_TIMEOUT:
                log.warning("Killing zombie AF3 process PID=%s (running %ds)", pid, total_secs)
                subprocess.run(
                    ["docker", "exec", "oih-alphafold3", "kill", "-9", pid],
                    capture_output=True, timeout=5,
                )
    except Exception as e:
        log.warning("Zombie cleanup failed: %s", e)


# ─── Main Loop ───────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("OIH Autopilot started — polling every %ds", POLL_INTERVAL)
    log.info("ipTM threshold: %.2f, ipSAE threshold: %.2f", IPTM_THRESHOLD, IPSAE_THRESHOLD)
    log.info("Results log: %s", RESULTS_LOG)
    log.info("=" * 60)

    state = load_state()

    while True:
        try:
            # ── Scan AF3 results ──
            af3_results = find_af3_results()
            passing = [r for r in af3_results if r["iptm"] >= IPTM_THRESHOLD]

            # ── Run ipSAE on qualifying designs ──
            for r in passing:
                if r["job_name"] in state["processed_ipsae"]:
                    continue
                log.info("Running ipSAE on %s (ipTM=%.3f)", r["job_name"], r["iptm"])
                scores = run_ipsae(r)
                state["processed_ipsae"].append(r["job_name"])
                save_state(state)

                if scores:
                    log_result({
                        "type": "ipsae",
                        "job_name": r["job_name"],
                        "iptm": r["iptm"],
                        **scores,
                    })

                    # ── Run ADC if ipSAE passes ──
                    if scores["ipsae"] >= IPSAE_THRESHOLD and r["job_name"] not in state["processed_adc"]:
                        log.info("Running ADC for %s (ipSAE=%.3f)", r["job_name"], scores["ipsae"])
                        adc = run_adc_pipeline(r, scores)
                        state["processed_adc"].append(r["job_name"])
                        save_state(state)

                        if adc:
                            log_result({
                                "type": "adc",
                                "job_name": r["job_name"],
                                "iptm": r["iptm"],
                                "ipsae": scores["ipsae"],
                                **adc,
                            })
                else:
                    log_result({
                        "type": "ipsae_failed",
                        "job_name": r["job_name"],
                        "iptm": r["iptm"],
                    })

            # ── GPU zombie cleanup (every cycle) ──
            cleanup_gpu_zombies()

            # ── Status summary ──
            running_tasks = api_get("/tasks/?limit=10")
            if running_tasks:
                tasks = running_tasks.get("tasks", running_tasks) if isinstance(running_tasks, dict) else running_tasks
                active = [t for t in tasks if t.get("status") in ("running", "pending")]
                if active:
                    for t in active:
                        log.info("  ACTIVE: %s %s %d%% %s",
                                 t.get("tool", "?"), t.get("task_id", "")[:8],
                                 t.get("progress", 0), t.get("progress_msg", "")[:50])
                else:
                    log.info("  No active tasks. Scanning for new AF3 results...")

        except KeyboardInterrupt:
            log.info("Autopilot stopped by user")
            break
        except Exception as e:
            log.error("Autopilot cycle error: %s", e, exc_info=True)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
