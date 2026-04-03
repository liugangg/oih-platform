# OIH Platform Self-Diagnosis and Repair Manual

## Core Principle
When any computational task fails, act autonomously without waiting for user instructions:
Diagnose → Adjust parameters → Retry → Verify → Report results

## AF3 Failure Diagnosis Tree

### ipTM=None (Timeout or Crash)
1. Check whether the output directory contains CIF files:
   find /data/oih/outputs -name "*model.cif" | grep -i af3 | head -5
2. CIF exists → Timeout false positive; read the actual scores:
   cat <output_dir>/*ranking_scores.csv
   → Use the real ipTM to continue the pipeline
3. No CIF → True failure; investigate the cause:
   docker logs oih-alphafold3 --tail 30
   nvidia-smi --query-gpu=memory.used,memory.free --format=csv
   → OOM: Wait for GPU to free up, then retry
   → Other errors: Check /tmp/fastapi.log

### ipTM < 0.5 (Poor Design Quality)
Cause: Incorrect hotspot residues or insufficient number of designs
Remediation steps:
1. Switch hotspot to the actual binding interface:
   - HER2 Domain II (pertuzumab epitope): T144, R150, S175, R177
   - HER2 Domain IV (trastuzumab epitope): S310, T311, N344, R375
   - TROP2 binding site: K65, R87, D110
2. Increase num_designs: 10 → 50
3. Lower validation threshold: 0.6 → 0.5 (get a successful run first, then optimize)

### ipTM 0.5-0.75 (Low-Confidence Pass)
- Continue the pipeline; flag as low_confidence
- Note in the report that wet-lab experimental validation is required

## VRAM Issue Diagnosis

### Quick Check
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

### Routing Rules
- GPU0 (23GB): Qwen3-14B resident; remaining space is unstable
- GPU1 (46GB): All bioinformatics tools; device=0 inside containers
- AF3/BindCraft require ≥20GB → Must use GPU1
- If GPU1 is insufficient → Wait; do not downgrade

## RFdiffusion Design Quality Optimization

### Automatic Parameter Adjustment Strategy
Round 1: num_designs=10, quick hotspot validation
→ All failed: Switch hotspot and retry
→ Some passed (ipTM>0.5): Increase num_designs=50
→ None passed: Lower AF3 threshold to 0.5

### HER2 Target Optimal Hotspots
- Avoid the trastuzumab epitope (competes with existing drugs)
- Recommended novel epitope: Domain II K505, E558, D561
  or Domain III: H473, N476

## GROMACS Common Issues

### nvt.gro Does Not Exist
→ NVT step failed silently; check:
   cat /data/oih/outputs/<job>/nvt.log | tail -20

### tc-grps Error
→ Dynamically detect protein+ligand group names; do not hardcode

## RFdiffusion ContigMap Crash

### AssertionError: ('A', N) is not in pdb file!
Cause: The PDB has residue gaps (crystallographically missing residues); ContigMap fails when traversing a continuous range that includes missing residues.
Fix: The router automatically renumbers the PDB (removes HETATM + sequential renumbering), implemented in `_renumber_pdb_for_rfdiffusion()`.
If the error persists: Check for non-standard residues or a truncated PDB file.

## Pipeline Automatic Retry Strategy

Automatic retry rules after failure:
1. Timeout errors → Retry immediately (up to 3 times)
2. OOM errors → Wait 60s, then retry
3. Parameter errors → Adjust parameters, then retry
4. File not found → Check whether the previous step completed

## Distillation Training Data

Location: `data/distillation/`
Format: JSONL, one case per line (task_id, error_msg, tool, fix_applied, outcome)
Categories: gpu / container / tool / pipeline / proxy / dashboard / abandoned / reference
Goal: Accumulate 100+ cases for LoRA fine-tuning of Qwen3-14B
Collection script: `scripts/collect_distillation_data.py` (automatically extracts failure patterns from task history)
