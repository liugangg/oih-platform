# AlphaFold3 Workflow Documentation

## Basic Information
- Container: oih-alphafold3
- Software path: /app/alphafold/run_alphafold.py
- Model path: /data/alphafold3_models/ (af3.bin)
- Database path: /data/alphafold3_db/
- Input/Output: /data/af3/input/ and /data/af3/output/
- GPU: NVIDIA_VISIBLE_DEVICES=1 -> container cuda:0
- Inference time: ~82 seconds (100aa single chain)

## Input JSON Format
```json
{
  "name": "job_name",
  "dialect": "alphafold3",
  "version": 1,
  "sequences": [
    {
      "protein": {
        "id": ["A"],
        "sequence": "SEQUENCE..."
      }
    }
  ],
  "modelSeeds": [1]
}
```

## Supported Input Types
- protein: protein sequences
- rna/dna: nucleic acid sequences
- ligand: small molecules (SMILES)
- Multi-chain complexes: multiple sequences entries

## Run Command
```bash
docker exec oih-alphafold3 bash -c "
python3 /app/alphafold/run_alphafold.py \
  --json_path=/data/af3/input/<input>.json \
  --model_dir=/data/alphafold3_models \
  --db_dir=/data/alphafold3_db \
  --output_dir=/data/af3/output/<job_name> \
  --flash_attention_implementation=triton"
```

## Output Files
- *_model.cif: best structure (mmCIF format)
- *_confidences.json: per-residue pLDDT confidence
- *_summary_confidences.json: overall metrics (ipTM, pTM, etc.)
- *_ranking_scores.csv: multi-seed ranking
- seed-X_sample-Y/: individual sampled structures

## Notes
- Data pipeline (MSA search) is CPU-intensive, runs at full speed on 128 cores
- --run_data_pipeline=false skips MSA (requires existing MSA cache)
- --run_inference=false runs MSA only without inference
- Multi-chain complex prediction ipTM>0.6 indicates reliable binding
- Warning: do not mount host cuda lib64 (same as BindCraft, JAX includes its own CUDA)

## Task Scheduling Rules

### Never Degrade (_NO_DEGRADED_TOOLS)
AF3 and BindCraft never enter the DEGRADED queue (CPU fallback would OOM crash).
When VRAM is insufficient, retry checking GPU1 every 60s, enter GPU queue when available.

### Timeout Handling (_wait_for_af3_task)
Pipeline AF3 uses infinite wait: poll task status every 30s.
Only the following conditions are considered failures:
- OOM (exit code 1 + memory keyword)
- Container exit 1
- Task cancelled
- 10 consecutive identical errors

**Timeout misdiagnosis**: if AF3 "timed out" but may have completed, check the output directory:
```bash
find /data/oih/outputs -name "*model.cif" | grep af3
cat <output_dir>/*ranking_scores.csv   # read actual ipTM
```
If CIF file exists -> timeout was a misdiagnosis, use actual ipTM to continue pipeline.

### AF3 Call Interval in Pipeline
When submitting multiple AF3 tasks consecutively, space each 5 seconds apart to avoid GPU OOM.

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## Warning: Notes (auto-synced from CLAUDE.md)

- Container NVIDIA_VISIBLE_DEVICES=1, always use device=0 / gpu_id=0 (do not use 1)
- **Root cause**: v3 task all 5 AF3 designs failed. rank1 timed out at 1800s (but actually completed, ipTM=0.48), rank2-5 were routed to DEGRADED queue OOM exit 1
- **Fix 1**: added `_wait_for_af3_task()` infinite wait function, polls every 30s, only considers OOM/exit1/cancelled/10 consecutive same errors as failure
- No longer degrades to DEGRADED causing OOM crash
- **Cause**: when CIF->PDB conversion fails `af3_pdb=None` -> `"No PDB for FreeSASA"` error
- During AF3 validation, truncate antigen sequence by domain to avoid full-length sequences reducing ipTM accuracy
- `num_seeds=3` for binder_design_pipeline AF3 validation (speed/accuracy balance)

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->

## DOMAIN_REGISTRY Antigen Domain Truncation (2026-03-22 validated)

AF3 is more accurate for short sequence complex predictions. Pipeline automatically truncates antigen domains per DOMAIN_REGISTRY.

### Registered Targets
| Target | Domain | UniProt Range | Truncated Length | Description |
|--------|--------|--------------|-----------------|-------------|
| HER2 | domain4 | 488-630 | ~202aa | trastuzumab epitope C558-C573 |
| HER2 | domain2 | 172-308 | ~196aa | pertuzumab epitope |
| CD36 | extracellular | 30-439 | ~469aa | Extracellular loop |
| EGFR | domain3 | 361-481 | ~180aa | cetuximab epitope |
| PD-L1 | IgV | 19-127 | ~168aa | PD-1 binding interface |

### HER2 Experimental Results
| Design | Antigen Length | ipTM | Conclusion |
|--------|--------------|------|------------|
| val_0 | 1015aa (full-length) | 0.84 | Pass |
| val_1 | 1015aa (full-length) | 0.70 | Pass |
| val_2 | 202aa (Domain4) | **0.86** | Pass - truncation is superior |

### Truncation Rules
- padding = 30aa (retain 30aa flexible region on each side of domain boundary)
- For unknown proteins, truncate hotspot center +/- 100aa
- Multiple druggable domains -> run AF3 separately, take highest ipTM

## AF3 Dynamic Timeout
| Total Sequence Length | Timeout |
|----------------------|---------|
| <500aa | 1200s (20min) |
| 500-1000aa | 2400s (40min) |
| >1000aa | 3600s (60min) |

## GPU VRAM Limitations
- AF3 ~20GB, must run exclusively on GPU
- GPU Semaphore = 1 (only 1 GPU task at a time)
- RTX 4090 total 44GB, two large models concurrently = OOM
