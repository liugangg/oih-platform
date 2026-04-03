# ProteinMPNN Complete Workflow (Verified)

## Environment Info
- Container name: oih-proteinmpnn
- Python: python3 (3.10.12)
- Main script: /app/ProteinMPNN/protein_mpnn_run.py
- Model weights: Inside container at /app/ProteinMPNN/vanilla_model_weights/ (built-in, no external mount needed)
- Examples: /app/ProteinMPNN/examples/

## GPU Notes
- NVIDIA_VISIBLE_DEVICES=1 maps host GPU1 as GPU0 inside the container
- ProteinMPNN automatically uses CUDA, no manual device specification needed
- VRAM requirement ~4GB

## Model Weights
| Path | Model | Use Case |
|------|-------|----------|
| vanilla_model_weights/v_48_002.pt | noise=0.02 | High precision, low diversity |
| vanilla_model_weights/v_48_010.pt | noise=0.10 | Balanced |
| vanilla_model_weights/v_48_020.pt | noise=0.20 | Recommended, balanced precision and diversity |
| vanilla_model_weights/v_48_030.pt | noise=0.30 | High diversity |
| ca_model_weights/v_48_020.pt | CA-only | Dedicated for RFdiffusion backbone outputs |
| soluble_model_weights/v_48_020.pt | soluble | Soluble protein optimization |

## Complete Inference Commands

### 1. Standard Sequence Design (Full Chain) - Verified
```bash
docker exec oih-proteinmpnn python3 /app/ProteinMPNN/protein_mpnn_run.py \
  --pdb_path /data/oih/inputs/1IVO.pdb \
  --out_folder /data/oih/outputs/proteinmpnn/ \
  --num_seq_per_target 4 \
  --sampling_temp 0.1 \
  --model_name v_48_020
```
Verified: 4 sequences, 1115 residues long, completed in 11 seconds

### 2. Specify Design Chains (Design only specific chains in multi-chain complexes)

**2026-03-26 Rule: When using API calls, leave chains_to_design as default "auto".**
The router automatically detects the shortest chain (=binder). Do not hardcode "A".

```bash
# When using docker exec directly, manually confirm which chain is the binder:
# First check: python3 -c "import gemmi; [print(c.name, sum(1 for r in c)) for c in gemmi.read_structure('complex.pdb')[0]]"
# Shortest chain = binder = chain to design
docker exec oih-proteinmpnn python3 /app/ProteinMPNN/protein_mpnn_run.py \
  --pdb_path /data/oih/inputs/complex.pdb \
  --pdb_path_chains "B" \
  --out_folder /data/oih/outputs/proteinmpnn/ \
  --num_seq_per_target 8 \
  --sampling_temp 0.1
# Verify: First sequence in FASTA should be 60-120aa (binder), not hundreds of aa (target)
```

### 3. Fix Specific Residues (Preserve active site/key residues)
```bash
# First use helper script to generate fixed_positions file
docker exec oih-proteinmpnn python3 /app/ProteinMPNN/helper_scripts/make_fixed_positions_dict.py \
  --input_path /data/oih/inputs/ \
  --output_path /data/oih/outputs/proteinmpnn/fixed.jsonl \
  --chain_list "A" \
  --position_list "30 31 32 33 55 56"

# Then run design
docker exec oih-proteinmpnn python3 /app/ProteinMPNN/protein_mpnn_run.py \
  --pdb_path /data/oih/inputs/target.pdb \
  --fixed_positions_jsonl /data/oih/outputs/proteinmpnn/fixed.jsonl \
  --out_folder /data/oih/outputs/proteinmpnn/ \
  --num_seq_per_target 8 \
  --sampling_temp 0.1
```

### 4. Process RFdiffusion Output (CA-only backbone)
```bash
docker exec oih-proteinmpnn python3 /app/ProteinMPNN/protein_mpnn_run.py \
  --pdb_path /data/oih/outputs/rfdiffusion/design_0.pdb \
  --ca_only \
  --out_folder /data/oih/outputs/proteinmpnn/ \
  --num_seq_per_target 8 \
  --sampling_temp 0.1 \
  --model_name v_48_020
```

### 5. Soluble Protein Optimized Design
```bash
docker exec oih-proteinmpnn python3 /app/ProteinMPNN/protein_mpnn_run.py \
  --pdb_path /data/oih/inputs/target.pdb \
  --use_soluble_model \
  --out_folder /data/oih/outputs/proteinmpnn/ \
  --num_seq_per_target 8 \
  --sampling_temp 0.15
```

### 6. Score Existing Sequences (Validate sequence-backbone compatibility)
```bash
docker exec oih-proteinmpnn python3 /app/ProteinMPNN/protein_mpnn_run.py \
  --pdb_path /data/oih/inputs/target.pdb \
  --score_only 1 \
  --path_to_fasta /data/oih/inputs/sequences.fasta \
  --out_folder /data/oih/outputs/proteinmpnn/ \
  --save_score 1
```

## Key Parameters
| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| --sampling_temp | Sequence diversity, higher = more diverse | 0.1~0.2 |
| --num_seq_per_target | Number of sequences per structure | 4~20 |
| --model_name | Model version | v_48_020 |
| --ca_only | CA atoms only, for RFdiffusion output | As needed |
| --use_soluble_model | Soluble protein optimization | As needed |
| --batch_size | Batch size, reduce if GPU OOM | 1 |
| --omit_AAs | Exclude certain amino acids, X excluded by default | 'CX' to exclude Cys |
| --save_score | Save per-position log probability scores | 1 (for debugging) |

## Output File Structure
```
/data/oih/outputs/proteinmpnn/
├── seqs/
│   └── 1IVO.fa          # Designed sequences (FASTA format)
├── scores/              # If --save_score enabled
│   └── 1IVO.npy
└── probs/               # If --save_probs enabled
    └── 1IVO.npy
```

## FASTA Output Format
```
>1IVO, score=1.2345, global_score=1.2345, seq_recovery=0.xx
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLVLNSPPSRENRVSSRQFVNHEEMVHHAETREAQHNAQARAAVTGSQPTIRRSAQN
```

### Parsing FASTA in Pipeline (_parse_mpnn_fasta)
Router returns `{fasta_file, output_dir}`, not a structured `sequences` list.
Pipeline uses `_parse_mpnn_fasta(fasta_path)` to parse the FASTA file:
- Reads `score=X` and `global_score=Y` from headers
- Returns `[{"sequence": "...", "score": 1.23, "global_score": 1.23}, ...]`
- Sorted by score ascending (lower is better), top 5 sent to AF3 validation

## Complete Protein Design Pipeline
```
RFdiffusion (backbone design)
    ↓ backbone.pdb
ProteinMPNN (sequence design) --ca_only
    ↓ sequences.fa
AlphaFold3 (structure validation: can the sequence fold back to target backbone?)
    ↓ predicted.pdb
RMSD analysis (designed backbone vs AF3 predicted structure)
```

## Helper Scripts (/app/ProteinMPNN/helper_scripts/)
```bash
docker exec oih-proteinmpnn ls /app/ProteinMPNN/helper_scripts/
# make_fixed_positions_dict.py  — Generate fixed residue dictionary
# make_tied_positions_dict.py   — Generate tied residue dictionary (symmetric design)
# make_bias_AA.py               — Generate amino acid bias dictionary
# parse_multiple_chains.py      — Batch parse multi-chain PDBs
# assign_fixed_chains.py        — Specify fixed chains
```

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## Notes (Auto-synced from CLAUDE.md)

- Container has NVIDIA_VISIBLE_DEVICES=1, always use device=0 / gpu_id=0 (not 1)

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->
