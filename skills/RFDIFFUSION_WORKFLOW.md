# RFdiffusion Complete Workflow (Verified)

## Environment Information
- Container name: oih-rfdiffusion
- Image: rfdiffusion:latest
- Python: python3 (3.10.12), note: not python, there is no python command
- Inference script: /app/RFdiffusion/scripts/run_inference.py
- Example scripts: /app/RFdiffusion/examples/

## Warning: Critical Bug Fix (must be executed after new container deployment)

### Bug: incorrect default input_pdb path during unconditional design
model_runners.py uses a relative path for the default pdb, causing file not found:
```
FileNotFoundError: .../rfdiffusion/inference/../../examples/input_pdbs/1qys.pdb
```

**Fix command:**
```bash
# Backup original file
docker exec oih-rfdiffusion bash -c "cp \
  /usr/local/lib/python3.10/dist-packages/rfdiffusion/inference/model_runners.py \
  /usr/local/lib/python3.10/dist-packages/rfdiffusion/inference/model_runners.py.bak"

# Fix path
docker exec oih-rfdiffusion bash -c "sed -i \
  's|../../examples/input_pdbs/1qys.pdb|/app/RFdiffusion/examples/input_pdbs/1qys.pdb|' \
  /usr/local/lib/python3.10/dist-packages/rfdiffusion/inference/model_runners.py"

# Verify
docker exec oih-rfdiffusion bash -c "grep '1qys' \
  /usr/local/lib/python3.10/dist-packages/rfdiffusion/inference/model_runners.py"
```

## Warning: Model Path Issue (fixed in docker-compose.yml)
- Actual model location (host): /data/rfdiffusion/models/
- Container mount path: /data/models/rfdiffusion
- Correct docker-compose.yml configuration:
  ```yaml
  volumes:
    - /data/rfdiffusion/models:/data/models/rfdiffusion:ro
  ```

## Warning: GPU Notes
- NVIDIA_VISIBLE_DEVICES=1 maps host GPU1 to container GPU0
- RFdiffusion automatically uses CUDA device 0, no manual specification needed
- VRAM requirement approximately 8GB

## Available Model Files (/data/rfdiffusion/models/)
| File | Purpose |
|------|---------|
| Base_ckpt.pt | Basic protein backbone design (most commonly used) |
| Base_epoch8_ckpt.pt | Base model early checkpoint |
| Complex_base_ckpt.pt | Protein-protein complex/binder design |
| Complex_Fold_base_ckpt.pt | Complex + fold conditioning |
| Complex_beta_ckpt.pt | Complex beta version |
| ActiveSite_ckpt.pt | Active site design |
| InpaintSeq_ckpt.pt | Sequence inpainting |
| InpaintSeq_Fold_ckpt.pt | Sequence inpainting + fold |
| RF_structure_prediction_weights.pt | Structure prediction weights (internal use) |

## Example Script Location
/app/RFdiffusion/examples/ contains complete official examples, consult before use:
```bash
docker exec oih-rfdiffusion bash -c "ls /app/RFdiffusion/examples/"
docker exec oih-rfdiffusion bash -c "cat /app/RFdiffusion/examples/design_ppi.sh"
```

## Complete Inference Commands

### 1. Unconditional Design (de novo design from scratch) - Verified
```bash
docker exec oih-rfdiffusion python3 /app/RFdiffusion/scripts/run_inference.py \
  inference.output_prefix=/data/oih/outputs/rfdiffusion/unconditional \
  inference.model_directory_path=/data/models/rfdiffusion \
  'contigmap.contigs=[100-200]' \
  inference.num_designs=10
```
Note: inference.input_pdb not needed; after bug fix, automatically uses default 1qys.pdb

### 2. Motif Scaffolding (design scaffold around fixed active site)
```bash
docker exec oih-rfdiffusion python3 /app/RFdiffusion/scripts/run_inference.py \
  inference.output_prefix=/data/oih/outputs/rfdiffusion/motif \
  inference.model_directory_path=/data/models/rfdiffusion \
  inference.input_pdb=/data/oih/inputs/5TPN.pdb \
  'contigmap.contigs=[10-40/A163-181/10-40]' \
  inference.num_designs=10
```

### 3. Binder Design (protein-protein interaction)
```bash
docker exec oih-rfdiffusion python3 /app/RFdiffusion/scripts/run_inference.py \
  inference.output_prefix=/data/oih/outputs/rfdiffusion/binder \
  inference.model_directory_path=/data/models/rfdiffusion \
  inference.input_pdb=/data/oih/inputs/target.pdb \
  'contigmap.contigs=[A1-150/0 70-100]' \
  'ppi.hotspot_res=[A59,A83,A91]' \
  denoiser.noise_scale_ca=0 \
  denoiser.noise_scale_frame=0 \
  inference.num_designs=10
```

### 4. Partial Diffusion (local diffusion on existing structure)
```bash
docker exec oih-rfdiffusion python3 /app/RFdiffusion/scripts/run_inference.py \
  inference.output_prefix=/data/oih/outputs/rfdiffusion/partial \
  inference.model_directory_path=/data/models/rfdiffusion \
  inference.input_pdb=/data/oih/inputs/input.pdb \
  'contigmap.contigs=[A1-100]' \
  diffuser.partial_T=10 \
  inference.num_designs=10
```

## Contig Syntax Reference (extracted from official examples)
| Syntax | Meaning |
|--------|---------|
| `[100-200]` | Design new protein of 100-200 residues (random length) |
| `[100]` | ERROR: parsed as integer, must use 100-100 |
| `[A163-181]` | Fix chain A residues 163-181 (motif) |
| `[10-40/A163-181/10-40]` | 10-40 new design + fixed motif + 10-40 new design |
| `[A1-150/0 70-100]` | Chain A 1-150 + chain break (0) + 70-100 new binder |
| `ppi.hotspot_res=[A59,A83,A91]` | Specify hotspot residues the binder must contact (recommend 3-6) |

## Performance (RTX 4090 45GB)
- Unconditional 100-200 residues: approximately 0.40 min/design
- Recommend num_designs=10 for batch generation, then select best

## Integration with ProteinMPNN
RFdiffusion only generates backbones (no sequence), must pair with ProteinMPNN for sequence design:
```
RFdiffusion -> backbone.pdb -> ProteinMPNN -> sequences.fa -> AlphaFold3 validation
```

## PDB Preprocessing (automatic)
Router automatically preprocesses target protein PDB in binder_design mode:
1. **Remove HETATM**: water/ligands/glycosylation modifications interfere with ContigMap
2. **Sequential renumbering**: eliminate residue gaps, e.g., PDB 2A91 original numbering 102->107 jump, renumbered to 102->103 continuous
3. **Hotspot mapping**: hotspot residue numbers automatically converted from original to new numbering
4. Output file: `{input}_renum.pdb`

### Hotspot Format
- User/agent can input: `S310,F311` (amino acid name + number) or `310,311` (number only)
- Router automatically normalizes to RFdiffusion format: `[A306,A307]` (chain ID + renumbered number)

### Why Renumbering Is Needed
RFdiffusion ContigMap iterates `A1-{max_resid}`, checking each residue exists in the PDB.
If PDB has gaps (e.g., crystallographically missing residues), ContigMap throws:
`AssertionError: ('A', 103) is not in pdb file!`

## Troubleshooting
| Error | Cause | Solution |
|-------|-------|----------|
| python: command not found | Container only has python3 | Use python3 |
| FileNotFoundError: 1qys.pdb | model_runners.py path bug | Execute bug fix command above |
| AttributeError: 'int' has no strip | contig written as [50] integer | Change to [50-50] or [100-200] |
| AssertionError: ('A', N) not in pdb | PDB has residue gaps | Auto-renumbering fixed; if still occurs check HETATM |
| Model loading failure | Mount path incorrect | Check docker-compose.yml mount |
| CUDA OOM | Insufficient VRAM | Reduce num_designs or shorten contig length |

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## Warning: Notes (auto-synced from CLAUDE.md)

- Container NVIDIA_VISIBLE_DEVICES=1, always use device=0 / gpu_id=0 (do not use 1)
- **timeout must be 7200s** (not 3600s)
- **num_designs use 10** (not 20, too slow)
- **hotspots must be spatially clustered** (<= 15A centroid), dispersed ones will be extremely slow

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->
