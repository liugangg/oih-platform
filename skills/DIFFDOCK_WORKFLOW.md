# DiffDock Complete Workflow

## Environment Information
- Container name: oih-diffdock
- Image: diffdock:latest
- Python: python3 (3.10.12), runs as root
- Inference script: /app/DiffDock/inference.py
- Working directory: /app/DiffDock/

## ⚠️ Important: This is NOT the official rbgcsail/diffdock image
- The official image uses micromamba + conda environment diffdock (torch 1.13.1+cu117)
- This image uses system Python (3.10.12) + pip (torch 2.1.0+cu121)
- Therefore do not use micromamba run -n diffdock; use python3 directly

## ⚠️ Dependency Issues (must reinstall after container rebuild)
The following packages are lost after container restart and must be reinstalled:
```bash
# torch_geometric (installed but needs verification after restart)
docker exec oih-diffdock bash -c "pip install torch_geometric 2>&1 | tail -3"

# fair-esm
docker exec oih-diffdock bash -c "pip install fair-esm 2>&1 | tail -3"

# torch geometric extensions (must match torch 2.1.0+cu121)
docker exec oih-diffdock bash -c "pip install torch-cluster torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html 2>&1 | tail -5"
```

**Permanent fix**: Add these dependencies to the Dockerfile and rebuild the image:
```dockerfile
# Add to /data/docking/Dockerfile.diffdock
RUN pip install torch_geometric fair-esm && \
    pip install torch-cluster torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

## Model Files
- Model location (inside container): /app/DiffDock/workdir/v1.1/
- score_model: /app/DiffDock/workdir/v1.1/score_model/best_ema_inference_epoch_model.pt
- confidence_model: /app/DiffDock/workdir/v1.1/confidence_model/best_model_epoch75.pt
- Source: HuggingFace reginabarzilaygroup/DiffDock-L

### Model Download Command
```bash
docker exec oih-diffdock bash -c "
cd /app/DiffDock/workdir/v1.1 &&
mkdir -p score_model confidence_model &&
pip install huggingface_hub -q &&
python3 -c \"
from huggingface_hub import hf_hub_download
import shutil
f = hf_hub_download(repo_id='reginabarzilaygroup/DiffDock-L', filename='score_model/best_ema_inference_epoch_model.pt')
shutil.copy(f, 'score_model/')
f = hf_hub_download(repo_id='reginabarzilaygroup/DiffDock-L', filename='confidence_model/best_model_epoch75.pt')
shutil.copy(f, 'confidence_model/')
print('Download complete')
\"
"
```

## ⚠️ GPU Notes
- NVIDIA_VISIBLE_DEVICES=1 maps host GPU1 as GPU0 inside the container
- DiffDock auto-detects CUDA; no need to specify device manually
- VRAM requirement: ~8GB

## Inference Commands

### 1. Single Protein-Ligand Docking
```bash
docker exec oih-diffdock python3 /app/DiffDock/inference.py \
  --protein_path /data/oih/inputs/1IVO.pdb \
  --ligand_description "COc1ccc2c(c1)nc(N)c(C(=O)Nc1ccc(F)cc1)c2" \
  --out_dir /data/oih/outputs/diffdock/ \
  --samples_per_complex 10 \
  --inference_steps 20 \
  --batch_size 4
```

### 2. Batch Docking (CSV Input)
```bash
# CSV format: protein_path,ligand_description,complex_name
cat > /data/oih/inputs/diffdock_input.csv << 'CSV'
protein_path,ligand_description,complex_name
/data/oih/inputs/1IVO.pdb,COc1ccc2c(c1)nc(N)c(C(=O)Nc1ccc(F)cc1)c2,complex_1
CSV

docker exec oih-diffdock python3 /app/DiffDock/inference.py \
  --protein_ligand_csv /data/oih/inputs/diffdock_input.csv \
  --out_dir /data/oih/outputs/diffdock/ \
  --samples_per_complex 10 \
  --inference_steps 20 \
  --batch_size 4
```

### 3. Using Default Config File (Recommended)
```bash
docker exec oih-diffdock python3 /app/DiffDock/inference.py \
  --config /app/DiffDock/default_inference_args.yaml \
  --protein_path /data/oih/inputs/1IVO.pdb \
  --ligand_description "SMILES_STRING" \
  --out_dir /data/oih/outputs/diffdock/
```

## Key Parameters
| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| --samples_per_complex | Number of poses to generate | 10~40 |
| --inference_steps | Number of diffusion steps | 20 |
| --batch_size | Batch size; reduce if VRAM insufficient | 4 |
| --no_final_step_noise | No noise at last step; improves quality | Include this flag |

## Confidence Score Interpretation
- c > 0: High confidence
- -1.5 < c < 0: Medium confidence
- c < -1.5: Low confidence

## Output File Structure
```
/data/oih/outputs/diffdock/
└── complex_name/
    ├── rank1_confidence0.xx.sdf    # Best pose
    ├── rank2_confidence0.xx.sdf
    └── ...
```

## Notes
- DiffDock is for small molecule-protein docking only; not suitable for protein-protein docking
- Ligands are input as SMILES strings
- Recommended to use with GNINA: DiffDock generates poses -> GNINA rescores/optimizes
- First run precomputes SO(2)/SO(3) distributions (~1-2 minutes); cached afterward

## Troubleshooting
| Error | Cause | Solution |
|-------|-------|----------|
| ModuleNotFoundError: torch_geometric | Dependency not installed | Run dependency install commands above |
| ModuleNotFoundError: esm | fair-esm not installed | pip install fair-esm |
| ModuleNotFoundError: torch_cluster | torch extension version mismatch | Install with cu121 version |
| FileNotFoundError: best_ema_inference_epoch_model.pt | Model not downloaded | Run model download command |
| micromamba: command not found | This is not the official image | Use python3 directly; do not use micromamba |

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## ⚠️ Notes (Auto-synced from CLAUDE.md)

- NVIDIA_VISIBLE_DEVICES=1 inside container; always use device=0 / gpu_id=0 (not 1)

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->
