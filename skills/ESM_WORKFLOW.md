# ESM Workflow Documentation

## Basic Information
- Container: oih-esm
- Model: ESM2-650M (esm2_t33_650M_UR50D)
- Model path: /root/.cache/torch/hub/checkpoints/
- GPU: NVIDIA_VISIBLE_DEVICES=1 → cuda:0 inside the container

## Main Functions
- ESM2: protein sequence embedding extraction (for downstream structure/function prediction tasks)
- ESM-1v: zero-shot variant effect prediction
- ESM-IF1: inverse folding (design sequences given a structure)

## Embedding Extraction (Python API)
```python
import torch
import esm

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval().to('cuda:0')

data = [("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQL...")]
batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_tokens = batch_tokens.to('cuda:0')

with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=False)

embeddings = results['representations'][33]
# shape: [batch, seq_len, 1280]
```

## Output Description
- repr_layers=[33]: last layer (layer 33), 1280 dimensions
- embeddings[:,0,:]: [CLS] token, represents the entire sequence
- embeddings[:,1:-1,:]: per-residue embeddings for each amino acid

## Command-line Batch Extraction (FASTA)
```bash
docker exec oih-esm bash -c "
python3 /app/esm/scripts/extract.py \
  esm2_t33_650M_UR50D \
  /data/oih/inputs/sequences.fasta \
  /data/oih/outputs/esm_embeddings/ \
  --repr_layers 33 \
  --include mean per_tok"
```

## Notes
- Model is cached; no re-download needed
- Long sequences (>1000aa) require significant VRAM; batch_size=1 recommended
- VRAM estimate: 650M model ~6GB (GPU1 45GB is sufficient)
- ESMFold requires python<=3.9, not supported in the current environment; use AlphaFold3 for structure prediction instead

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## Notes (auto-synced from CLAUDE.md)

- Inside the container NVIDIA_VISIBLE_DEVICES=1; always use device=0 / gpu_id=0 (do not use 1)

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->
