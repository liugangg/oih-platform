# Chemprop Workflow Documentation

## Basic Info
- Container: oih-chemprop
- Python: 3.11 (container: /usr/bin/python3.11, symlink /usr/bin/python3 → python3.11)
- Version: chemprop 2.2.2
- **Inference uses CPU** (--accelerator cpu), GPU acceleration mainly for training large datasets
- Queue: CPU queue (chemprop/chemprop_predict assigned to _CPU_TOOLS in task_manager)

## CLI Commands
chemprop {train, predict, convert, fingerprint, hpopt}

## Training (GPU-accelerated, recommended for large datasets)
```bash
docker exec oih-chemprop bash -c "
chemprop train \
  -i /data/oih/inputs/train.csv \
  --smiles-columns smiles \
  --target-columns <target_column_name> \
  --accelerator gpu \
  --devices 1 \
  --epochs 50 \
  --save-dir /data/oih/outputs/<task>/"
```

## Prediction (CPU mode, avoids GPU OOM)
```bash
docker exec oih-chemprop bash -c "
chemprop predict \
  -i /data/oih/inputs/test.csv \
  --model-paths /data/oih/outputs/<task>/model_0/best.pt \
  --accelerator cpu \
  -o /data/oih/outputs/<task>/predictions.csv"
```

## Important Notes
- During training, --devices 1 means "use 1 GPU" (not device index)
- For prediction, use --accelerator cpu; CPU is sufficient for small molecule inference (3 molecules <5s)
- Container has NVIDIA_VISIBLE_DEVICES=1, host GPU1 mapped as GPU0 inside container
- Model save location: <save-dir>/model_0/best.pt
- Recommend adding --num-workers 8 to speed up data loading
- Test model inside container: /opt/chemprop/tests/data/example_model_v2_regression_mol.pt

## ADC ADMET Evaluation Usage

chemprop predict can be used to evaluate ADMET properties of ADC payload/linker-payload conjugates.

**Upstream**: `adc_smiles` field from rdkit_conjugate output (both dot-disconnected and covalent SMILES are accepted)

**API Call**:
```bash
curl -s --noproxy '*' -X POST http://localhost:8080/api/v1/ml/chemprop/predict \
  -H "Content-Type: application/json" \
  -d '{
    "job_name": "adc_admet",
    "smiles": ["<adc_smiles_from_rdkit_conjugate>"],
    "model_path": "/path/to/admet_model/best.pt",
    "task": "predict"
  }'
```

**Output**: Predicted values per molecule (toxicity/solubility/permeability etc., depends on trained model)

**Typical ADC Workflow**:
1. `fetch_molecule` → Get payload SMILES (e.g. MMAE)
2. `linker_select` → Select linker (e.g. MC-VC-PABC)
3. `rdkit_conjugate` → Generate linker-payload conjugate adc_smiles
4. `chemprop_predict` → Run ADMET prediction on adc_smiles

## Typical Applications
- Solubility/logS prediction
- ADMET property prediction (Absorption/Distribution/Metabolism/Excretion/Toxicity)
- Bioactivity prediction (IC50, Ki, etc.)
- Antibiotic activity screening (ref: Cell 2020 Halicin)
- ADC payload toxicity/selectivity evaluation

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## Notes (Auto-synced from CLAUDE.md)

- 1. **Dockerfile must include**: `RUN ln -sf /usr/bin/python3.11 /usr/bin/python3` (container's python3 defaults to 3.10, but all packages are installed under the 3.11 path)
- 2. **predict calls must include**: `--accelerator cpu` (small batches on GPU will OOM), assigned to `_CPU_TOOLS` queue
- 3. **`--devices 1` = "use 1 GPU device"**, not device index=1. Training: `--accelerator gpu --devices 1`, Prediction: `--accelerator cpu`
- **Root cause**: Container's `python3` symlink points to 3.10, but all packages (torch/numpy/chemprop) are installed under python3.11 path → `No module named 'numpy'`
- **Fix**: `docker exec oih-chemprop bash -c "rm /usr/bin/python3 && ln -s /usr/bin/python3.11 /usr/bin/python3"` (lost on container rebuild, must be in Dockerfile)
- **Router fix**: `routers/ml_tools.py` adds `--accelerator cpu` to avoid GPU OOM; `task_manager.py` adds `chemprop_predict` to `_CPU_TOOLS`
- **Verified**: 3 molecule prediction completed, CPU queue, 5 seconds

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->
