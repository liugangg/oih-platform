# BindCraft Workflow Documentation

## Basic Information
- Container: oih-bindcraft
- Function: AF2 backpropagation + MPNN + PyRosetta de novo protein binder design
- GPU: NVIDIA_VISIBLE_DEVICES=1 → cuda:0 inside the container
- VRAM requirement: ~16GB
- Do not mount host cuda lib64 (JAX bundles its own CUDA; mounting will cause conflicts)

## Target Configuration File (JSON)
```json
{
    "design_path": "/data/oih/outputs/<task>/",
    "binder_name": "<name>",
    "starting_pdb": "/data/oih/inputs/<target>.pdb",
    "chains": "A",
    "target_hotspot_residues": null,
    "lengths": [50, 80],
    "number_of_final_designs": 1
}
```

## Run Command
```bash
docker exec oih-bindcraft bash -c "
cd /app/BindCraft &&
python3 -u bindcraft.py \
  --settings '/data/oih/inputs/bindcraft_target.json' \
  --filters './settings_filters/default_filters.json' \
  --advanced './settings_advanced/default_4stage_multimer.json'
"
```

## Notes
- Set hotspot_residues to null to let AF2 automatically select binding sites
- At least 100 final designs recommended for experimental screening
- Each trajectory takes approximately several minutes; difficult targets may require thousands of attempts
- Output: PDB structures + CSV statistics (ipTM, pLDDT, etc.)
- ipTM is a binary indicator for binding prediction and does not directly reflect affinity

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## Notes (auto-synced from CLAUDE.md)

- Inside the container NVIDIA_VISIBLE_DEVICES=1; always use device=0 / gpu_id=0 (do not use 1)

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->
