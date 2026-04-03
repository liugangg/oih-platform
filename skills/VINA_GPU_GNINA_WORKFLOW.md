# Vina-GPU 2.1 & GNINA Docking Workflow (Verified)

## Tool Information
- Vina-GPU container: oih-vina-gpu, command: vina_gpu (AutoDockVina-GPU 2.1)
- GNINA container: oih-gnina, command: gnina (deep learning CNN scoring)
- GPU: NVIDIA_VISIBLE_DEVICES=1 → device 0 inside the container

## When to Use Vina-GPU vs GNINA vs AutoDock-GPU
| Tool | Use Case |
|------|------|
| Vina-GPU 2.1 | Fast screening, known pocket coordinates, standard docking |
| GNINA | Best general-purpose, CNN rescoring, higher accuracy, outputs SDF |
| AutoDock-GPU | Precise free energy calculations, large-scale virtual screening |

## Vina-GPU Command
```bash
docker exec oih-vina-gpu vina_gpu \
  --receptor /data/oih/inputs/<job>/receptor.pdbqt \
  --ligand <lig.pdbqt> \
  --out /data/oih/outputs/<job>/vina_gpu/out.pdbqt \
  --center_x <x> --center_y <y> --center_z <z> \
  --size_x 25 --size_y 25 --size_z 25 \
  --num_modes 9 \
  --exhaustiveness 8 \
  --thread 8000
```
Output format: PDBQT; parse affinity lines

## GNINA Command
```bash
docker exec oih-gnina gnina \
  --receptor /data/oih/inputs/<receptor>.pdb \
  --ligand <lig.pdbqt> \
  --out /data/oih/outputs/<job>/gnina/poses.sdf \
  --center_x <x> --center_y <y> --center_z <z> \
  --size_x 25 --size_y 25 --size_z 25 \
  --num_modes 9 \
  --exhaustiveness 8 \
  --autobox_add 4 \
  --device 0
```
Output format: SDF; parse CNNscore and affinity

## Ligand Preparation (same for both)
```bash
# SMILES to pdbqt (obabel inside container)
obabel -:"<SMILES>" --gen3d -O ligand.pdbqt -h
```

## Notes
- Vina-GPU: receptor must be pre-converted to pdbqt format
- GNINA: receptor can use PDB directly, no conversion needed
- GNINA SDF output is easier to parse and visualize than PDBQT
- smart_dock routing: with coordinates → Vina-GPU; without coordinates → DiffDock blind docking

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## Notes (auto-synced from CLAUDE.md)

- Inside the container NVIDIA_VISIBLE_DEVICES=1; always use device=0 / gpu_id=0 (do not use 1)

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->
