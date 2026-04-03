# Fpocket & P2Rank Binding Pocket Detection Workflow (Verified)

## Tool Information
- fpocket container: oih-fpocket
- p2rank container: oih-p2rank
- Purpose: detect protein binding site coordinates prior to molecular docking

## When to Use fpocket vs p2rank
- fpocket: general proteins, fast, returns drug_score + coordinates
- p2rank: recommended alphafold mode for AlphaFold-predicted structures, higher accuracy

## fpocket Invocation Command
```bash
docker exec oih-fpocket fpocket \
  -f /data/oih/outputs/<job>/<input>.pdb \
  -m <min_sphere_size>
```
Output directory: `/data/oih/outputs/<job>/<pdb_stem>_out/`
File to parse: `**/*_info.txt`
Output fields: pocket_id, drug_score, volume, x_centroid, y_centroid, z_centroid

## p2rank Invocation Command
```bash
docker exec oih-p2rank bash -c "
/app/p2rank/prank predict [-c alphafold] \
  -f /data/oih/inputs/<input>.pdb \
  -o /data/oih/outputs/<job>/p2rank
"
```
Model parameter: default | alphafold | conservation
Output file: `*_predictions.csv`
Output fields: name, rank, score, probability, center_x, center_y, center_z

## Notes
- fpocket: copy input_pdb from inputs/ to outputs/ before running
- p2rank: reads input_pdb directly from inputs/, writes output to outputs/
- CSV field names have leading spaces; strip when parsing

## Workflow: Pocket Detection to Docking
1. Call fpocket or p2rank to obtain the pocket list
2. Use pockets[0] center_x/y/z as the docking box center
3. box_size defaults to 25x25x25 angstrom
4. Pass center_x/y/z parameters to dock_ligand

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## Notes (auto-synced from CLAUDE.md)

- **Legacy workflow** (11 steps): P2Rank top pocket → directly take top 6 residues → DiffDock blind docking cross-validation → RFdiffusion
- 10. AF3 validation — ipTM >= 0.6

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->
