# PTM Upload Paradigm — Qwen Decision Framework

## When receiving a message tagged with [User Uploaded File]

### Rule 1: Prefer pre-generated input files; do not rebuild them
- If af3_input.json path is present → pass directly to the alphafold3 tool; do not rewrite the JSON
- If gromacs_ptm_notes.json is present → read the force_field and disulfide_pairs parameters from it
- If adc_input.json is present → pass to freesasa; conjugation_sites are already preset

### Rule 2: PTM-to-tool mapping
| PTM Type | Available Tools | Unavailable Tools | Notes |
|--------|---------|---------|---------|
| Glycosylation (NAG/FUC) | AlphaFold3 | GROMACS | GROMACS lacks GLYCAM force field; use AF3 prediction only |
| Phosphorylation (SEP/TPO) | AlphaFold3 | GROMACS (limited) | GROMACS requires CHARMM36; inform user that accuracy is limited |
| Disulfide bond (SSBOND) | AF3+GROMACS+RFdiffusion | — | Add -ss flag for GROMACS, add --disulfide for RFdiffusion |
| Cys conjugation site | freesasa+rdkit_conjugate | — | First confirm accessibility with freesasa, then run rdkit_conjugate |
| Unsupported PTM | — | All tools | Explicitly inform the user that this PTM is not currently supported by the platform |

### Rule 3: FASTA sequence upload
- No structure available → first call alphafold3 to predict structure, then proceed with downstream analysis
- Structure exists but mutations need to be designed → call proteinmpnn or rfdiffusion

### Rule 4: SMILES upload
- Small molecule ligand → first call chemprop for ADMET prediction, then call gnina/autodock for docking
- ADC payload → call rdkit_conjugate; pass SMILES directly

### Rule 5: When uncertain
- Unrecognized PTM type → report to user, explain the platform does not support it, suggest removing it and re-uploading
- File parsing failure → report the specific error; do not guess

### Typical Workflow Examples

**User uploads glycosylated antibody PDB + "predict binding to CD36"**
1. Read af3_input.json (already contains glycosylation ligands)
2. Call alphafold3(input_json=af3_input.json path)
3. Use AF3 output structure to call fpocket for binding pocket detection
4. Call gnina for docking
5. Report binding energy + ipTM

**User pastes FASTA + "design nanobody"**
1. Call alphafold3 to predict target structure (FASTA → structure)
2. Call rfdiffusion(target_pdb=predicted structure, mode=binder_design)
3. Call proteinmpnn to optimize sequences
4. Call alphafold3 to validate complex ipTM
