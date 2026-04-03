# IgFold Antibody Structure Prediction Workflow

## Basic Information
- Container: oih-igfold (based on proteinmpnn:latest)
- Software: igfold 0.4.0 + antiberty + abnumber
- GPU: NVIDIA_VISIBLE_DEVICES=1 → cuda:0 inside the container
- Queue: GPU (VRAM ~4GB)
- Timeout: 120s per sequence
- Model: bundled in pip package, frozen via docker commit

## Functionality
Rapidly predicts 3D structures from antibody/nanobody sequences (~2-4s per sequence on GPU).
Primarily used as a pre-screening funnel between ProteinMPNN and AF3.

## Parameters
- `job_name`: task identifier
- `sequences`: chain sequence dictionary, e.g. `{"H": "EVQLVE..."}` (nanobodies only need the H chain)
- `do_refine`: OpenMM structure refinement (slower but better geometry, only for final candidates)
- `do_renum`: ANARCI CDR numbering (currently disabled, anarci package name conflict pending fix)

## Output
- `output_pdb`: predicted structure PDB file path
- `mean_plddt`: average pLDDT score (0-100, higher is better)
- `mean_prmsd`: average predicted RMSD (angstrom, lower is better)

## Upstream/Downstream Dependencies
- **Upstream**: proteinmpnn_sequence_design → sequences
- **Downstream**: alphafold3_predict (complex validation only for candidates with IgFold pLDDT > 70)

## Role in Pipeline (Step 10b/16)
```
ProteinMPNN (50-100 sequences) → ESM2 PPL filter (PPL < 15) → 50
→ IgFold pLDDT filter (pLDDT > 70) → 10-15 → AF3 validation (ipTM ≥ 0.6)
```

## pLDDT Quality Expectations (by Target Tier)
- **Tier 1 hotspots** (known complex interface residues) → pLDDT > 70 expected, high pass rate
- **Tier 2 hotspots** (computationally predicted epitopes) → pLDDT 40-70 common, requires more design candidates

## Warning: IgFold Only Applies to Antibody/Nanobody Sequences
- IgFold internally uses AntiBERTy, trained only on antibody sequences
- **de novo binders (RFdiffusion designs) are not antibodies → pLDDT is meaningless** (observed pLDDT 0.4-12.6)
- In the pipeline, IgFold is automatically skipped when `binder_type='de_novo'` (default)
- IgFold filtering is only enabled when `binder_type='nanobody'` or `'antibody'`
- de novo binders go through ESM2 PPL filter → directly to AF3 (no structure pre-screening)

## Notes
- `do_renum=False` (ANARCI package name conflict anarci vs anarcii, to be fixed later)
- Models are not lost on container restart (frozen via docker commit + /root/.cache volume mount)
- For batch calls, use a separate task per sequence (to avoid OOM)
- `mean_plddt` may return None → use `or 0` as a safeguard in the pipeline
