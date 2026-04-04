# DiscoTope3 B-Cell Epitope Prediction Workflow

## Basic Information
- Container: oih-discotope3
- Software path: /app/discotope3/discotope3/main.py
- GPU: NVIDIA_VISIBLE_DEVICES=1 -> cuda:0 inside container
- Model: XGBoost ensemble + ESM embeddings
- Queue: GPU (VRAM ~6GB)
- Timeout: 600 seconds

## Function
Predicts B-cell epitope propensity on protein structures (which residues are most likely to be recognized by antibodies).

## Parameters
- `input_pdb`: protein PDB file path (antigen or antibody structure)
- `struc_type`: `solved` (experimental structure) or `alphafold` (AF2/AF3 predicted structure)
- `calibrated_score_epi_threshold`: epitope threshold, low=0.40 (sensitive) / moderate=0.90 (default) / high=1.50 (strict)
- `multichain_mode`: predict all chains in the entire complex (default false)
- `cpu_only`: force CPU inference (default false, uses GPU)

## Output
- CSV file: per-residue epitope propensity score (calibrated score)
- PDB file: high-scoring residues annotated

## Upstream/Downstream Relationships
- **Upstream**: fetch_pdb -> input_pdb | alphafold3_predict -> file after CIF->PDB conversion
- **Downstream**:
  - High-scoring epitope residues -> hotspot_residues parameter for rfdiffusion_design
  - Epitope information -> design reference for binder_design_pipeline
  - Cross-validation with fpocket/P2Rank pocket results

## Use Cases
1. **Antibody target analysis**: predict which regions on the antigen surface can be targeted by antibodies
2. **Vaccine antigen design**: select highly immunogenic epitopes for vaccines
3. **Binder design guidance**: use epitope residues as RFdiffusion hotspots
4. **AF3 validation**: compare overlap between AF3-predicted interface and DiscoTope3-predicted epitopes

## Notes
- CLI must be run from `/app/discotope3/discotope3/` directory (bare imports)
- Use `python3` inside container (not `python`)
- AlphaFold structures must set `struc_type=alphafold` (affects B-factor handling)
- Large complexes (>1000 residues) automatically switch to CPU embedding

## DT3 Score Distribution (hard-won lessons)
- `DiscoTope-3.0_score` raw score range is typically **0.001 ~ 0.5** (varies significantly between structures)
- `calibrated_score` has a wider range after calibration (up to 6+), but the pipeline uses raw scores
- **Never use a fixed threshold of 0.7** -- most structures' highest scores do not reach 0.5
- Correct approach: **adaptive threshold = max(scores[top_20%], 0.10)**

| Structure | Residue count | Highest DT3 score | top-20% threshold | Residues >= threshold |
|------|--------|-----------|-------------|-------------|
| 2A91 chain A (HER2 ECD) | 506 | 0.361 | 0.142 | 102 |
| 1N8Z chain C (HER2+Ab) | 581 | 0.513 | ~0.15 | ~95 |

## known_epitope_override Key Parameters (2026-03-21 validated)

### Threshold Settings
- Do not use a fixed threshold (e.g., 0.7) -- DT3 score ranges vary greatly between structures
- Correct approach: `adaptive threshold = max(top 20% scores, 0.10)`
- 2A91 measured: max score=0.361, fixed 0.7 -> 0 high-confidence residues, override never triggers

### Numbering Alignment (PDB vs UniProt)
- Residue numbers returned by IEDB/RAG may be **UniProt numbering**, offset from PDB numbering
- 2A91: PDB numbering has a **+22 offset** from UniProt numbering (PDB res 245 ~ UniProt res 267)
- Solution: try multiple common offsets `[0, +/-22, +/-23]`, select the one with most matches
- Each comparison allows **+/-3 tolerance** (`abs(dt3_num + offset - rag_num) <= 3`)

### Override Trigger Conditions
- `overlap >= 2` residues (not 3, to avoid missed detections)
- Once triggered, directly use overlap residues as hotspot, skip 6D scoring (steps 4-6)
- If not triggered, proceed normally with 6D composite scoring + agent selection

### 2A91 HER2 Validation Results
- DT3 adaptive threshold: **0.142**（top 20%）
- offset=0: 7 matches (direct match)
- offset=+22: **18 matches** (best, PDB->UniProt offset)
- Final hotspots: `A245, A246, A157, A258, A256, A286, A255, A260`
- Known hits: A245/A246 (Domain I/III interface), A286 (Domain II proximal)

### 6D Scoring Formula (used when override is not triggered)
```
composite = p2rank(0.15) + sasa(0.15) + conservation(0.15) + rag(0.25) + electrostatics(0.10) + epitope(0.20)
```
epitope dimension: proportion of pocket residues within 8A of high-scoring DT3 residues (_compute_epitope_score_for_pocket)

## Target Tier Classification (2026-03-22)
- **Tier 1 (Known complex)**: Use `extract_interface_residues` → most reliable, experimentally validated
  - HER2 = Tier 1, use 1N8Z (trastuzumab complex) → skip DiscoTope3
- **Tier 2 (Homolog)**: Use homology transfer → moderate confidence
- **Tier 2 (Novel)**: Use DiscoTope3 + IEDB + RAG → computational prediction
  - DiscoTope3 is only used for Tier 2 targets (novel, no known complex)
  - For Tier 1, interface extraction from known complex is far more reliable

## Scope Limitations

DiscoTope3 predicts B-cell epitopes on antigen surfaces and is only applicable to antibody design scenarios.

### Valid Use Cases
- Target is a protein antigen -> use DiscoTope3
- Designing antibody/nanobody/binder -> use DiscoTope3
- No known complex structure (Tier 2) -> use DiscoTope3

### Prohibited Use Cases
- Small molecule drug docking -> use fpocket/P2Rank
- Co-crystal structure available (Tier 1) -> use extract_interface_residues directly
- Enzyme active site prediction -> use fpocket + conservation analysis
