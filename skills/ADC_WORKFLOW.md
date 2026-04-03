# ADC Drug Design Workflow

## Overview
ADC (Antibody-Drug Conjugate) = Antibody (targeting) + Linker (connector) + Payload (cytotoxin)
The platform supports end-to-end design from target PDB to complete ADC small molecule structure.

## Tool Chain

### 1. freesasa — Antibody Conjugation Site Prediction
- Route: /api/v1/design/freesasa
- Queue: CPU
- Input: antibody PDB path (from rfdiffusion/proteinmpnn output)
- Output: exposed Lys/Cys list with SASA values (>40A^2 recommended as conjugation sites)
- Upstream: rfdiffusion -> proteinmpnn -> antibody PDB after af3 validation
- Downstream: conjugation_site parameter for rdkit_conjugate

### 2. linker_select — Linker Selection
- Route: /api/v1/adc/linker_select
- Queue: synchronous return
- Input: payload_smiles, cleavable(bool), reaction_type, compatible_payload, clinical_status
- Output: recommended linker list (with SMILES/DAR/approved_adcs/stability_note)
- Library file: /data/oih/oih-api/data/linker_library.json (20 clinically validated linkers)
- Default priority: approved > clinical > research
- Upstream: fetch_molecule to obtain payload_smiles
- Downstream: linker_smiles/linker_name parameters for rdkit_conjugate

### 3. rdkit_conjugate — Covalent Conjugation
- Route: /api/v1/adc/rdkit_conjugate
- Queue: CPU
- Input: job_name, antibody_pdb, conjugation_site, linker_smiles, linker_name, payload_smiles, reaction_type
- Output: adc_smiles, covalent(bool), reaction_type_used, dar_range, output_sdf, atom_count
- reaction_type="auto" enables automatic detection
- covalent=false falls back to dot-disconnected (still usable for ADMET)
- Upstream: freesasa(conjugation_site) + linker_select(linker_smiles) + fetch_molecule(payload_smiles)
- Downstream: chemprop ADMET prediction + gnina off-target assessment

## 7 Types of Conjugation Reaction Chemistry
| Reaction Type | Linker End | Payload End | Representative Linker | Representative ADC |
|--------------|------------|-------------|----------------------|-------------------|
| maleimide_thiol | Maleimide | Thiol/Cys | MC-VC-PABC, SMCC | Adcetris, Kadcyla |
| nhs_amine | NHS ester | Lysine/amine | VC-PABC | Most lysine-conjugated ADCs |
| hydrazone | Hydrazide | Ketone | Hydrazone, AcBut | Mylotarg |
| oxime | Hydroxylamine | Aldehyde/ketone | aminooxy-PEG | Investigational |
| disulfide | Pyridyldithiol | Thiol | SPDP, CL2 | DM4-class ADCs |
| dbco_azide | DBCO/alkyne | Azide | DBCO-PEG4 | Investigational site-specific |
| transglutaminase | Enzymatic Q295 | Amine | TGase-cadaverine | Investigational |

## Complete ADC Design Workflow — binder_design_pipeline (7 steps in one)

**Recommended: directly call binder_design_pipeline, supports pdb_id parameter for automatic PDB download**

```
Step 0: fetch_pdb(pdb_id="1N8Z") -> automatically download target protein PDB
Step 1: RFdiffusion -> generate protein backbone (binder_design mode)
Step 2: ProteinMPNN -> design sequences for each backbone (8 seq/backbone)
Step 3: AlphaFold3 -> top5 sequence complex prediction, ipTM>=0.75=high, >=0.6=low_confidence
Step 4: FreeSASA -> surface SASA analysis of best AF3 structure, Lys/Cys with SASA>40A^2 as conjugation sites
Step 5: Linker Select -> select based on site type: Cys->maleimide_thiol / Lys->NHS_ester, cleavable, compatible MMAE
Step 6: Fetch Payload -> obtain MMAE SMILES + MW from PubChem
Step 7: RDKit Conjugate -> antibody + linker + payload -> ADC SMILES, DAR=4
```

Output adc_design fields: nanobody_sequence, iptm, conjugation_site, linker, payload, dar, adc_smiles, adc_structure_path

**Conjugation chemistry selection rules:**
- Top conjugation site is Cys -> preferred_chemistry = maleimide_thiol
- Top conjugation site is Lys -> preferred_chemistry = NHS_ester
- DAR=4 is the standard drug-to-antibody ratio for ADCs
- MMAE (monomethyl auristatin E) is the standard payload, DM1 as backup

## Common Payloads
- MMAE: tubulin inhibitor, used in Adcetris, requires maleimide/PABC linker
- DM1/DM4: maytansinoid, used in Kadcyla, requires SMCC/SPDB linker
- SN-38: topoisomerase inhibitor, used in Trodelvy, requires CL2A linker
- DXd: used in Enhertu, requires GGFG linker
- Calicheamicin: used in Mylotarg, requires hydrazone linker

## Important Notes
- Maleimide end connects to antibody Cys, not to payload
- MMAE with VC-PABC: PABC connects to MMAE N-terminal amine via carbamate
- rdkit_conjugate generates the linker-payload fragment, used for ADMET
- covalent=false does not affect downstream ADMET prediction workflow

## Pipeline Implementation Details

### Progress Mapping
| Steps | Progress % | Phase |
|-------|-----------|-------|
| Step 0 (fetch_pdb) | 2% | Fetch target protein |
| Step 1 (RFdiffusion) | 5-50% | Backbone generation |
| Step 2 (ProteinMPNN) | 50-75% | Sequence design |
| Step 3 (AF3 validation) | 75-95% | Structure validation |
| Step 4-7 (ADC) | 80-100% | ADC construction |

### Error Isolation
Each step has try/except; single-step failure sets `adc_design.partial=true` without blocking subsequent steps.
Final output returns results from all completed steps + error messages from failed steps.

### AF3 Validation Strategy
- Sort all MPNN designs by score and take top 5
- Submit AF3 complex predictions one by one (binder + target dual-chain)
- 5s interval between each AF3 task to avoid GPU OOM
- ipTM classification: >=0.75 = high, >=0.6 = low_confidence, <0.6 = low
- Final ranking by ipTM descending, select best for ADC step

### RFdiffusion PDB Preprocessing
binder_design mode automatically preprocesses target protein PDB:
- Remove HETATM (water/ligands/glycosylation modifications)
- Sequential renumbering to eliminate residue gaps
- Hotspot residue numbers automatically mapped to new numbering

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## Warning: Notes (auto-synced from CLAUDE.md)

- Step 3: AF3 validation — top5 MPNN -> AF3 complex -> ipTM classification (>=0.75 high / >=0.6 low_confidence)
- Each step has try/except, single-step failure marks `partial: true` without blocking subsequent steps
- 5 second interval between AF3 tasks to avoid GPU OOM, timeout 1800s
- Step 1-3: RFdiffusion -> ProteinMPNN -> AF3 validation (ipTM >=0.75 high / >=0.6 low_confidence)
- Each step has try/except, single-step failure marks partial, without blocking subsequent steps

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->

## NHS-amine Linker Support (2026-03-22)
de novo binder typically has no free Cys -> use NHS-amine (Lys conjugation)

| Linker | Type | Cleavage | DAR | Application |
|--------|------|----------|-----|-------------|
| NHS-PEG4 | nhs_amine | Non-cleavable | 2-4 | Lys conjugation, similar to T-DM1 |
| NHS-PEG2-Val-Cit-PABC | nhs_amine | cathepsin B | 2-4 | Lys conjugation + cleavable |

### Selection Rules
- FreeSASA detects Cys (SASA>40) -> maleimide_thiol
- Only Lys available -> nhs_amine
- De novo binder typically only has Lys -> default nhs_amine

## FreeSASA Notes
- **Does not accept CIF files** — must first convert to PDB (BioPython MMCIFParser -> PDBIO)
- Pipeline already handles CIF->PDB conversion automatically

## RDKit 3D Embedding Limitation
- Large molecule ADC EmbedMolecule may fail
- Fallback: returns 2D SMILES (`embedding_status="2d_only"`), task marked completed
