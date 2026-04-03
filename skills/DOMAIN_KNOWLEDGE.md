# Domain Expert Knowledge Base — Computational Drug Discovery

This document is the core knowledge source for the OIH platform AI assistant. Prioritize information from this document when answering customer questions.

---

## 1. Protein Structure Prediction

### AlphaFold3
- Developed by DeepMind, published in 2024, Nature paper
- Can predict: protein monomers, protein-protein complexes, protein-ligand, protein-nucleic acid, protein-ion
- Input: amino acid sequence (FASTA) or PDB ID
- Output: 3D structure (CIF format) + confidence scores
- Key metrics:
  - **pLDDT** (0-100): per-residue local confidence. >90 atomic-level accuracy, 70-90 backbone reliable, <50 likely disordered region
  - **pTM** (0-1): overall structure confidence
  - **ipTM** (0-1): interface/complex prediction quality. >=0.75 high confidence, 0.6-0.75 moderate, <0.6 unreliable
  - **ipSAE** (0-1): interface alignment error. >=0.15 true binding, =0 false positive
- Typical runtime: small proteins 5-10 min, large complexes 20-40 min
- Limitations: inaccurate prediction for disordered regions, novel folds may be unreliable, experimental validation required

### ESM2
- Protein language model developed by Meta AI (650M parameters)
- Uses: sequence embeddings, pseudo-perplexity scoring (PPL), zero-shot mutation effect prediction
- PPL < 15 indicates natural sequence, PPL > 15 may be problematic
- ESM-1v: point mutation scanning, predicts the effect of mutations on protein function

---

## 2. Protein Design

### RFdiffusion
- Developed by David Baker's lab (Nature 2023)
- Principle: diffusion model, generates protein backbone structures from noise
- Modes:
  - **binder design**: designs proteins that bind to a target
  - **scaffold design**: designs protein scaffolds with specific functions
  - **motif scaffolding**: embeds functional motifs into new protein backbones
- Key parameters:
  - hotspot_residues: specifies binding site residues on the target
  - num_designs: number of designs to generate (typically 10-50)
  - binder_length: designed protein length (typically 70-100 aa)
- Note: hotspots must be spatially clustered (<15A), dispersed hotspots cause extremely slow runs
- Output: PDB backbone files (only CA/C/N/O, no side chains)

### ProteinMPNN
- Inverse folding model developed by Baker's lab
- Input: protein backbone structure (from RFdiffusion or experimental structure)
- Output: amino acid sequences (can generate multiple candidate sequences)
- Principle: given 3D backbone coordinates, predicts the sequence most likely to fold into that structure
- Key parameters: temperature (0.1=conservative, 0.5=diverse)
- Typically generates 8-10 sequences per backbone

### BindCraft
- All-in-one binder design (integrates backbone + sequence + optimization)
- Based on JAX, requires significant GPU resources (~16GB VRAM)
- Suitable for rapid prototyping

### IgFold
- Antibody/nanobody sequence to 3D structure rapid prediction
- Outputs pRMSD (not pLDDT), conversion: pseudo_pLDDT = 100 - pRMSD * 20
- Suitable for evaluating foldability of designed antibody sequences

---

## 3. Molecular Docking

### Docking Principles
Molecular docking predicts the binding mode (binding pose) and binding affinity of small molecule ligands with protein receptors.

### Three Docking Tools

#### GNINA (Recommended, Highest Accuracy)
- GPU-accelerated docking based on CNN scoring function
- Inherits AutoDock Vina's search algorithm, replaces scoring function with deep learning CNN
- Output: binding conformation + CNN score (0-1) + affinity (kcal/mol)
- CNN score > 0.7 typically indicates reliable binding prediction
- Affinity < -7 kcal/mol typically indicates significant binding
- Outperforms Vina on PDBbind benchmark
- Suitable for: high-accuracy target-ligand binding prediction

#### AutoDock-GPU
- GPU-accelerated version of AutoDock 4
- Classic Lamarckian genetic algorithm + force field scoring
- Requires preprocessing: PDB to PDBQT conversion + grid calculation (autogrid4)
- Scoring: binding free energy (kcal/mol), more negative is better
- Suitable for: large-scale virtual screening (fast)

#### AutoDock Vina
- Most widely used molecular docking tool
- Trott & Olson, J Comput Chem 2010
- Search algorithm: Iterated Local Search + Monte Carlo
- Scoring function: empirical force field (van der Waals + hydrogen bonds + solvation + entropy penalty)
- Key parameters:
  - center_x/y/z: docking box center coordinates
  - size_x/y/z: box size (typically 20-30 A)
  - exhaustiveness: search thoroughness (default 8, recommended 32)
  - num_modes: number of output conformations
- Affinity unit: kcal/mol, more negative is better
  - < -10: very strong binding
  - -7 to -10: strong binding
  - -5 to -7: moderate binding
  - > -5: weak binding
- Limitations: cannot accurately predict absolute binding free energy, suitable for ranking

#### DiffDock
- Diffusion model docking (MIT, ICLR 2023)
- No need to specify binding pocket (blind docking)
- Suitable when the binding site is unknown
- Outputs confidence scores

### Docking Best Practices
1. **Prepare receptor**: remove water molecules and non-essential HETATM, add hydrogen atoms
2. **Prepare ligand**: generate 3D conformation, assign Gasteiger charges
3. **Define docking box**: use pocket center detected by fpocket/P2Rank
4. **Run docking**: recommend GNINA (accuracy) or AutoDock-GPU (speed)
5. **Validate results**: use MD simulation to verify binding stability

### Common Issues
- **Binding affinity does not equal IC50**: docking predicts relative ranking, not absolute activity
- **Multiple conformations**: select the conformation with best affinity, but also assess chemical reasonableness
- **Metal ions**: active sites containing metals require special handling

---

## 4. Molecular Dynamics Simulation (GROMACS)

### Principles
- Solves Newton's equations of motion, simulates molecular movement under force fields
- Time scale: femtoseconds (fs) to microseconds (us)
- Force fields: describe interatomic interactions (bonds, angles, dihedrals, van der Waals, electrostatics)

### Common Force Fields
- **AMBER99SB-ILDN**: proteins, extensively validated
- **CHARMM36**: proteins + lipid membranes
- **OPLS-AA**: proteins + small molecules
- **GAFF/CGenFF**: small molecule ligand parameterization

### MD Workflow
1. **pdb2gmx**: generate topology
2. **editconf**: set simulation box (margin 1.0 nm)
3. **solvate**: add solvent (TIP3P water)
4. **genion**: add ions to neutralize charge
5. **Energy minimization (EM)**: eliminate unreasonable contacts
6. **NVT equilibration** (100 ps): temperature equilibration (300K)
7. **NPT equilibration** (100 ps): pressure equilibration (1 bar)
8. **Production MD** (10 ns-1 us): generate trajectory

### Key Analyses
- **RMSD**: backbone deviation from initial structure, stable < 2A
- **RMSF**: per-residue flexibility, high RMSF = flexible region
- **Hydrogen bond analysis**: protein-ligand hydrogen bond count and lifetime
- **MM-PBSA/GBSA**: binding free energy calculation
- **Principal Component Analysis (PCA)**: extract major motion modes

### When MD Is Needed
- Validate binding stability after docking
- Study protein conformational changes
- Calculate binding free energy
- Study drug release mechanisms

---

## 5. ADC Drug Design

### ADC Overview
Antibody-Drug Conjugate (ADC) = Antibody + Linker + Payload
- Antibody: targeted delivery, selectively binds tumor cell surface antigens
- Linker: connects antibody and payload, controls drug release
- Payload: cytotoxic drug, kills tumor cells

### Key Parameters
- **DAR** (Drug-to-Antibody Ratio): number of drugs conjugated per antibody
  - DAR 2-4: traditional ADC (e.g., T-DM1)
  - DAR 8: high drug loading ratio (e.g., DS-8201/T-DXd)
  - Excessively high DAR affects pharmacokinetics
- **Target selection**: antigens highly expressed on tumors, low expression on normal tissues
- **Internalization efficiency**: ADC must be internalized by cells before drug release

### Linker Types
#### Cleavable Linkers
- **Acid-labile**: hydrazone, cleaves in lysosomal low pH environment
- **Protease-sensitive**: Val-Cit (VC), cleaved by cathepsin B
  - MC-VC-PABC: most commonly used cleavable linker
- **Disulfide**: cleaves in intracellular reducing environment
- Advantages: efficient drug release
- Disadvantages: may cleave prematurely in blood circulation

#### Non-cleavable Linkers
- **SMCC**: linker used in T-DM1
- Requires complete antibody degradation in lysosomes to release drug
- Advantages: good blood stability
- Disadvantages: released drug retains linker residue

### Common Payloads
| Payload | Mechanism | IC50 | Representative ADC |
|---------|-----------|------|--------------------|
| MMAE | Tubulin inhibition | 0.1-1 nM | Adcetris (brentuximab vedotin) |
| MMAF | Tubulin inhibition | 1-10 nM | Blenrep (belantamab mafodotin) |
| DM1 | Tubulin inhibition | 0.1-1 nM | Kadcyla (T-DM1) |
| DXd | TopoI inhibition | 1-10 nM | Enhertu (T-DXd, DS-8201) |
| SN-38 | TopoI inhibition | 1-10 nM | Trodelvy (sacituzumab govitecan) |
| PBD | DNA crosslinking | pM range | Zynlonta (loncastuximab tesirine) |

### Conjugation Chemistry
- **Lysine conjugation** (NHS-amine): non-site-specific, heterogeneous DAR
- **Cysteine conjugation** (maleimide-thiol): conjugation after partial reduction, more uniform DAR
- **Site-specific conjugation**: engineered sites (e.g., unnatural amino acids), uniform DAR
- Conjugation site selection criteria:
  - SASA > 40 A^2 (solvent accessible)
  - Not in CDR regions (does not affect target binding)
  - Not in Fc effector regions (does not affect immune function)

### Approved ADCs
| Drug | Target | Indication | Approval Year |
|------|--------|------------|---------------|
| Adcetris | CD30 | Hodgkin lymphoma | 2011 |
| Kadcyla | HER2 | Breast cancer | 2013 |
| Besponsa | CD22 | ALL | 2017 |
| Enhertu | HER2 | Breast cancer/gastric cancer | 2019 |
| Padcev | Nectin-4 | Urothelial carcinoma | 2019 |
| Trodelvy | TROP2 | Triple-negative breast cancer | 2020 |
| Zynlonta | CD19 | DLBCL | 2021 |

---

## 6. ADMET Prediction

### What Is ADMET
- **A**bsorption: whether the drug can be absorbed by the body
- **D**istribution: drug distribution in the body
- **M**etabolism: how the drug is metabolized
- **E**xcretion: how the drug is eliminated from the body
- **T**oxicity: adverse effects of the drug

### Five Properties We Predict
1. **ESOL Solubility** (logS):
   - logS > -1: high solubility
   - -1 to -4: moderate
   - < -4: low solubility (may affect oral absorption)

2. **Lipophilicity** (logP/logD):
   - 1 < logP < 3: ideal range (Lipinski Rule of Five)
   - logP > 5: may have membrane permeability issues

3. **Blood-Brain Barrier (BBB) Permeability**:
   - Predicted probability > 0.7: likely crosses BBB
   - Important for CNS drugs, should be avoided for non-CNS drugs

4. **Tox21 Toxicity**:
   - NR-AhR pathway activation prediction
   - Score < 0.5: low toxicity risk
   - Score > 0.5: requires further toxicological study

5. **Hydration Free Energy** (FreeSolv):
   - Reflects strength of molecule-water interactions
   - Related to solubility

### Lipinski Rule of Five (Oral Drugs)
- Molecular weight < 500 Da
- logP < 5
- Hydrogen bond donors < 5
- Hydrogen bond acceptors < 10
- Violating more than 2 rules indicates potentially poor oral bioavailability

---

## 7. Pocket Detection and Interface Prediction

### fpocket
- Geometric algorithm based on Voronoi tessellation and alpha-spheres
- Detects cavities (pockets) on protein surfaces
- Output: pocket ranking + druggability score + residue list

### P2Rank
- Machine learning pocket prediction
- Random forest trained on surface features
- Output: pocket probability + center coordinates + residues

### PeSTo (PPI Interface Prediction)
- Predicts protein-protein interaction interfaces
- ROC AUC = 0.92 (outperforms MaSIF-site at 0.80)
- Output: PPI interface probability for each residue
- Difference from DiscoTope3:
  - PeSTo predicts PPI interfaces (protein-protein binding sites) -- suitable for binder design
  - DiscoTope3 predicts B-cell epitopes (immunogenic surfaces) -- suitable for vaccine design
  - Results may be entirely different (validated with CD36 case)

### DiscoTope3
- B-cell epitope prediction (predicts surface residues recognizable by antibodies)
- Based on ESM-IF1 + XGBoost
- Note: DT3 raw score range varies by structure, do not use a fixed threshold

---

## 8. Target Tier Classification System

| Tier | Method | Reliability | Application Scenario |
|------|--------|-------------|----------------------|
| Tier 1 | Structural database | Highest | Known antibody-antigen co-crystal structure in PDB |
| Tier 2 | RAG literature-guided | Moderate | Homologous structures or mutagenesis data in literature |
| Tier 2 | Computational prediction | Lower | Novel targets, relies on PeSTo/DT3 prediction |

### Validated Targets
| Target | PDB | Tier | Best ipTM | Best ipSAE | Application |
|--------|-----|------|-----------|------------|-------------|
| HER2 | 1N8Z | 1 | 0.85 | 0.529 | Breast cancer |
| Nectin-4 | 4MZV | 2 | 0.87 | 0.679 | Urothelial carcinoma |
| EGFR | 1YY9 | 1 | 0.52 | 0.190 | Non-small cell lung cancer |
| CD36 | 5LGD | 3 | 0.58 | 0.056 | Metabolic disease |
| TROP2 | -- | 3 | 0.22 | 0.000 | Triple-negative breast cancer |
| PD-L1 | 5XXY | 1 | -- | -- | Immune checkpoint |

---

## 9. Common Customer Scenarios

### Scenario 1: Binder Design for Known Target
Input: target name/PDB ID
Workflow: obtain structure -> hotspot analysis -> RFdiffusion -> ProteinMPNN -> AF3 validation -> ADC assembly
Output: designed sequences + structures + validation metrics

### Scenario 2: Small Molecule Virtual Screening
Input: target PDB + candidate molecule SMILES
Workflow: pocket detection -> molecular docking -> ADMET prediction
Output: binding conformations + affinity ranking + ADMET assessment

### Scenario 3: ADMET Assessment of Known Drugs
Input: molecule name or SMILES
Workflow: obtain SMILES from PubChem -> Chemprop prediction
Output: 5 ADMET properties + interpretation

### Scenario 4: Protein Structure Prediction
Input: FASTA sequence or protein name
Workflow: AlphaFold3 prediction
Output: 3D structure + confidence scores

### Scenario 5: Molecular Dynamics Validation
Input: PDB complex + simulation duration
Workflow: GROMACS MD pipeline
Output: RMSD/RMSF analysis + binding stability assessment

---

## 10. Advanced Docking Result Interpretation

### Relationship Between Binding Affinity and Experimental Values
- Docking affinity is an **estimate**, not equal to experimental IC50/Kd
- Vina/GNINA affinity error is typically +/-2 kcal/mol
- Affinity ranking (relative ranking) is more meaningful than absolute values
- -7 kcal/mol corresponds to approximately uM-level binding (Kd ~ exp(dG/RT))
- -10 kcal/mol corresponds to approximately nM-level binding

### Docking Box Setup Recommendations
- **Box center**: use pocket center detected by fpocket/P2Rank
- **Box size**: pocket diameter + 10A buffer (typically 20-30A)
- **Too small**: ligand cannot find correct conformation
- **Too large**: search space too large, slow and inaccurate results

### Common Docking Failure Causes
1. **Improper receptor preparation**: missing hydrogen atoms, unprocessed water molecules, metal ion parameterization
2. **Incorrect ligand 3D conformation**: conformational search needed first
3. **Wrong pocket**: incorrect binding site selected
4. **Scoring function limitations**: inaccurate assessment of hydrophobic effects and entropy

---

## 11. ADC Clinical Development Key Points

### Common Reasons for ADC Failure
1. **Poor target selection**: high expression on normal tissues -> systemic toxicity
2. **Unstable linker**: premature release in blood circulation -> systemic toxicity
3. **Excessively high DAR**: accelerated clearance, reduced efficacy
4. **Payload bystander effect**: kills surrounding normal cells
5. **Immunogenicity**: anti-drug antibody (ADA) generation

### Gold Standard for ADC Design
- Target: tumor/normal expression ratio > 10:1
- Internalization efficiency > 50%
- DAR 4+/-0.5 (uniform)
- Plasma half-life > 3 days
- Controllable bystander effect

### Key Data for Approved ADCs
| Drug | Target | Payload | Linker | DAR | ORR |
|------|--------|---------|--------|-----|-----|
| T-DM1 (Kadcyla) | HER2 | DM1 | SMCC (non-cleavable) | 3.5 | 44% |
| Adcetris | CD30 | MMAE | MC-VC-PABC | 4 | 75% |
| Enhertu (T-DXd) | HER2 | DXd | GGFG tetrapeptide | 8 | 62% |
| Padcev | Nectin-4 | MMAE | MC-VC-PABC | 3.8 | 44% |
| Trodelvy | TROP2 | SN-38 | CL2A (acid-labile) | 7.6 | 33% |

---

## 12. Advanced Molecular Dynamics Analysis

### MM-PBSA Binding Free Energy
- dG_bind = dG_complex - dG_receptor - dG_ligand
- Includes: van der Waals + electrostatics + polar solvation + nonpolar solvation - TdS
- dG < -10 kcal/mol: strong binding
- dG < -5 kcal/mol: moderate binding
- Typically calculated from the last 2-5 ns frames of the MD trajectory

### MD Quality Control
- **Temperature fluctuation**: within +/-5K is normal
- **Pressure fluctuation**: within +/-100 bar is normal (NPT)
- **RMSD convergence**: first 1-2 ns is equilibration period, should stabilize afterwards
- **Energy drift**: total energy should be conserved, should not continuously increase/decrease

### When Longer MD Is Needed
- 10 ns: basic validation of binding stability
- 100 ns: observe conformational changes and ligand dissociation
- 1 us: protein folding/large conformational changes
- Above 100 ns, enhanced sampling is recommended (MetaD, REST2)
