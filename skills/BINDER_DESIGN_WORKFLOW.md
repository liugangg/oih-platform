# Binder Design Workflow — Unified Decision Chain

## Complete Workflow
Target input -> Tier classification -> RAG 2-layer search -> PeSTo PPI prediction ->
Hotspot selection -> Domain truncation -> RFdiffusion/BindCraft -> ProteinMPNN ->
ESM2 -> AF3 validation -> ADC assembly

## 1. Tier Classification
- Tier1: KNOWN_COMPLEXES hit or RAG Layer1 found co-crystal/mutagenesis interface
  -> extract_interface_residues -> use directly as hotspot
- Tier3: no known interface -> PeSTo + RAG + conservation multi-tool consensus

## 2. RAG Two-Layer Search (must be executed first for all targets)
Layer 1 (PPI interface — highest priority):
  "{target} protein complex co-crystal binding partner interface residues"
  -> found experimentally validated residues -> use directly, skip computational prediction

Layer 2 (epitope fallback — only when Layer1 yields no results):
  "{target} epitope binding site"

Key rule: B-cell epitope ≠ binder design hotspot
- DiscoTope3/IEDB predicts immunogenicity, not suitable for binder hotspot
- CD36 evidence: DiscoTope3 A397-400 -> ipTM=0.33 all failed
- Baker lab has never used epitope tools to select hotspots

## 3. Hotspot Discovery: Tier Determines Method

### Core Rule: Tier1 uses extract_interface, Tier3 uses PeSTo
| Tier | Condition | Hotspot Method | Example |
|------|-----------|---------------|---------|
| Tier1 | PDB has co-crystal complex | `extract_interface_residues` directly extracts interface residues | HER2(1N8Z), EGFR(1YY9), PD-L1(4ZQK) |
| Tier3 | No co-crystal/no known binding partner | PeSTo predicts single-chain PPI interface | CD36(5LGD), TROP2(7PEE), Nectin-4(4GJT) |

**Prohibited**: running PeSTo on Tier1 targets (wastes time, co-crystal interface is more reliable than prediction)
**Prohibited**: running extract_interface on Tier3 targets (no co-crystal -> no interface to extract)

### Tier1: extract_interface_residues
- Input: co-crystal complex PDB + receptor_chain + ligand_chains
- Output: interface residue list (contact residues within <5A distance)
- Used directly as RFdiffusion hotspot, no additional prediction needed

### Tier3: PeSTo PPI Interface Prediction
- ROC AUC 0.92（vs MaSIF-site 0.80）
- **Must input single-chain PDB** (complex will suppress occupied interface scores, e.g., PD-L1 from 0.44->0.99)
- Operation: first extract target single-chain PDB with gemmi -> then call pesto_predict
- Deployment: inside proteinmpnn container at /app/pesto/

### PeSTo Validated Target Difficulty Table (Tier3 targets only)
| Target | PDB | Chain | Max PPI | >0.5 residues | Difficulty |
|--------|-----|-------|---------|----------------|------------|
| TrkA | 1HE7 | A | 0.999 | 59 | Very easy |
| Nectin-4 | 4GJT | A | 0.966 | 56 | Easy |
| CD36 | 5LGD | A | 0.865 | 8 | Moderate |
| TROP2 | 7PEE | full | 0.422 | 0 | Difficult |

## 4. Hotspot Selection Rules
Source priority:
1. Co-crystal interface residues (extract_interface, Tier1) >>> all computational predictions
2. PeSTo PPI interface (score > 0.5, Tier3) > single computational tool
3. Conserved + surface-exposed + moderate concavity > highly exposed but flat

Spatial clustering: _cluster_hotspots()
- CA distance <= 15A, max 5 residues per group
- Dispersed hotspots -> _multi_cluster_hotspots() grouped into independent designs
- Must contain at least 1 charged residue + 1 hydrophobic residue

## 5. Scoring Formula (PPI-optimized version, replaces old 6D)
rag(0.30) + pesto_ppi(0.25) + conservation(0.20) +
sasa(0.10) + electrostatics(0.15)

Removed:
- P2Rank (small molecule pocket tool, not suitable for PPI)
- DiscoTope3 epitope (B-cell epitope ≠ PPI interface)
- DiffDock (small molecule docking tool, removed from pipeline)

## 6. Domain Truncation (DOMAIN_REGISTRY)
Registered: HER2 Domain IV / CD36 CLESH region / EGFR Domain III / PDL1 IgV
Rule: full-length >500aa must be truncated to ~200aa
Benefit: AF3 speed 5-8x, accuracy equal or better

## 7. Dual-Path Parallel Design
PathA: RFdiffusion (hotspot specified) -> ProteinMPNN -> ESM2
PathB: BindCraft (no hotspot, AF2 free exploration)
Both paths enter AF3 validation, select the best

## 8. AF3 Validation
- num_seeds=3
- Current threshold ipTM >= 0.6
- Dynamic timeout: <500aa 1200s, 500-1000aa 2400s, >1000aa 3600s

## 9. ADC Assembly (only for designs passing AF3)
FreeSASA -> surface-exposed Lys -> linker_select -> rdkit_conjugate
DAR=4, NHS-PEG4-MMAE, 7 NHS-amine SMARTS patterns

## 10. Tool Scope
- DiffDock: removed, small molecules only
- P2Rank/fpocket: small molecule docking pipeline only
- DiscoTope3: does not participate in binder scoring, auxiliary reference only
- IgFold: nanobody/antibody only, skip for de novo binder
- PeSTo: PPI interface prediction, replaces P2Rank+DiscoTope3

## 11. Multi-Target Parallel Scheduling Strategy

### GPU Resource Model (RTX 4090, 44GB)
| Tool | VRAM | Time | Parallelizable |
|------|------|------|--------|
| RFdiffusion | 4-10GB | 20min | 3 parallel (~20GB total) |
| ProteinMPNN | 4GB | 2min | Can run parallel with RFdiff |
| ESM2 | 6GB | 5min | Can run parallel with MPNN |
| AF3 | 20GB | 15min | Must run exclusively |
| BindCraft | 16GB | 30min | Max +1 small task |
| PeSTo | 0 (CPU) | 10s | No GPU usage |
| FreeSASA | 0 (CPU) | 5s | No GPU usage |

### Optimal Scheduling: Pipeline-Style
```
GPU timeline:
─────────────────────────────────────────────────
RFdiff(target1) + RFdiff(target2) + RFdiff(target3)  <- 3 parallel
    ↓              ↓              ↓
MPNN(target1) + RFdiff(target4) + MPNN(target2)      <- interleaved
    ↓                              ↓
AF3(target1_val0) + MPNN(target3)                <- AF3 exclusive + small task
    ↓
AF3(target1_val1)
    ↓
AF3(target2_val0) + MPNN(target4)
    ...
─────────────────────────────────────────────────
CPU timeline (fully parallel, does not wait for GPU):
PeSTo(all targets) | RAG(all targets) | FreeSASA | ipSAE calculation
─────────────────────────────────────────────────
```

### Scheduling Rules
1. **Pipeline runs on CPU queue**: pipeline is an orchestrator, does not occupy GPU slot
2. **GPU semaphore=3**: max 3 GPU tasks in parallel (VRAM-based routing auto-determines)
3. **RFdiffusion can run 3 parallel**: 4-10GB each, 3 total 20-30GB < 44GB
4. **AF3 must run exclusively**: 20GB, at most 1 AF3 + 1 small task (MPNN/ESM2) simultaneously
5. **CPU tasks run anytime**: PeSTo/FreeSASA/RAG/RDKit not limited by GPU
6. **Failure recovery**: completed RFdiffusion backbones remain in outputs/, MPNN can be manually re-run

### Batch Submission Strategy
When users request multiple targets:
```python
# Correct: submit all at once, let scheduler auto-queue
for target in targets:
    submit_pipeline(target)  # all pending, auto-scheduled by VRAM

# Wrong: wait for one to complete before submitting next
for target in targets:
    submit_pipeline(target)
    wait_until_complete()  # wastes GPU idle time
```

### Failure Recovery Mode
If pipeline fails midway (e.g., MPNN timeout):
1. Check which step completed (RFdiffusion backbone in outputs/)
2. Manually re-run failed step (no need to re-run RFdiffusion)
3. Example: CLESH v4 RFdiffusion done -> MPNN timeout -> manually submit 9 MPNN tasks

## Case References
HER2 (Tier1): C558-C573 -> ipTM=0.86, 3/10 pass, ADC successful
CD36 DiscoTope3: A397-400 -> ipTM=0.33, ipSAE=0.000, 0/10 all failed
CD36 PeSTo v5: A187-194 -> ipTM=0.55, ipSAE=0.193, best CD36 design
CD36 PeSTo v6 n=50: ipTM=0.18, more != better (diluted exploration focus)
CD36 CLESH: E101/D106/E108/D109 -> MPNN timeout, manual re-run in progress

## Domain Truncation Autonomous Decision Rules (independent of hardcoded DOMAIN_REGISTRY)

When DOMAIN_REGISTRY does not have the target, the agent should autonomously determine truncation range:

### Step 1: Check UniProt domain annotations
- Signal peptide (remove)
- Transmembrane (remove)
- Topological domain: Extracellular (keep)
- Domain: Ig-like D1/D2/D3

### Step 2: Determine the minimal domain containing the hotspot
- Truncate to that domain +/- 30 residue buffer

### Step 3: Size validation
- Target: 100-250aa (AF3 most efficient range)
- <100aa: expand buffer; >500aa: must truncate

### Step 4: Structural integrity
- Do not cut through beta-sheet/alpha-helix
- Disulfide bond Cys-Cys pairs must both be within range
- Choose loop/linker regions as boundaries

### Truncation Formula
center = mean(hotspots); range = (min(hotspots)-50, max(hotspots)+50)
If <80aa, expand to center +/- 60

### Cases
- HER2 C558-C573 -> Domain IV (488-630) 142aa -> ipTM=0.86
- CD36 PeSTo A187-194 -> (140-240) 100aa -> pending validation
- CD36 full-length 469aa -> ipTM=0.33 failed

## CRITICAL: RFdiffusion Output Chain Order Rules

In RFdiffusion binder_design mode output PDB, **binder is always the shortest chain**, but chain names are not fixed:
- HER2 (original chain C) -> RFD output chain A=binder(214aa), chain B=target(76aa truncated)
- CD36 (original chain A) -> RFD output chain A=target(400aa), chain B=binder(78-95aa)
- EGFR (original chain A) -> RFD output chain A=target(613aa), chain B=binder(74aa)

### MPNN chains_to_design Rules (2026-03-26 final fix)
- **Default value "auto"**: router automatically uses gemmi to detect shortest chain and designs that chain's sequence. No need to specify.
- **Never pass chains_to_design="A"**: this was a critical bug from 2026-03-24, caused CD36/EGFR/Trop2 to all design the target protein (400aa) instead of the binder (80aa)
- **Validation**: MPNN FASTA designed sequence length should be 60-120aa (binder). If >200aa = designed the wrong chain.

### ipSAE Validation (2026-03-26 addition, must execute after AF3)
After AF3 completes, must call `ipsae_score(af3_output_dir=AF3_output_directory)` to check interface quality:
- ipSAE > 0.15 -> true positive, proceed downstream
- ipSAE = 0.000 -> false positive, binder did not truly bind antigen
- CD36 DT3 route: all 21 designs had ipSAE=0 (highest ipTM=0.43 was also false positive)
- Nectin-4 best: ipTM=0.87, ipSAE=0.679 (true positive benchmark)

## New Target Checklist (MANDATORY)
Must complete before adding target to KNOWN_COMPLEXES:
1. `gemmi.read_structure()` to check all chain lengths and numbering
2. Confirm receptor_chain (usually the longest protein chain = target)
3. Confirm ligand_chains (antibody Fab = Heavy + Light two chains, do not guess)
4. After writing to KNOWN_COMPLEXES, use `extract_interface` to verify reasonable hotspot count (>=3 residues)
5. Update chain ID list in `qwen_tools.py` tool description accordingly

**Counter-example:** EGFR 1YY9 guessed ligand_chains=["B"], actual was ["C","D"]. CD20 6Y4J has only a single chain with no antibody, cannot be added to KNOWN_COMPLEXES.
