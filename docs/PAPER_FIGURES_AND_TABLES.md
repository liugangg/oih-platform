# OIH Paper — Figures and Tables List
Target: Nature Methods / Nature Machine Intelligence

## Title (draft)
"Open Intelligence Hub: An LLM-Agent Platform for Autonomous Computational Drug Discovery with PPI-Guided Binder Design"

---

## Figure 1 — Platform Architecture (double column, 183mm)
**Panel A**: Tool ecosystem (32 tools, 15 containers, GPU layout)
**Panel B**: Pipeline flowchart with Tier system decision tree
- Tier 1 (known complex) → extract_interface → skip 6D
- Tier 3 (novel) → PeSTo PPI + RAG → PPI-optimized scoring
**Data**: Already generated (fig1_platform.pdf), needs Tier system update
**Status**: Update needed

## Figure 2 — HER2 Case Study (3-panel)
**Panel A**: ipTM progression across pipeline versions
- v1 baseline: 0.48
- v2 6D scoring: 0.57
- v3 +epitope: 0.60
- **v4 Tier1+domain truncation: 0.86** ← best
**Panel B**: ipSAE vs ipTM scatter plot (all designs)
- HER2 val_0: ipTM=0.84, ipSAE=0.485
- HER2 val_1: ipTM=0.70, ipSAE=0.238
- HER2 val_2: ipTM=0.86, ipSAE=0.529
**Panel C**: ADC assembly summary (3 designs, covalent NHS-amine)
**Data**: All available
**Status**: Regenerate with ipSAE data

## Figure 3 — PeSTo vs DiscoTope3 Comparison
**Panel A**: CD36 PeSTo PPI score heatmap along sequence
- X: residue position, Y: PPI score
- Highlight PeSTo core (187-194) vs DiscoTope3 (397-400)
**Panel B**: CD36 binder design ipTM comparison
- DiscoTope3 hotspots: 0/20 pass, best ipTM=0.43, ipSAE=0.000
- PeSTo hotspots: best ipTM=0.55, ipSAE=0.193
- CLESH literature: pending
**Panel C**: ipSAE reveals false positives
- DT3 ipTM=0.43 but ipSAE=0.000 (no real interface)
- PeSTo ipTM=0.55 and ipSAE=0.193 (genuine binding)
**Data**: All available
**Status**: Need to generate

## Figure 4 — 7-Target Benchmark
**Panel A**: PeSTo difficulty spectrum
- Bar chart: TrkA(0.999) → PD-L1(0.994) → Nectin-4(0.966) → CD36(0.865) → EGFR(0.759) → TROP2(0.422)
**Panel B**: Binder design success rate per target
- HER2: 3/10 (30%), best ipSAE=0.529
- Nectin-4: 2/5 (40%), best ipSAE=0.679 ★
- CD36: 0/25 (ipTM) but 1/5 with ipSAE signal (0.193)
- EGFR: 0/1 partial, TROP2: 0/3 partial
- TrkA/PD-L1: pending
**Panel C**: GPU scheduling efficiency
- 3-parallel RFdiffusion: timeline visualization
**Data**: PeSTo done, Nectin-4+HER2+CD36 AF3 complete, EGFR/TROP2 partial
**Status**: Can generate with current data (3/7 targets complete)

## Figure 5 — Agent Decision Making (schematic)
**Panel A**: RAG 2-layer search decision tree
- Layer 1: PPI interface (co-crystal, mutagenesis)
- Layer 2: Epitope fallback
**Panel B**: Domain truncation strategy
- Full length vs truncated: AF3 speed + accuracy comparison
**Panel C**: Qwen autonomous pipeline planning
- Screenshot: agent chat → tool selection → pipeline execution
**Data**: Available
**Status**: Need to generate

## Figure 6 — Candidate Filtering Funnel (updated)
Horizontal funnel with multi-path:
- PathA: RFdiffusion → MPNN → ESM2 → AF3
- PathB: BindCraft → AF3
- ipSAE post-filter replacing ipTM
**Data**: Available
**Status**: Update from original

---

## Table 1 — Platform Tool Stack
32 tools × 15 containers, categorized by function
Include: PeSTo, extract_interface_residues, pesto_predict

## Table 2 — HER2 Complete Results
| Design | ipTM | ipSAE | ipSAE_d0dom | pDockQ | Site | Linker | ADC |
|--------|------|-------|-------------|--------|------|--------|-----|
| val_2 | 0.86 | 0.529 | 0.669 | 0.664 | K188 | NHS-PEG4 | ✅ |
| val_0 | 0.84 | 0.485 | 0.649 | 0.660 | K107 | NHS-PEG4 | ✅ |
| val_1 | 0.70 | 0.238 | 0.405 | 0.618 | K188 | NHS-PEG4 | ✅ |

## Table 3 — CD36 Hotspot Source Comparison
| Source | Hotspots | n | Best ipTM | Best ipSAE | Method |
|--------|----------|---|-----------|-----------|--------|
| DiscoTope3 (scattered) | A164-A400 | 10 | 0.43 | 0.000 | B-cell epitope |
| DiscoTope3 (clustered) | A397-A400 | 10 | 0.33 | 0.011 | B-cell epitope |
| PeSTo PPI | A187-194 | 5 | 0.55 | 0.193 | PPI interface |
| PeSTo PPI (n=50) | A187-193 | 5 | 0.18 | 0.000 | PPI interface |
| CLESH literature | E101-D109 | — | pending | — | Mutagenesis |

## Table 4 — PeSTo vs DiscoTope3 Accuracy
| Target | PeSTo max | DT3 max | PeSTo >0.5 | Notes |
|--------|-----------|---------|-----------|-------|
| TrkA | 0.999 | — | 59 | Small ECD, extremely druggable |
| PD-L1 | 0.994* | 0.440 | 23 | *Single chain required |
| Nectin-4 | 0.966 | — | 56 | Ig-like D1+D2 |
| CD36 | 0.865 | ~0.36 | 8 | DT3 picks wrong region |
| EGFR | 0.759 | — | 19 | Multi-domain |
| TROP2 | 0.422 | — | 0 | Flat surface, difficult |

## Table 5 — Complete Validated Binder Results (All Targets)

### Nectin-4 (Tier3, PeSTo PPI, PDB 4GJT)
| Design | ipTM | ipSAE | ipSAE_d0chn | pDockQ2 | LIS | Site | SASA | ADC |
|--------|------|-------|-------------|---------|-----|------|------|-----|
| **val_2** | **0.870** | **0.679** | 0.854 | **0.762** | 0.506 | B:K195 | 213.1 Å² | SMCC-NHS+MMAE ✅ |
| val_4 | 0.780 | 0.523 | 0.742 | 0.534 | 0.389 | B:K195 | 214.2 Å² | SMCC-NHS+MMAE ✅ |
| val_1 | 0.310 | — | — | — | — | — | — | — |
| val_3 | 0.260 | — | — | — | — | — | — | — |
| val_0 | 0.210 | — | — | — | — | — | — | — |
Novel binding site on D1-D2 domain (distinct from 4GJT antibody interface at D3-D4)

### HER2 (Tier1, extract_interface from 1N8Z trastuzumab)
| Design | ipTM | ipSAE | ipSAE_d0chn | pDockQ2 | LIS | Site | SASA | ADC |
|--------|------|-------|-------------|---------|-----|------|------|-----|
| **val_2** | **0.850** | **0.529** | 0.823 | 0.222 | 0.400 | A:K188 | 155.6 Å² | NHS-PEG4+MMAE ✅ |
| val_0 | 0.830 | 0.485 | 0.803 | 0.189 | 0.347 | — | — | — |
| val_1 | 0.700 | 0.238 | 0.701 | 0.051 | 0.176 | — | — | — |

### EGFR (Tier1, extract_interface from 1YY9 cetuximab, Domain III)
| Design | ipTM | ipSAE | ipSAE_d0chn | pDockQ2 | LIS | Site | SASA | ADC |
|--------|------|-------|-------------|---------|-----|------|------|-----|
| **val_0** | **0.520** | **0.190** | 0.461 | 0.220 | 0.253 | A:K38 | 171.8 Å² | NHS-PEG4+MMAE ✅ |
| val_1 | 0.300 | 0.016 | 0.335 | 0.050 | 0.124 | — | — | — |
| val_3 | 0.210 | 0.000 | 0.000 | 0.017 | 0.000 | — | — | — |
| val_4 | 0.150 | 0.000 | 0.000 | 0.013 | 0.000 | — | — | — |
| val_2 | 0.080 | 0.000 | 0.000 | 0.010 | 0.000 | — | — | — |
Medium-difficulty target (PeSTo max=0.759). v2 wrong hotspot (domain I, ipTM=0.27) → v5 correct cetuximab hotspot (domain III, ipTM=0.52)

### CD36 (Tier3, PeSTo PPI, PDB 5LGD)
| Design | ipTM | ipSAE | ipSAE_d0chn | pDockQ2 | LIS | Site | SASA | ADC |
|--------|------|-------|-------------|---------|-----|------|------|-----|
| PeSTo v5 val_2 | 0.550 | 0.193 | 0.615 | 0.167 | — | K398 | 184.6 Å² | NHS-PEG4+MMAE ✅ |
| DT3 best | 0.430 | 0.000 | 0.000 | 0.018 | 0.000 | — | — | false positive |
B-cell epitope (DiscoTope3) ≠ PPI interface (PeSTo): ipSAE proves DT3 hotspots are non-binding

## Table 6 — Scoring Formula Evolution
| Version | Formula | Best Result |
|---------|---------|-------------|
| v1 (6D) | p2rank+sasa+conservation+rag+electrostatics+epitope | HER2 ipTM=0.60 |
| v2 (PPI) | rag(0.30)+pesto(0.25)+conservation(0.20)+sasa(0.10)+electrostatics(0.15) | HER2 ipTM=0.86 |

## Table 6 — 7-Target Benchmark Summary
| Target | PDB | Tier | Hotspot Method | Best ipTM | Best ipSAE | Pass Rate | ADC Site (binder) |
|--------|-----|------|---------------|-----------|-----------|-----------|-------------------|
| **Nectin-4** | **4GJT** | **3** | **PeSTo PPI (D1-D2 novel)** | **0.870** | **0.679** | **2/5** | **B:K195 ✅** |
| HER2 | 1N8Z | 1 | extract_interface (trastuzumab) | 0.850 | 0.529 | 3/10 | A:K188 ✅ |
| EGFR | 1YY9 | 1 | extract_interface (cetuximab) | 0.520 | 0.190 | 0/5* | A:K38 ✅ |
| CD36 | 5LGD | 3 | PeSTo PPI | 0.550 | 0.193 | 0/5* | K398 ✅ |
| TROP2 | 7PEE | 3 | PeSTo PPI | 0.220 | — | 0/3 | — |
| TrkA | 1HE7 | 3 | PeSTo PPI | pending | — | — | — |
| PD-L1 | 4ZQK | 1 | extract_interface (atezolizumab) | pending | — | — | — |
*0/5 by ipTM≥0.6, but best has ipSAE>0.15 (genuine binding signal confirmed by ipSAE)
ADC sites verified on binder chain only (antigen Lys excluded)

## Table 7 — Distillation Training Cases
90 cases across 6 sessions, covering:
- Tool selection errors (epitope vs PPI vs pocket)
- Pipeline failures and recovery
- Scoring formula evolution
- GPU scheduling optimization
- Docker orphan process management
- ADC conjugation site validation (binder chain only)
- KNOWN_COMPLEXES chain verification

---

## Supplementary
- S1: All AF3 confidence scores (ipTM + ipSAE) per target
- S2: PeSTo per-residue scores for all 7 targets
- S3: ADC SMILES + 3D structures
- S4: RFdiffusion backbone PDB files
- S5: MPNN designed sequences (FASTA)
- S6: Complete pipeline execution logs
- S7: GPU scheduling timeline visualization
- S8: Distillation training data (81 cases)

---

## Key Citations
1. Abramson et al. Nature 2024 — AlphaFold3
2. Watson et al. Nature 2023 — RFdiffusion
3. Dauparas et al. Science 2022 — ProteinMPNN
4. Modi et al. NEJM 2022 — T-DXd/Enhertu
5. Yin et al. Protein Science 2024 — AF3 nanobody benchmark
6. Tubiana et al. Nature Comms 2022 — PeSTo
7. Dunbrack et al. 2025 — ipSAE metric
8. Gainza et al. Nature Methods 2020 — MaSIF
9. Jumper et al. Nature 2021 — AlphaFold2
10. Baek et al. Science 2021 — RoseTTAFold
