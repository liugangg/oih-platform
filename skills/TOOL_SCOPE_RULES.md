# Tool Scope Rules

## Core Principle
Each tool has a specific scope of applicability. Incorrect usage will not throw errors, but output will be meaningless.
The agent must determine tool applicability before making a call.

## Tool Applicability Table

| Tool | Valid Input | Prohibited Input | Failure Mode |
|------|------------|-----------------|--------------|
| igfold_predict | Antibody/nanobody sequences | de novo binder sequences | pLDDT=5-40, all filtered out |
| discotope3_predict | Protein antigen PDB | Small molecule targets | Meaningless epitope prediction |
| extract_interface_residues | PDB with known complex | Targets without structure | File not found error |
| fpocket | Any protein PDB | Antibody binding surface prediction | Finds small molecule pockets, not antibody sites |
| esm2_score_sequences | Any protein sequence | None | General purpose, safe |
| esm2_mutant_scan | Any protein sequence | None | General purpose, safe |
| rfdiffusion_design | Any protein target | None | General purpose |
| proteinmpnn_design | Any backbone PDB | None | General purpose |
| alphafold3_predict | Any sequence/complex | None | General purpose, final validation |

## binder_type Determination Rules

| User Says | binder_type | IgFold |
|-----------|-------------|--------|
| "nanobody/VHH/single-domain antibody" | nanobody | Enabled |
| "antibody/IgG" | antibody | Enabled |
| "design binder/binding protein/de novo" | de_novo | Skipped |
| Not specified | de_novo (default) | Skipped |

## Binder Design Path Selection

### Known complex PDB available (Tier 1, e.g., HER2+trastuzumab=1N8Z)
```
extract_interface_residues -> RFdiffusion -> ProteinMPNN -> ESM2 -> AF3
```

### No known structure (Tier 2, novel target)
```
DiscoTope3 + IEDB + RAG -> known_epitope_override -> RFdiffusion -> ProteinMPNN -> ESM2 -> AF3
```

### Nanobody design (IgFold applicable)
```
... -> ProteinMPNN -> ESM2(PPL<15) -> IgFold(pLDDT>70) -> AF3
```

### De novo binder design (IgFold not applicable)
```
... -> ProteinMPNN -> ESM2 perplexity filter (top 10) -> AF3 direct validation
```

## Validation Results (2026-03-22)

### HER2 Tier 1 + de_novo + Domain Truncation
- extract_interface_residues -> C558-C573 (Domain IV) -> RFdiffusion -> AF3
- val_0: ipTM=0.84 (full-length 1015aa), val_2: ipTM=**0.86** (Domain4 202aa)
- **Domain truncation outperforms full-length** — AF3 is more accurate for short sequence complexes
- 3/3 passed ipTM>=0.6 (100% pass rate)

## RAG Priority Rules (universal for all targets)

**PPI interface > epitope prediction**

When RAG returns known protein-protein interaction interfaces (co-crystal structures, mutagenesis-validated binding residues),
these residues must be prioritized as hotspots, overriding DiscoTope3/IEDB epitope predictions.

- B-cell epitope prediction identifies immunogenicity, not optimal binder design sites
- PPI interface residues are experimentally validated protein binding sites, directly applicable to binder design
- RAG search has two layers: Layer 1 (PPI co-crystal/mutagenesis) -> Layer 2 (epitope fallback)
- CD36 lesson: DiscoTope3 selected surface-exposed residues A397/A400, but should have used CLESH domain (93-120)

## MPNN Chain Detection (2026-03-26 final fix)

Chain order in RFdiffusion binder_design output is not fixed:
- Shortest chain = binder scaffold (60-120aa)
- Longest chain = target protein (100-600aa)

**chains_to_design has been changed to default "auto"**. Router automatically detects the shortest chain using gemmi.
When calling `proteinmpnn_sequence_design`, no need to specify chains_to_design, leave at default.

| Target | RFdiff chain A | RFdiff chain B | MPNN auto detection |
|--------|---------------|---------------|---------------------|
| HER2 (original chain C) | binder 214aa | target 76aa | A (shortest) |
| CD36 (original chain A) | target 400aa | binder 78-95aa | B (shortest) |
| EGFR (original chain A) | target 613aa | binder 74aa | B (shortest) |
| Trop2 (original chain A) | target 274aa | binder ~80aa | B (shortest) |

Validation: MPNN FASTA designed sequence length 60-120aa = correct (binder), >200aa = incorrect (target)

## ipSAE Interface Validation (2026-03-26 addition)

After AF3 validation completes, **must call ipsae_score** to check interface quality:
- ipSAE > 0.15 = true positive (binder actually binds antigen)
- ipSAE = 0.000 = false positive (AF3 gave ipTM but no actual interface contact)
- Call method: `ipsae_score(af3_output_dir="AF3 output directory path")`
- CD36 DT3 route lesson: 21 designs with ipTM=0.12-0.43, but all ipSAE=0.000

## Sequence Extraction Safety Rules (2026-03-27 bug fix)

When extracting sequences from PDB, **must specify chain ID**, otherwise multiple chains will be concatenated into one:
- Wrong: `_extract_sequence_from_pdb(pdb)` -> 400aa target + 69aa binder = 469aa concatenated
- Correct: `_extract_sequence_from_pdb(pdb, chain_id='B')` -> 69aa binder
- Impact: 61 pre-fix designs all submitted incorrect sequences to AF3
- Validation: AF3 input JSON binder chain should be 60-120aa; if >200aa it is definitely a concatenation error

## GPU Resource Leak Protection (2026-03-27 critical fix)

`docker exec` processes created inside containers have containerd-shim as parent, **not FastAPI**.
task_manager releases the semaphore after task completion, but python/jackhmmer processes inside containers may still be running, occupying 44GB VRAM.

**Fixed**: after each GPU task completes (success or failure), automatically kill matching processes inside the container.
Two-layer protection:
1. `_cleanup_container_after_task()` — immediate cleanup after task completion
2. `_kill_orphaned_gpu_processes()` — cleanup historical orphans on FastAPI restart

If GPU is full but no running tasks -> check `nvidia-smi` for PID -> `docker exec <container> kill -9 <pid>`

### Historical Lessons
- IgFold incorrectly used on de novo binder -> pLDDT all 0.4-12.6 -> 0 candidates entered AF3
- Root cause: IgFold uses AntiBERTy, trained only on antibody datasets
- Fix: pipeline added binder_type parameter, skips IgFold for de_novo
- FreeSASA does not accept CIF -> must convert to PDB first
- GPU Semaphore must be =1, otherwise two large models running concurrently cause OOM
