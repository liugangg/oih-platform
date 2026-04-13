# OIH Platform Overview & Key Rules

## Server Requirements
- GPU0: 24GB+ VRAM → Dedicated to LLM inference (vLLM)
- GPU1: 24GB+ VRAM (48GB recommended) → All bioinformatics computation tools
- Ports: FastAPI=8080, vLLM=8002

## GPU Mapping Key Rule
**All computation containers must always use gpu_id=0 / device=0**
- docker-compose sets `NVIDIA_VISIBLE_DEVICES=1` which maps host GPU1 as GPU0 inside containers
- gpu_id=1 does not exist inside containers; using 1 will cause "Device ID 1 did not correspond to any of the 1 detected device(s)"
- Applies to: gromacs, diffdock, rfdiffusion, proteinmpnn, bindcraft, alphafold3, esm, gnina

## Container Inventory

| Container | Image | Purpose | Status |
|-----------|-------|---------|--------|
| oih-gromacs | gromacs:2024.4 | MD simulation | Verified |
| oih-autodock-gpu | autodock-gpu:latest | Molecular docking | Verified |
| oih-gnina | gnina:latest | GPU docking | Verified |
| oih-fpocket | fpocket:latest | Pocket detection | Verified |
| oih-p2rank | p2rank:latest | ML pocket prediction | Verified |
| oih-rfdiffusion | rfdiffusion:latest | Protein backbone design | Verified |
| oih-proteinmpnn | proteinmpnn:latest | Sequence design | Verified |
| oih-diffdock | diffdock:latest | Flexible docking | Verified |
| oih-bindcraft | bindcraft:latest | Binder design | Verified |
| oih-alphafold3 | alphafold3:latest | Structure prediction | Verified |
| oih-esm | esm:latest | Sequence embedding | Verified |
| oih-chemprop | chemprop:latest | Small molecule property prediction | Verified (CPU mode) |

## Model File Locations (Host)

| Tool | Model Path |
|------|-----------|
| RFdiffusion | /data/rfdiffusion/models/ |
| ProteinMPNN | /data/proteinmpnn/ProteinMPNN/ |
| AlphaFold3 | /data/af3/models/ (TBC) |
| Qwen3-14B | /data/oih/models/Qwen3-14B-AWQ |

## Key Paths

| Purpose | Path |
|---------|------|
| OIH API | /data/oih/oih-api/ |
| docker-compose | /data/oih/oih-api/docker-compose.yml |
| Input data | /data/oih/inputs/ |
| Output data | /data/oih/outputs/ |
| Skills docs | /data/oih/oih-api/skills/ |
| AutoDock executable | /usr/local/bin/autodock_gpu_128wi (inside container) |

## Python Command Notes
- oih-rfdiffusion: Use `python3` (no `python`)
- oih-gromacs: Use `gmx`
- oih-autodock-gpu: Use `python3`, prody is installed

## Tools Running Outside Containers (Host)
```bash
obabel           # Ligand/receptor format conversion, generate pdbqt
autogrid4        # AutoDock grid generation
```

## API Route Structure
```
/api/v1/structure  → structure_prediction.py  (AF3, ESM)
/api/v1/design     → protein_design.py        (RFdiffusion, ProteinMPNN, BindCraft)
/api/v1/pocket     → pocket_analysis.py       (fpocket, p2rank)
/api/v1/docking    → molecular_docking.py     (AutoDock-GPU, gnina, DiffDock, Vina)
/api/v1/md         → md_simulation.py         (GROMACS)
/api/v1/ml         → ml_tools.py              (ESM embed, Chemprop)
/api/v1/adc        → adc.py                   (ADC design: linker_select, rdkit_conjugate)
/api/v1/pipeline   → pipeline.py              (Full pipelines)
/api/v1/tasks      → tasks.py                 (Task management)
```

## Task Queue (Implemented)
- CPU queue Semaphore(8): fpocket/p2rank/chemprop/chemprop_predict/freesasa/rdkit_conjugate
- GPU queue Semaphore(3): gnina/autodock/af3/rfdiffusion/proteinmpnn/bindcraft/diffdock/gromacs/esm
- Fallback queue Semaphore(4): Auto-fallback to CPU+memory when GPU VRAM is insufficient
- **Never fallback**: AF3/BindCraft do not enter the fallback queue (OOM crash), wait for GPU availability

## Task Persistence & Cancellation
- State changes are written to `data/tasks/{task_id}.json`
- On service startup, scan directory to recover historical tasks (running/pending marked as failed)
- Fields: task_id, tool, status, progress, progress_msg, result, error, created_at, updated_at
- Cancel endpoint: `DELETE /api/v1/tasks/{task_id}` (pending/running → cancelled)

## System Status Tool
- `GET /api/v1/system/status` → containers[], gpu_queue[], task_summary{}
- LLM can call `get_system_status` to answer questions like "how many containers are running"

## Static File Serving
- `/outputs/` mounts `/data/oih/outputs/`, direct HTTP download of CIF/PDB/SDF files
- Dashboard 3D viewer and download buttons use `/outputs/...` paths

## ADC Toolchain (Added 2026-03-15)

| Tool | Route | Function | Queue |
|------|-------|----------|-------|
| freesasa | /api/v1/design/freesasa | Antibody conjugation site prediction (Lys/Cys SASA>40A^2 analysis) | CPU |
| linker_select | /api/v1/adc/linker_select | ADC linker selection (20 clinically validated library, approved priority) | Sync |
| rdkit_conjugate | /api/v1/adc/rdkit_conjugate | ADC payload-linker covalent conjugation (7 reaction chemistry types) | CPU |

**ADC Design Flow**: fetch_pdb → freesasa (conjugation sites) → fetch_molecule (payload) → linker_select (select linker) → rdkit_conjugate (conjugation) → chemprop (ADMET)

## VRAM Estimates
| Tool | VRAM |
|------|------|
| AlphaFold3 | 20GB |
| BindCraft | 16GB |
| DiffDock | 8GB |
| RFdiffusion | 8GB |
| GROMACS | 6GB |
| ESM | 6GB |
| gnina | 4GB |
| ProteinMPNN | 4GB |
| Vina-GPU | 3GB |
| AutoDock-GPU | 2GB |

## JAX/BindCraft GPU Key Rules
- JAX uses pip-installed CUDA libraries, **does not need** host cuda lib64 mount
- Mounting `/usr/local/cuda-12.6/lib64` causes LD_LIBRARY_PATH conflicts, JAX cannot detect GPU
- BindCraft container **does not mount** any cuda paths, JAX manages CUDA by itself
- Verify: `unset LD_LIBRARY_PATH && python3 -c 'import jax; print(jax.devices())'`
- Other containers (gromacs/gnina etc.) use PyTorch and need cuda lib64 mount; JAX containers do not
