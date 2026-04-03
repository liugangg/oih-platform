# Startup Rules (execute at the beginning of every session)
1. Find and read the latest session log: ls -t docs/*.md | head -1, then Read the entire file
2. Report: pending items, caveats, issues from previous session
3. Check service status: curl -s --noproxy '*' http://localhost:8080/health
4. Confirm everything is ready before accepting tasks

---

# OIH — Open Intelligence Hub

Drug discovery AI agent platform. Researchers submit natural language requests (e.g. "design an antibody against human CD36"), Qwen3-14B autonomously plans and executes full computational biology workflows using available tools.

---

## Server & Ports

| Service | Address |
|---------|---------|
| FastAPI backend | 192.168.31.23:8080 |
| Qwen3-14B via vLLM | 192.168.31.23:8002 |
| Nginx (public entry) | 192.168.31.23:8000 |

---

## GPU Layout — CRITICAL

- **GPU0 (RTX 4090)**: Qwen3-14B inference only via vLLM
- **GPU1 (RTX 4090, ~45GB VRAM)**: ALL bioinformatics compute tools

All compute containers set `NVIDIA_VISIBLE_DEVICES=1` in docker-compose, which maps host GPU1 as **device 0 inside the container**. Therefore:

> **Always use `gpu_id=0` or `device=0` inside any container. Never use 1.**

Frameworks that ignore `NVIDIA_VISIBLE_DEVICES` (e.g. DiffDock) also need explicit `CUDA_VISIBLE_DEVICES=0` in container environment.

JAX-based containers (AlphaFold3, BindCraft): **never mount host cuda lib64 paths** — JAX bundles its own CUDA libraries. Mounting host paths breaks GPU detection.

---

## Key File Locations

```
/data/oih/oih-api/
├── qwen_agent.py          # Agent core: 20-round tool-calling loop
├── skills_loader.py       # Keyword → skill doc injection
├── pipeline.py            # High-level pipelines (binder_design_pipeline etc.)
├── main.py                # FastAPI app, router registration
├── tool_definitions/
│   └── qwen_tools.py      # Tool definitions visible to Qwen
├── routers/               # Backend execution logic per tool
│   ├── structure_prediction.py
│   ├── docking.py
│   └── ...
└── skills/                # Markdown skill docs injected into system prompt
    ├── ALPHAFOLD3.md
    ├── AUTODOCK.md
    └── ...

/data/alphafold3_models    # AF3 model weights
/data/alphafold3_db        # AF3 databases
/data/af3                  # AF3 input/output
/usr/local/bin/autodock_gpu_128wi  # AutoDock-GPU executable (on host)
```

---

## Three-Location Rule — Every New Tool Needs Changes in Exactly 3 Places

1. **Router** (`routers/<category>.py`) — backend execution logic, docker exec, file I/O
2. **Tool definition** (`tool_definitions/qwen_tools.py`) — what Qwen sees: name, description, parameters
3. **TOOL_MAP** (`qwen_agent.py`) — agent routing table mapping tool name → router function

Skills documents (`skills/`) are NOT needed for self-contained tools like `fetch_pdb` or `fetch_molecule`.

---

## Agent Architecture — Core Principle

**No hardcoded pipelines.** The agent does only three things:

1. Write accurate tool descriptions including upstream/downstream relationships
2. Give the system prompt domain knowledge (not workflow templates)
3. Run a multi-round tool-calling loop — Qwen plans dynamically

Qwen reads file paths from tool outputs and passes them to subsequent tools itself. No code-level path injection needed.

---

## Validated Tool Stack (13 containers, all on GPU1/device=0)

### Protein design
| Tool | Container | Notes |
|------|-----------|-------|
| AlphaFold3 | `oih-alphafold3` | ✅ Validated end-to-end |
| RFdiffusion | `oih-rfdiffusion` | ✅ Validated |
| ProteinMPNN | `oih-proteinmpnn` | ✅ Validated |
| BindCraft | `oih-bindcraft` | ✅ Validated (JAX, no host cuda mount) |
| ESM2-650M | `oih-base` | ✅ GPU inference validated |

### Pocket detection
| Tool | Container | Notes |
|------|-----------|-------|
| fpocket | `oih-fpocket` | ✅ Returns top pockets; output used as hotspot_residues for RFdiffusion and box coords for docking |
| P2Rank | `oih-p2rank` | ✅ ML-based pocket prediction |

### Docking
| Tool | Container | Notes |
|------|-----------|-------|
| GNINA | `oih-gnina` | ✅ CNN scoring, GPU |
| AutoDock-GPU | `oih-autodock-gpu` | ✅ Frontend validated; pipeline integration in progress |
| DiffDock | `oih-diffdock` | ✅ Validated (needs `CUDA_VISIBLE_DEVICES=0`) |

### MD & ML
| Tool | Container | Notes |
|------|-----------|-------|
| GROMACS | `oih-gromacs` | ✅ GPU command: `gmx mdrun -gpu_id 0 -nb gpu -pme gpu -update gpu -ntmpi 1 -ntomp 16` |
| Chemprop | `oih-chemprop` | ✅ See Chemprop rules below |

### Chemprop — Three mandatory rules (hard-won lessons)

1. **Dockerfile must include**: `RUN ln -sf /usr/bin/python3.11 /usr/bin/python3` (container python3 defaults to 3.10, but all packages are installed under the 3.11 path)
2. **predict calls must include**: `--accelerator cpu` (small-batch GPU inference causes OOM), routed to `_CPU_TOOLS` queue
3. **`--devices 1` = "use 1 GPU device"**, not device index=1. For train use `--accelerator gpu --devices 1`, for predict use `--accelerator cpu`

**ADMET model paths** (trained on MoleculeNet standard datasets, 30 epochs):
```
/data/oih/models/admet/esol/model_0/best.pt          — Solubility logS (regression, MSE=0.54)
/data/oih/models/admet/freesolv/model_0/best.pt      — Hydration free energy (regression)
/data/oih/models/admet/lipophilicity/model_0/best.pt  — Lipophilicity logD (regression, MSE=0.43)
/data/oih/models/admet/bbbp/model_0/best.pt          — Blood-brain barrier (classification, AUC=0.89)
/data/oih/models/admet/tox21/model_0/best.pt         — Tox21 NR-AhR toxicity (classification, AUC=0.90)
```

### GROMACS — Critical pitfalls for protein-ligand systems

1. **tc-grps must use `Protein_LIG Water_and_ions`** (not the default `Protein Non-Protein`), otherwise NVT grompp reports "group not found"
2. **Verify output files exist after each mdrun step**: em.gro -> nvt.gro -> npt.gro -> md.xtc; raise immediately on any missing file with the last 20 lines of the .log
3. **`gmx make_ndx` to create Protein_LIG group**: run `echo '1 | 13\nq' | gmx make_ndx` after genion and before EM to merge Protein and LIG groups
4. **Never silently ignore mdrun return values**: `retcode != 0 and "WARNING" not in stderr` is unsafe — WARNING messages can mask real errors

### Abandoned
- ESMFold: incomplete openfold dependencies — do not attempt
- Vina-GPU: segfault on RTX 4090 OpenCL kernel — set aside

---

## AutoDock-GPU Preprocessing Pipeline (verified)

User provides only PDB + ligand SMILES. Agent handles all preprocessing automatically:

```
1. ProDy (in container)     → clean PDB, remove water/HETATM, keep protein only
2. Host obabel + Gasteiger  → receptor.pdbqt
3. Auto-generate .gpf       → npts / gridcenter / box_size / atom_types (from fpocket output)
4. Host autogrid4           → receptor.maps.fld + *.map files
5. docker exec oih-autodock-gpu → autodock_gpu_128wi
```

If `mk_prepare_receptor.py` fails on unknown residues (e.g. NAG glycosylation):
- Auto-switch to `obabel` or MGLTools `prepare_receptor4.py`
- Pre-detect non-standard HETATM residues → use `--delete_residues` flag

Box coordinates come from fpocket pocket results automatically.

---

## GROMACS MD Workflow

```
pdb2gmx → editconf → solvate → genion → em → nvt → npt → md
```

GPU run command:
```bash
gmx mdrun -gpu_id 0 -nb gpu -pme gpu -update gpu -ntmpi 1 -ntomp 16
```

CPU fallback:
```bash
gmx mdrun -nb cpu -pme cpu -ntmpi 1 -ntomp 32
```

---

## Task Manager — Three-Queue Architecture

**Implemented** in `core/task_manager.py`.

| Queue | Semaphore | Tools |
|-------|-----------|-------|
| CPU queue | `Semaphore(8)` | fpocket, P2Rank, Chemprop |
| GPU queue | `Semaphore(1)` | GNINA, AutoDock, AF3, BindCraft, DiffDock, RFdiffusion, GROMACS, ESM2 — one at a time to prevent OOM |
| Degraded queue | `Semaphore(4)` | Auto-fallback to CPU+RAM when GPU VRAM insufficient |

VRAM estimates: AF3=20GB, BindCraft=16GB, DiffDock/RFdiffusion=8GB, GROMACS=6GB, ESM=6GB, GNINA=4GB, ProteinMPNN=4GB, AutoDock=2GB

Routing logic in `TaskManager._resolve_queue()`:
- CPU tools → always cpu_sem
- GPU tools → query `nvidia-smi --id=1`; if free VRAM ≥ tool requirement → gpu_sem, else → degraded_sem
- `/health` returns structured `task_queue` dict: `{pending_total, cpu_active, gpu_active, degraded_active, *_slots}`

Concurrency test: `tests/test_queue_concurrency.py --n 10` — verified peak=8 at limit, 2 queued.

---

## Skills System

- `skills_loader.py`: keyword matching → loads relevant `.md` files from `skills/` → injects into Qwen system prompt
- 13 skill docs exist at `/data/oih/oih-api/skills/`
- Verified working: detected skills + prompt injection confirmed (~3503 chars for AF3+AutoDock combo)
- RFdiffusion keywords: `antibody/nanobody/binder design/binder`

---

## Completed Work (2026-03-14)

### Router path bug — fixed in all 5 affected routers

**Pattern**: routers were stripping the provided path with `os.path.basename()` then hardcoding `/data/oih/inputs/`, ignoring paths returned by upstream tools (e.g. `fetch_pdb` returns `/data/oih/outputs/fetch_pdb/5XWR.pdb`).

**Fix** (applied to `pocket_analysis.py`, `protein_design.py`, `molecular_docking.py`, `md_simulation.py`):
```python
# Before (bug)
req.input_pdb = os.path.basename(req.input_pdb)
src = f"/data/oih/inputs/{req.input_pdb}"

# After (fix)
src = req.input_pdb if os.path.exists(req.input_pdb) \
      else f"/data/oih/inputs/{os.path.basename(req.input_pdb)}"
```
Intentional `inputs/` references (ligand.pdbqt, MDP files, BindCraft settings written by the router itself) were left unchanged.

### Agent thinking_budget default raised to 2048

In `qwen_agent.py` `_get_thinking_budget()`, the final fallback was 256 — too low for Qwen to reason about multi-step tool ordering. Changed to 2048. Higher tiers (pipeline=4096, named-tool=2048, action-verb=1024) unchanged.

### Validated end-to-end chain

`fetch_pdb → fpocket_detect_pockets → fetch_molecule → dock_ligand (GNINA)`

Tested with "fetch PDB 5XWR, find binding pockets, then dock aspirin using GNINA":
- Qwen called tools sequentially, passing `output_pdb` path correctly each round
- fpocket: 80 pockets found
- GNINA: 9 poses, best minimizedAffinity = **-5.95 kcal/mol** (Pose 3), best CNNscore = 0.672 (Pose 1)
- Output: `/data/oih/outputs/aspirin_docking_5XWR/gnina/poses.sdf`

### Proxy fix for FastAPI startup

`http_proxy=http://127.0.0.1:7890` is set in the shell environment. Always start uvicorn with:
```bash
NO_PROXY='*' no_proxy='*' nohup python -m uvicorn main:app --host 0.0.0.0 --port 8080 --workers 1 > /tmp/fastapi.log 2>&1 &
```
Without this, internal httpx calls to `192.168.31.23:8002` (Qwen) route through the proxy and get 502.

Agent log is at `/tmp/fastapi.log` (not `logs/agent.log`).

---

## Completed Work (2026-03-15)

### drug_discovery_pipeline — full implementation in `routers/pipeline.py`

End-to-end pipeline: target fetch → AF3 structure prediction → fpocket pocket detection → PubChem ligand fetch → GNINA docking → GROMACS 10 ns MD.

Key implementation details:
- `convert_af3_cif_to_pdb()`: converts AF3 `.cif` output to PDB using BioPython
- `_compute_rmsd_af3_gnina()`: RMSD comparison between AF3 predicted structure and GNINA docked pose
- `pdbfixer` used to complete missing sidechain atoms in AF3 output before docking/MD
- fpocket box coordinates extracted from `pocket_atm.pdb` (not summary file) — fixes coordinate parsing bug
- PubChem field name fallback fixed: `IUPACName` → `IUPACName` or `CID` string fallback
- 44 unit tests all passing: `tests/test_pipeline.py`

### GROMACS protein-ligand MD — two critical bugs fixed (`routers/md_simulation.py`)

**Bug 1 — CUDA not visible to GROMACS**:
- `NVIDIA_VISIBLE_DEVICES=1` is set in the container but GROMACS ignores it; `nvidia-smi` works but `gmx mdrun` reported "GPU detection failed".
- Fix: prefix all four `gmx mdrun` calls (EM, NVT, NPT, production) with `CUDA_VISIBLE_DEVICES=0`.

**Bug 2 — Ligand placed at wrong position after editconf**:
- Old code appended `LIG_GMX.gro` (from acpype) into `solv_ions.gro` AFTER `editconf -c` had already re-centered the protein box. Ligand coordinates stayed in the original PDB frame → atom overlap → `Fmax=inf` → EM crash.
- Fix: merge `protein.gro + LIG_GMX.gro → complex.gro` immediately after pdb2gmx (step 1b), before `editconf`. Also moved topology patching (`#include "LIG_GMX.itp"` + `LIG 1` in molecules) to the same pre-editconf step. `editconf -c` now centers protein + ligand together; `gmx solvate` places water correctly around the full complex.

### Full drug_discovery_pipeline — first end-to-end validation (2026-03-15)

**5XWR (COX-2) + aspirin**, complete run:
- fetch_pdb → fpocket (80 pockets) → GNINA docking (9 poses, best affinity **-5.68 kcal/mol**, CNNscore 0.661) → AF3 complex prediction (3 seeds × 5 samples) → GROMACS 0.1 ns MD
- All stages passed; trajectory at `/data/oih/outputs/test_gromacs_fix_v2/gromacs/md.xtc`
- `/api/v1/agent/chat` analysis (600s timeout): Qwen3 evaluated binding significance, inhibition likelihood, next steps

**Agent `/api/v1/agent/chat` timeout**: synchronous endpoint; Qwen3-14B with thinking_budget=2048 needs up to ~500s. Use `--max-time 600` with `curl`.

---

## 2026-03-15 Fix Log

### Chemprop container fix
- **Root cause**: Container `python3` symlink pointed to 3.10, but all packages (torch/numpy/chemprop) installed under the python3.11 path -> `No module named 'numpy'`
- **Fix**: `docker exec oih-chemprop bash -c "rm /usr/bin/python3 && ln -s /usr/bin/python3.11 /usr/bin/python3"` (lost on container rebuild, must be added to Dockerfile)
- **Router fix**: `routers/ml_tools.py` added `--accelerator cpu` to avoid GPU OOM; `task_manager.py` added `chemprop_predict` to `_CPU_TOOLS`
- **Validation**: 3-molecule prediction completed, CPU queue, finished in 5 seconds

### ADC toolchain (3 new tools)
- **freesasa**: Computes antibody SASA, filters Lys/Cys conjugation sites (SASA>40 Angstrom squared), routed to CPU queue
- **linker_select**: Filters from `data/linker_library.json` (20 clinically validated linkers), supports 4-dimension filtering by cleavable/reaction_type/compatible_payload/clinical_status, defaults to approved > clinical > research priority
- **rdkit_conjugate**: 7 ADC conjugation chemistries (maleimide_thiol/nhs_amine/hydrazone/oxime/disulfide/dbco_azide/transglutaminase), auto-detects reaction type, multi-SMARTS fallback chain + generic fallback
- E2E validation: Qwen autonomously called fetch_pdb -> freesasa -> fetch_molecule -> linker_select -> rdkit_conjugate, recommended MC-VC-PABC

### Qwen agent error retry fix
- `qwen_agent.py`: Tracks consecutive failure count per tool; >= 2 consecutive failures auto-skips and notifies Qwen to continue with subsequent steps
- Prevents the previous issue where chemprop failed 7 times causing timeout

### Service management clarification
- OIH platform: `/data/oih/miniconda/bin/python`, `main:app`, port 8080
- Gemini frontend: `/opt/oih-agent/app.py`, `app:app`, port 8001 — do not touch
- The two services use different Python environments; do not confuse them

---

## 2026-03-15 Evening (Claude Code session 2)

### ADMET model training (5 standard benchmarks)
- ESOL solubility (regression, MSE=0.54, 1128 mol) -> `/data/oih/models/admet/esol/model_0/best.pt`
- FreeSolv hydration free energy (regression, 642 mol) -> `/data/oih/models/admet/freesolv/model_0/best.pt`
- Lipophilicity logD (regression, MSE=0.43, 4200 mol) -> `/data/oih/models/admet/lipophilicity/model_0/best.pt`
- BBBP blood-brain barrier (classification, AUC=0.89, 2039 mol) -> `/data/oih/models/admet/bbbp/model_0/best.pt`
- Tox21 NR-AhR toxicity (classification, AUC=0.90, 6542 mol) -> `/data/oih/models/admet/tox21/model_0/best.pt`
- Paths written into `qwen_tools.py` tool descriptions so Qwen can select them directly

### binder_design_pipeline — full 7-step implementation (including ADC assembly)
- Step 1-2: RFdiffusion -> ProteinMPNN (existing)
- Step 3: AF3 validation — top 5 MPNN -> AF3 complex -> ipTM grading (>=0.75 high / >=0.6 low_confidence)
- Step 4: FreeSASA — nanobody surface Lys/Cys SASA>40 Angstrom squared filtering, select top 3 conjugation sites
- Step 5: Linker Select — choose maleimide(Cys) / NHS(Lys) based on site type, cleavable, compatible with MMAE
- Step 6: Fetch Payload — retrieve MMAE SMILES + MW from PubChem
- Step 7: RDKit Conjugate — antibody + linker + payload -> ADC, DAR=4
- Each step has try/except; single-step failures marked `partial: true` without blocking subsequent steps
- `dry_run=true` returns full mock structure with `adc_design` field
- 5-second interval between AF3 tasks to avoid GPU OOM, timeout 1800s
- `BinderDesignPipelineRequest`: `num_designs` (alias), `dry_run`, `hotspot_residues: list[str]`

### GROMACS protein-ligand MD fixes (3 critical bugs)
1. **Dynamic tc-grps detection**: Parse `index.ndx` after `make_ndx` to find actual merged group names (e.g. `Protein_UNL`), no longer hardcoded `Protein_LIG`
2. **Deferred MDP generation**: NVT/NPT/MD MDP files written inside `_run()` after `make_ndx`, ensuring correct tc-grps
3. **Per-step file validation**: em.gro -> nvt.gro -> npt.gro -> md.xtc; raise immediately on missing file with last 20 lines of .log
4. **Deprecated parameter removal**: `dispdivcorr` -> `DispCorr`, removed `ns_type = grid` (deprecated in GROMACS 2024)

### RAG knowledge base initialization
- 6 rounds of ADC literature retrieval (HER2/CD30/TROP2/DAR/Linker), 29 papers indexed in ChromaDB
- Local embedding query validation passed (BAAI/bge-m3 ONNX)

### dashboard.html frontend (single file, 760 lines)
- Dark glassmorphism UI, Space Grotesk + IBM Plex Mono, canvas particle background
- Top bar: natural language input -> POST `/api/v1/agent/chat`, 4 example prompts
- Left panel: Agent Pipeline Timeline (2s polling, tool node animation)
- Right panel: 9 dynamic tabs (Chat/Protein3D/Molecule2D/Docking/ADMET/Sites/Linker/Payload/ADC Assembly/MD)
- Agent Chat: bottom input box + markdown rendering + 600s timeout
- Service address: `http://192.168.31.23:9099/dashboard.html`

### sync_claude_to_skills.py documentation sync script
- 5 sync targets: skills/*.md + routers/*.py + qwen_tools.py + skills_loader + CLAUDE.md (read-only source)
- Idempotent and repeatable; uses marker wrappers (`AUTO_SYNC_FROM_CLAUDE_MD` / `SYNC_NOTES`)
- Router comment safety: newline/quote sanitization, does not break Python syntax
- Usage: `python scripts/sync_claude_to_skills.py --apply`

### ADC toolchain validation
- freesasa / linker_select / rdkit_conjugate fully registered in all three locations (router + tool_definitions + TOOL_MAP)

---

## Current Outstanding Work

1. **100 ns MD in progress** — task_id `2add68d5`, EM passed, NVT/NPT/MD in progress (third submission after tc-grps fix)
2. **MM/PBSA** — post-process md.xtc after 100ns MD completes to compute binding free energy
3. **AutoDock pipeline integration** — complete `pipeline.py` autodock preprocessing chain
4. ~~**Frontend**~~ — dashboard.html done, Agent Chat connected, real-time task tracking working
5. ~~**RAG**~~ — completed
6. ~~**chemprop Dockerfile**~~ — Dockerfile has `ln -sf`, container python3.11 working
7. ~~**ADMET models**~~ — 5 standard benchmark models trained
8. ~~**RAG knowledge base initialization**~~ — 29 ADC papers indexed
9. **End-to-end testing** — validate dashboard dynamic tab unlocking with real tasks (protein structure/docking/ADMET/ADC)
10. **Chemprop container image rebuild** — Dockerfile fixed, waiting for idle time to `docker build` to persist in image

---

## pip / Docker Rules

- pip installs belong in **Dockerfiles**, not in running containers (lost on rebuild)
- Use `pip install --break-system-packages` when installing on host
- Verify actual file state with grep/cat before assuming prior work was done

---

## Testing a Tool Manually

```bash
# Check all services alive
curl -s http://localhost:8080/health
curl -s http://localhost:8002/v1/models

# Check compute containers running
docker ps --format "table {{.Names}}\t{{.Status}}"

# Send agent request
curl -X POST http://localhost:8080/api/v1/agent \
  -H "Content-Type: application/json" \
  -d '{"message": "fetch PDB 5XWR and find binding pockets", "session_id": "test-001"}'

# Tail agent logs
tail -f /data/oih/oih-api/logs/agent.log
```

---

## Critical Environment Notes (hard-won lessons — check here first when troubleshooting)

### Two independent FastAPI services (do not confuse them!)

| Service | Entry | Port | Python | Management |
|---------|-------|------|--------|------------|
| OIH Platform | `/data/oih/oih-api/main.py` | 8080 | `/data/oih/miniconda/bin/python` | systemd `oih-api` |
| Gemini Frontend | `/opt/oih-agent/app.py` (`app:app`) | 8001 | `/opt/oih-agent/fastapi/bin/python3` | Manual |

**OIH Platform** (this project):
- Restart (use this command; sudo requires password in non-interactive terminals, so avoid systemctl):
  ```bash
  NO_PROXY='*' no_proxy='*' nohup /data/oih/miniconda/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8080 --workers 1 > /tmp/fastapi.log 2>&1 &
  ```
- View logs: `tail -f /tmp/fastapi.log` or `journalctl -u oih-api -n 30 --no-pager`
- pip install: `/data/oih/miniconda/bin/pip install <package>`
- Python: `/data/oih/miniconda/bin/python`
- Verify: `curl -s --noproxy '*' http://localhost:8080/health` -> returns `{"status":"ok", "containers":{...}}`
- Do not use anything under `/opt/oih-agent/fastapi/` — that is the Gemini frontend's venv

**Gemini Frontend** (do not touch!):
- Process: `/opt/oih-agent/fastapi/bin/uvicorn app:app --host 127.0.0.1 --port 8001`
- Verify: `curl -s --noproxy '*' http://127.0.0.1:8001/health` -> returns `{"ok":true, "model":"gemini-2.5-flash"}`
- Do not kill, restart, modify, or install packages into its venv

### Proxy
- Clash port 7890
- localhost requests must include --noproxy '*' or you get 502
- curl to internal network: curl -s --noproxy '*' http://127.0.0.1:8001/...

### GPU Rules
- All containers use NVIDIA_VISIBLE_DEVICES=1; always use device=0 inside containers
- JAX containers (AF3, BindCraft) must not mount host cuda lib64 paths

### Three-location registration rule (new tools must be changed in all three places)
1. Add route in the corresponding file under routers/
2. Add tool definition in tool_definitions/qwen_tools.py
3. Add entry in TOOL_MAP in qwen_agent.py

## Claude Code Safety Prompt Handling
The following command patterns trigger security confirmations — select Yes:
- `nohup ... > /tmp/fastapi.log 2>&1 &` -> Yes (standard background launch)
- `sudo systemctl ...` -> Yes
- `/data/oih/miniconda/bin/pip install ...` -> Yes (already configured to not ask)

---

## oih-api.service Pitfall
- systemd has oih-api.service (enabled), which auto-starts on server reboot
- However, normally started manually via nohup — the conflict means systemd respawns the process after kill
- Always run `systemctl stop oih-api` before restarting
- For production deployment, consider writing the nohup command into the service file for unified management

---

## 2026-03-16 Completed

### Frontend
- dashboard.html added attachment upload UI (794->959 lines)
  - Paperclip button popup menu (PDB upload / FASTA paste / SMILES paste)
  - Attachment tag preview (above input box, removable)
  - PTM detection result card (categorized display)
  - Send logic extended: includes pdb_content/fasta_sequence/smiles/filename
- Cancel task button (next to running tasks) + clear conversation button

### Backend
- AgentChatRequest extended with 4 Optional fields (pdb_content/fasta_sequence/smiles/filename)
- detect_ptm() function: auto-detects glycosylation/phosphorylation/disulfide bonds/acetylation
- generate_tool_inputs() function: auto-generates af3_input.json / gromacs_ptm_notes.json / adc_input.json
- is_simple_message() dynamic thinking_budget (greetings=0, complex tasks=1024+)
- /api/v1/tasks/{task_id}/cancel endpoint

### Knowledge files
- skills/PTM_UPLOAD_PARADIGM.md: Qwen PTM decision paradigm
- docs/paper/OIH_manuscript.md: bioRxiv manuscript draft framework

### Test validation
- Test 1 plain text hello: 200, turns=2
- Test 2 SMILES aspirin: chemprop auto-invoked, Tox21=0.0154
- Test 3 PDB+PTM: af3_input.json (NAG glycosylation) + gromacs_ptm_notes.json (disulfide bonds) generated correctly
- is_simple_message: hi -> budget=0, predict ADMET -> budget=1024+

### System status
- 100ns MD: 22.74ns/100ns (22.7%), trajectory 24GB, performance 19.2ns/day, exited normally
- All 11 containers running

### Remaining work
- Case 2: Nanobody design experiment (Fig 4 data)
- Case 3: HER2-ADC design experiment (Fig 5 data)
- Fig 6: Tool runtime statistics
- MD animation rendering (MDAnalysis -> gif)
- 100ns MD resubmission (resume from 22.74ns checkpoint)
- MM/PBSA binding free energy calculation

---

## 2026-03-17-18 Completed

### binder_design_pipeline full 7-step + ADC
- Step 1-3: RFdiffusion -> ProteinMPNN -> AF3 validation (ipTM >=0.75 high / >=0.6 low_confidence)
- Step 4-7: FreeSASA conjugation sites -> Linker selection -> MMAE payload -> RDKit conjugation (DAR=4)
- Each step has try/except; single-step failures marked partial without blocking subsequent steps
- `pdb_id` parameter support (auto fetch_pdb)
- `dry_run` parameter support
- `_parse_mpnn_fasta()` FASTA parsing fix
- `hotspot_residues` changed to `list[str]`
- HER2 nanobody ADC v3 submitted (task: 6a34f32f)

### Task persistence
- `core/task_manager.py`: writes to `data/tasks/{task_id}.json` on state changes
- Scans directory on startup to recover historical tasks (running/pending marked as failed)
- Fields: task_id, tool, status, progress, progress_msg, result, error, created_at, updated_at

### Cancel endpoint
- `DELETE /api/v1/tasks/{task_id}`: cancels pending/running tasks
- `core/task_manager.py` cancel_task extended to support running status
- Completed/failed tasks return 400

### Dashboard Results Hub
- Persistent Results Hub tab, displays all historical tasks grouped by tool
- Left: tool tree + Right: tool-type-specific rendering (12 types)
- Path fields with copy buttons
- Pipeline results inline: 6 stat cards + AF3 table + ADC summary + download buttons

### Dashboard Pipeline Preview
- Frontend keyword matching on input, shows pipeline preview (6 pipeline types)
- Horizontal flowchart: emoji + tool name + estimated time

### Dashboard subtask collapsing
- Pipeline-triggered subtasks grouped by created_at time window
- Left panel shows only pipeline card + subtask summary (e.g. ProteinMPNN x20, 18 done, 2 pending)

### Dashboard Tab highlighting
- running -> cyan pulse animation, completed -> green for 3 seconds

### Dashboard 3Dmol.js fix
- Data validation + addModel try/catch + user-friendly error messages

### Qwen sync
- qwen_tools.py: binder_design_pipeline description updated (7 steps + pdb_id + adc_design)
- skills_loader.py: added ADC keyword mappings
- ADC_WORKFLOW.md: complete 7-step workflow + conjugation chemistry rules + DAR documentation

---

## 2026-03-18 Completed

### AF3 timeout fix — pipeline.py `_wait_for_af3_task()`
- **Root cause**: v3 task had all 5 AF3 designs fail. rank1 timed out at 1800s (but actually finished, ipTM=0.48); rank2-5 routed to DEGRADED queue, OOM exit 1
- **Fix 1**: Added `_wait_for_af3_task()` infinite-wait function, polls every 30s, only declares failure on OOM/exit1/cancelled/10 consecutive identical errors
- **Fix 2**: AF3 calls in binder_pipeline + drug_discovery_pipeline switched to `_wait_for_af3_task()`
- **Fix 3**: `_wait_for_task()` supports `timeout <= 0` for infinite waiting

### AF3 excluded from DEGRADED queue — task_manager.py `_NO_DEGRADED_TOOLS`
- Added `_NO_DEGRADED_TOOLS = {"alphafold3", "bindcraft"}`
- In `_resolve_queue()`, these tools retry every 60s when VRAM is insufficient, waiting until enough VRAM is available before entering GPU queue
- No longer degrades to DEGRADED queue causing OOM crash

### Static file /outputs mount — main.py
- `app.mount("/outputs", StaticFiles(directory="/data/oih/outputs"), name="outputs")`
- dashboard.html 3D viewer and download button paths changed to `/outputs/...` (previously `/static/outputs/...` returned 404)
- Validation: val_0 CIF file HTTP 200

### HER2 ADC v4 submission
- task_id: `b07f91e0-004f-4ec8-ad00-d7a84f826859`
- Parameters: 1N8Z, hotspot S310/T311/Q313/L317, 10 designs
- AF3 now waits indefinitely for completion, no timeout

### Knowledge distillation system (2026-03-18)
- `skills/SELF_DIAGNOSIS_WORKFLOW.md` — self-diagnosis handbook (AF3/VRAM/RFdiffusion/GROMACS)
- `qwen_tools.py` system prompt — added autonomous diagnosis-repair instructions (4-step loop: diagnose -> fix -> rerun -> report)
- `data/distillation/` — distillation training data directory
- `scripts/collect_distillation_data.py` — auto-collects task cases
- Current case count: **67** (4 manual + 35 historical extraction + 28 auto-collected)
- Category coverage: gpu(6) / container(3) / tool(4) / pipeline(16) / proxy(2) / dashboard(1) / auto(28) / manual(4) / abandoned(2) / reference(1)
- Next step: LoRA fine-tuning after accumulating 100 cases

---

## 2026-03-20 Completed

### pocket_guided_binder_pipeline refactor — multi-dimensional pocket scoring system (14 steps)

**Old pipeline** (11 steps): P2Rank top pocket -> directly select top 6 residues -> DiffDock blind docking cross-validation -> RFdiffusion
**New pipeline** (14 steps): multi-dimensional scoring + Qwen selection -> DiffDock only as druggability reference

**Pocket scoring formula**:
```
composite = p2rank_prob × 0.2 + sasa_score × 0.2 + conservation × 0.2 + rag_score × 0.3 + electrostatics × 0.1
```

| Dimension | Calculation method | Weight |
|-----------|-------------------|--------|
| P2Rank | P2Rank ML probability (normalized to 0-1) | 0.2 |
| SASA | FreeSASA on target PDB, mean_SASA / 150 per pocket residue, capped at 1.0 | 0.2 |
| Conservation | B-factor as flexibility proxy: 1 - mean_normalised_bfactor (low B = rigid = conserved = good target) | 0.2 |
| RAG | Literature residue overlap: 0 (none) / 0.5 (1-2 residues) / 1.0 (3+ residues) | 0.3 |
| Electrostatics | Fraction of charged residues (K/R/E/D/H) in pocket | 0.1 |

**14-step detailed pipeline**:
1. fetch_pdb — download target structure
2. RAG literature search — `{target} {pdb_id} binding site epitope domain experimental validation`
3. fpocket + P2Rank — parallel pocket detection (top 5)
4. FreeSASA per pocket — target PDB surface exposure
5. B-factor conservation + electrostatics — conservation and electrostatic analysis
6. Composite scoring + Qwen structural biologist pocket selection — returns pocket_id + 6 hotspots + selection rationale
7. DiffDock — druggability reference only for selected pocket (labeled `small molecule druggability reference`)
8. RFdiffusion — binder backbone design using Qwen-selected hotspots
9. ProteinMPNN — sequence design
10. AF3 validation — ipTM >= 0.6
11-14. ADC assembly — FreeSASA conjugation sites -> Linker -> MMAE -> RDKit (DAR=4)

**New helper functions** (`routers/pipeline.py`):
- `_parse_p2rank_residues()` — P2Rank residue parsing (extracted from inline code)
- `_compute_bfactor_conservation()` — B-factor conservation scoring
- `_compute_electrostatics_from_pdb()` — charged residue fraction
- `_compute_sasa_score_for_pocket()` — pocket SASA scoring
- `_compute_rag_score()` — literature residue overlap scoring
- `_extract_residue_numbers_from_text()` — regex extraction of residue numbers from literature
- `_rag_search_pocket_context()` — direct call to `HybridRetriever.retrieve()` (bypasses HTTP)
- `_compute_freesasa_per_residue()` — direct call to FreeSASA C library (bypasses task)
- `_qwen_select_pocket()` — Qwen3-14B pocket selection (returns JSON)

**New return fields**:
- `pocket_scores` — 5-dimensional scores per pocket + composite
- `selected_pocket` — `{id, center, hotspots, reason, composite_score}`
- `diffdock_reference` — `{label, confidence, pose_path}`

### AF3 PDB parsing fix — 3-layer fallback
- **Cause**: CIF->PDB conversion failure left `af3_pdb=None` -> `"No PDB for FreeSASA"` error
- **Fix**:
  1. `convert_af3_cif_to_pdb()` (gemmi + pdbfixer full atom completion)
  2. Fallback: plain `gemmi.write_pdb()` (skip pdbfixer)
  3. Fallback: scan AF3 directory and seed subdirectories for existing `.pdb` files
  4. Already-converted `_for_sasa.pdb` files skip redundant conversion

### Pocket scoring runtime — 3 bug fixes (2026-03-21)

**Bug 1 — RAG self-HTTP deadlock**:
- `_rag_search_pocket_context()` called local RAG endpoint via `httpx.get("http://127.0.0.1:8080/api/v1/rag/search")`
- Single uvicorn worker (`--workers 1`) blocked on pipeline coroutine, unable to process RAG request -> deadlock
- **Fix**: Changed to direct call `from retrieval.rag_router import get_retriever; retriever.retrieve(...)` bypassing HTTP

**Bug 2 — Qwen3 thinking mode returns null content**:
- Qwen3-14B has thinking mode enabled by default; `content` field is `null`, reasoning is in the `reasoning` field
- `_qwen_select_pocket()` got None for `content` -> subsequent `re.sub()` raised TypeError, silently caught by except
- **Fix**: Added `"chat_template_kwargs": {"enable_thinking": False}` to request body to disable thinking, returning JSON directly
- **Defense**: Added `content = msg.get("content") or ""` null guard + `exc_info=True` for full traceback logging

**Bug 3 — RAG residue number matching failure**:
- RAG extracts pure numeric residue numbers from literature `['42', '310']`
- P2Rank residue format includes chain prefix `['A42', 'A310']`
- Old code attempted to add chain prefix to RAG residues, but the logic was unreliable
- **Fix**: `_compute_rag_score()` strips leading alphabetic characters from both sides during comparison, unifying to pure numeric comparison

### RAG ChromaDB knowledge base configuration and expansion

**ChromaDB path**: `/data/oih/knowledge/chroma` (collection: `oih_knowledge`)
- Config location: `/data/oih/retrieval/config.py` -> `cfg.chroma_path`
- Environment variable: `OIH_CHROMA_PATH` (already points to the correct path by default)
- `/data/oih/retrieval/chroma_db` is an empty database, not in use
- Embedding: BAAI/bge-m3 (ONNX)

**Document count**: 80 (50 existing ADC papers + 26 PubMed HER2 structural papers + 4 curated residue entries)

**HER2 residue-level data** (4 curated entries):
- `curated_her2_pertuzumab_residues` — Domain II dimerization arm: S267, L269, T271, K273, E280, G282 (PDB 1S78)
- `curated_her2_trastuzumab_residues` — Domain IV: K505, F506, P557, E558, D560, E561 (PDB 1N8Z)
- `curated_her2_domain_hotspots` — 4 therapeutic pocket summary + Kd values
- `curated_her2_2a91_pocket1` — P2Rank pocket 1 residues mapped to Domain I/III interface

**Important**: PubMed abstracts typically do not contain residue numbers; they need to be extracted from full text or manually curated. `rag_score > 0` requires residue-level data in the knowledge base. For new targets, curated entries must be added following this pattern.

**Ingestion methods**:
- PubMed batch: `POST /api/v1/rag/local/add-papers` + PMID list
- Manual curated: `chromadb.PersistentClient(path).get_collection('oih_knowledge').upsert()`
- PDF upload: `POST /api/v1/rag/local/upload-pdf` (requires `OIH_PDF_INGEST=true`)

### BindCraft parallel path (2026-03-21)
- `pocket_guided_binder_pipeline` expanded from 14 steps to **15 steps**
- Step 8 split into 8a (RFdiffusion) + 8b (BindCraft), executed in parallel with `asyncio`
- BindCraft failure does not block pipeline (`try/except`, marked `status: failed`)
- BindCraft results: sequences extracted (`_extract_sequence_from_pdb`), up to 2 merged into AF3 candidate list
- AF3 validation selects top 5 from merged candidates (MPNN + BindCraft)
- BindCraft parameters: `num_designs=10` (cannot be <10, pydantic ge=10), uses `default_4stage_multimer.json`

---

## 2026-03-21 Completed

### DiscoTope3 integration (14th container oih-discotope3)
- Registered in all 3 locations, B-cell epitope prediction
- Validation: 1N8Z HER2 -> 195 epitopes / 1015 residues
- DT3 raw score range 0.001-0.5 (varies by structure), **do not use fixed threshold 0.7**
- `calibrated_score_epi_threshold` default 0.5

### IgFold integration (15th container oih-igfold)
- Antibody/nanobody sequence -> fast 3D structure prediction (~2s/seq GPU)
- Based on proteinmpnn:latest, igfold 0.4.0 + antiberty
- Outputs pRMSD (not pLDDT), conversion: `pseudo_plddt = 100 - prmsd * 20`
- `do_renum=False` (anarcii vs anarci package name conflict)

### ESM2 new tools
- `esm2_score_sequences`: pseudo-perplexity scoring (PPL<15 = passing sequence)
- `esm2_mutant_scan`: ESM-1v single-point mutation scan (delta-delta-G proxy)
- ESM-1v model 7.3GB, requires download on first load

### IEDB + SAbDab integrated into RAG
- IEDB: `https://query-api.iedb.org/bcell_search` PostgREST API
- SAbDab: full TSV download + antigen name filtering (8MB, 20K rows)
- `extract_antigen_name()` regex extraction of target name
- RAG gather total timeout 30s, PubMed efetch 10s+15s timeout protection

### pocket_guided_binder_pipeline upgraded to 16-step 6D scoring
- Scoring formula: `p2rank(0.15) + sasa(0.15) + conservation(0.15) + rag(0.25) + electrostatics(0.10) + epitope(0.20)`
- DiscoTope3 runs in parallel with fpocket/P2Rank (step 3)
- `_compute_epitope_score_for_pocket()`: DT3 epitope fraction within 8 Angstrom of pocket residues
- ESM2 PPL filter -> IgFold pLDDT filter -> AF3 (funnel)

### known_epitope_override (hard-won lesson)
- Trigger condition: DT3 high-scoring residues intersection with RAG literature residues >= 2
- DT3 threshold: **adaptive = max(top 20% score, 0.10)**, do not use fixed 0.7
- Numbering alignment: PDB and UniProt numbering may have offset, **auto-detect offset (0, +/-22, +/-23)**
- Fuzzy matching: `abs(dt3_num + offset - rag_num) <= 3`
- **Spatial clustering**: `_cluster_hotspots()` centroid distance <= 15 Angstrom, **max 5 hotspots**
- Consequence of not clustering: 8 scattered hotspots -> RFdiffusion 30+ min per design -> timeout

### RFdiffusion large target considerations
- HER2 ECD 506 residues -> 20-40 min per design (normal ~2 min)
- **timeout must be 7200s** (not 3600s)
- **num_designs use 10** (not 20, too slow)
- **hotspots must be spatially clustered** (<= 15 Angstrom centroid), scattered ones are extremely slow
- Recover already-generated PDBs after timeout (`_wait_for_task` try/except + scan output dir)

### Model cache volume mounts
| Container | Host path | Container path |
|-----------|-----------|----------------|
| oih-esm | /data/oih/model_cache/esm_torch_hub | /root/.cache/torch/hub |
| oih-discotope3 | /data/oih/model_cache/torch/hub | /root/.cache/torch/hub |
| oih-discotope3 | /data/oih/model_cache/discotope3_models | /app/discotope3/models |
| oih-igfold | /data/oih/model_cache/igfold | /root/.cache |

### Current tool count: 30, container count: 15
- 5 new tools: discotope3_predict, igfold_predict, esm2_score_sequences, esm2_mutant_scan, extract_interface_residues
- 2 new containers: oih-discotope3, oih-igfold
- RAG 5 sources: PubMed + bioRxiv + IEDB + SAbDab + LocalDB

---

## Target Tier System (2026-03-22)

Always classify target tier BEFORE hotspot selection in `pocket_guided_binder_pipeline`.

### Tier Definitions
| Tier | Method | Reliability | When |
|------|--------|-------------|------|
| 1 | structural_database | Highest | Known antibody-antigen co-crystal exists in PDB |
| 2 | rag_guided | Moderate | Homologous structure found via RAG literature |
| 3 | computational_prediction | Lower | Novel target — DiscoTope3 + IEDB + RAG override |

### Known Complexes Registry (`KNOWN_COMPLEXES` in pipeline.py)
| Target | Complex PDB | Receptor Chain | Ligand Chains | Source |
|--------|-------------|----------------|---------------|--------|
| HER2/ERBB2 | 1N8Z | C | A,B | trastuzumab |
| PD-L1 | 5XXY | A | B | atezolizumab |
| EGFR | 1YY9 | A | B | cetuximab |
| VEGF | 1BJ1 | V | A,B | bevacizumab |
| CD20 | 6Y4J | A | H,L | rituximab |
| TNF | 3WD5 | A | H,L | adalimumab |

### Decision Flow
1. `_classify_target_tier(target_name, pdb_id, rag_result)` → tier_info
2. Tier 1 → `extract_interface_residues` from known complex → hotspots (skip steps 3-6)
3. Tier 3 → fpocket + P2Rank + DiscoTope3 → 6D scoring OR epitope_override → hotspots

### Key Insight
- Tier 1 hotspots come from **experimentally validated** antibody interfaces
- Tier 3 hotspots come from **computational prediction** (often scattered, low pLDDT)
- HER2 = Tier 1 → fetch 1N8Z → extract C-chain interface → clustered hotspots → better RFdiffusion

### Automatic Hotspot Clustering Rules (universal for all targets)
- All targets undergo automatic spatial clustering of hotspots (`_multi_cluster_hotspots`) before submitting to RFdiffusion
- Distance threshold: 15 Angstrom (CA-CA distance); residues within 15 Angstrom are grouped together
- Maximum 5 residues per cluster
- Multiple clusters → submit to RFdiffusion separately → merge results → unified MPNN → ESM2 → AF3
- BindCraft runs only on the first (largest) cluster
- Single cluster (3 or fewer hotspots, or all within 15 Angstrom) → no splitting
- Applicable to: CD36/EGFR/CD20/HER2 multi-domain/any large protein
- CD36 lesson: 5 scattered hotspots (A164-A400 spanning 236 residues) → ipTM=0.43, all failed

### pLDDT Quality Expectations by Tier
- Tier 1 hotspots → RFdiffusion backbone quality: pLDDT > 70 expected
- Tier 3 hotspots → RFdiffusion backbone quality: pLDDT 40-70, need more designs

### AF3 Antigen Domain Registry (`DOMAIN_REGISTRY` in pipeline.py)
- During AF3 validation, the antigen sequence is truncated by structural domain to avoid full-length sequences reducing ipTM accuracy
- Known proteins are truncated at UniProt/Pfam domain boundaries with padding=30aa
- When adding new targets, both `DOMAIN_REGISTRY` and `KNOWN_COMPLEXES` must be updated simultaneously
- Multiple druggable domains → run AF3 separately for each, select the highest ipTM
- `num_seeds=3` is used for binder_design_pipeline AF3 validation (speed/accuracy trade-off)

### GPU VRAM Limits (RTX 4090, 44GB)
- GPU Semaphore = 1 (only one GPU task at a time to prevent OOM)
- AF3: ~20GB (requires exclusive access)
- RFdiffusion: ~8GB
- BindCraft: ~16GB
- DiffDock: ~8GB
- DiscoTope3: ~6GB (ESM-IF1 1.6GB + XGBoost)
- Two large models running concurrently = guaranteed OOM

### 5LGD (CD36) Important Notes
- PDB contains chain A (CD36) + chain B (PfEMP1 malaria protein)
- Pipeline must specify `chains="A"` to filter; otherwise DiscoTope3 reports "No valid PDB"
- CD36 residue range: 35-434 (400 residues)

### PeSTo PPI Interface Prediction (deployed 2026-03-23)
- Replaces P2Rank + DiscoTope3 in binder design scoring
- ROC AUC 0.92 (vs MaSIF-site 0.80)
- Deployed in oih-proteinmpnn container: /app/pesto/
- Must extract target single chain for complex PDBs
- PPI-optimized scoring: rag(0.30)+pesto(0.25)+conservation(0.20)+sasa(0.10)+electrostatics(0.15)

### Key finding: B-cell epitope ≠ PPI interface
- DiscoTope3 predicts immunogenicity, not optimal binder sites
- CD36 proof: DiscoTope3 A397-400 → 0/10 pass; PeSTo A187-194 → testing
- Baker lab never used epitope tools for hotspot selection
- RAG-first: literature PPI interfaces always override computational epitope predictions

### 2026-03-24 Session Summary
- PeSTo deployed (tool #32): ROC AUC=0.92, replaces P2Rank+DiscoTope3
- ipSAE deployed: reveals DT3 ipTM=0.43 is false positive (ipSAE=0.000)
- Scoring formula: 6D → PPI-optimized (rag+pesto+conservation+sasa+electrostatics)
- RAG: 2-layer search (PPI interface > epitope fallback)
- Scheduler: pipeline→CPU, GPU sem=3, RFdiff 3-parallel
- DOMAIN_REGISTRY: +Nectin-4, +TROP2, +TrkA, CD36 split to 3 sub-domains
- DiffDock removed from binder pipeline
- 7-target benchmark: PeSTo done, AF3 running for 5 new targets
- Distillation: 81 cases (target 100 for LoRA)

## 2026-03-24 CRITICAL BUG FIX: MPNN chains_to_design
RFdiffusion binder_design output: chain A = binder (shorter), chain B = target (longer).
MPNN previously hardcoded designed_chains=['A'], which happened to be correct for HER2 (original chain C) but incorrectly designed the target chain for all other targets (original chain A).
Fix: pipeline.py now dynamically detects the shortest chain as binder_chain.
HER2 (ipTM=0.86) data remains valid. The other 5 targets need to be re-run with the fixed code.

## CRITICAL: KNOWN_COMPLEXES chain assignments must be verified from the PDB file
**Never guess ligand_chains.** Before adding a new target to KNOWN_COMPLEXES, always run:
```python
import gemmi
st = gemmi.read_structure('path/to/PDB.pdb')
for c in st[0]:
    nres = sum(1 for r in c if r.entity_type == gemmi.EntityType.Polymer)
    print(f'Chain {c.name}: {nres}aa')
```
Then manually determine: which chain is the target (usually the longest), and which chains are the ligand (antibody Fab = two Heavy+Light chains).

**Verified KNOWN_COMPLEXES:**
| PDB | Target | receptor_chain | ligand_chains | Verified |
|-----|--------|---------------|---------------|----------|
| 1N8Z | HER2 | C(581aa) | [A(214),B(220)] | 2026-03-25 |
| 5XXY | PD-L1 | A(99aa) | [H(208),L(211)] | 2026-03-25 |
| 1YY9 | EGFR | A(613aa) | [C(211),D(220)] | 2026-03-25 |
| 1BJ1 | VEGF | V(94aa) | [H(218),L(213)] | 2026-03-25 |
| 3WD5 | TNF | A(152aa) | [H(214),L(213)] | 2026-03-25 |

**Lesson learned:** 1YY9 EGFR was initially written with ligand_chains=["B"] (guessed), but cetuximab actually uses chains C+D. This caused extract_interface to extract the wrong interface, wasting the entire v2 pipeline run.

## CRITICAL: ADC conjugation sites must be on the binder chain
When FreeSASA analyzes AF3 complexes, it must only examine Lys/Cys on the binder chain; the antigen chain must not be selected.
ADC architecture: binder-linker-payload, where the payload is attached to the binder.
Lys residues on the antigen chain (e.g. HER2 B:K603) must never be used as conjugation sites.
