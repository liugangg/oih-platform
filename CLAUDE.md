# 启动规则（每次会话开始必须执行）
1. 找到并读取最新 session log：ls -t docs/*.md | head -1，然后 Read 全文
2. 汇报：待完成事项、注意事项、上次遗留问题
3. 检查服务状态：curl -s --noproxy '*' http://localhost:8080/health
4. 确认后再接受任务

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

### Chemprop — 三个必须遵守的规则（血泪经验）

1. **Dockerfile 必须有**：`RUN ln -sf /usr/bin/python3.11 /usr/bin/python3`（容器内 python3 默认指向 3.10，但所有包装在 3.11 路径下）
2. **predict 调用必须加**：`--accelerator cpu`（小批量走 GPU 会 OOM），归属 `_CPU_TOOLS` 队列
3. **`--devices 1` = "使用 1 个 GPU 设备"**，不是 device index=1。train 用 `--accelerator gpu --devices 1`，predict 用 `--accelerator cpu`

**ADMET 模型路径**（MoleculeNet 标准数据集训练，30 epochs）：
```
/data/oih/models/admet/esol/model_0/best.pt          — 溶解度 logS（回归，MSE=0.54）
/data/oih/models/admet/freesolv/model_0/best.pt      — 水化自由能（回归）
/data/oih/models/admet/lipophilicity/model_0/best.pt  — 脂溶性 logD（回归，MSE=0.43）
/data/oih/models/admet/bbbp/model_0/best.pt          — 血脑屏障（分类，AUC=0.89）
/data/oih/models/admet/tox21/model_0/best.pt         — Tox21 NR-AhR 毒性（分类，AUC=0.90）
```

### GROMACS — 蛋白配体体系必须注意的坑

1. **tc-grps 必须用 `Protein_LIG Water_and_ions`**（不是默认的 `Protein Non-Protein`），否则 NVT grompp 报 "group not found"
2. **每步 mdrun 之后必须验证输出文件存在**：em.gro → nvt.gro → npt.gro → md.xtc，任何一步缺失立刻 raise 并附 .log 最后20行
3. **`gmx make_ndx` 创建 Protein_LIG 组**：在 genion 之后、EM 之前运行 `echo '1 | 13\nq' | gmx make_ndx` 合并 Protein 和 LIG 组
4. **不要静默忽略 mdrun 返回值**：`retcode != 0 and "WARNING" not in stderr` 这种判断不安全，WARNING 可能掩盖真实错误

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
| GPU queue | `Semaphore(3)` | GNINA, AutoDock, AF3, BindCraft, DiffDock, RFdiffusion, GROMACS, ESM2 |
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
- RFdiffusion keywords: `antibody/抗体/nanobody/纳米抗体/binder design/binder`

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

## 2026-03-15 修复记录

### chemprop 容器修复
- **根因**：容器内 `python3` symlink 指向 3.10，但所有包（torch/numpy/chemprop）装在 python3.11 路径下 → `No module named 'numpy'`
- **修复**：`docker exec oih-chemprop bash -c "rm /usr/bin/python3 && ln -s /usr/bin/python3.11 /usr/bin/python3"`（容器重建会丢失，需写入 Dockerfile）
- **router 修复**：`routers/ml_tools.py` 加 `--accelerator cpu`，避免 GPU OOM；`task_manager.py` 把 `chemprop_predict` 加入 `_CPU_TOOLS`
- **验证**：3 分子预测 completed，CPU queue，5 秒完成

### ADC 工具链（新增 3 个工具）
- **freesasa**：计算抗体 SASA，筛选 Lys/Cys 偶联位点（SASA>40Å²），走 CPU queue
- **linker_select**：从 `data/linker_library.json`（20 条临床验证 linker）筛选，支持 cleavable/reaction_type/compatible_payload/clinical_status 四维过滤，默认按 approved > clinical > research 优先
- **rdkit_conjugate**：7 类 ADC 偶联化学（maleimide_thiol/nhs_amine/hydrazone/oxime/disulfide/dbco_azide/transglutaminase），自动检测反应类型，多 SMARTS fallback chain + 通用降级
- E2E 验证：Qwen 自主调用 fetch_pdb → freesasa → fetch_molecule → linker_select → rdkit_conjugate，推荐 MC-VC-PABC

### Qwen agent 错误重试修复
- `qwen_agent.py`：记录每个工具连续失败次数，≥2 次自动跳过并通知 Qwen 继续后续流程
- 避免之前 chemprop 失败 7 次导致超时的问题

### 服务管理明确
- OIH 平台：`/data/oih/miniconda/bin/python`，`main:app`，端口 8080
- Gemini 前端：`/opt/oih-agent/app.py`，`app:app`，端口 8001，不要动
- 两个服务用不同 Python 环境，不要混淆

---

## 2026-03-15 晚间完成（Claude Code session 2）

### ADMET 模型训练（5 个标准 benchmark）
- ESOL 溶解度（回归，MSE=0.54，1128mol）→ `/data/oih/models/admet/esol/model_0/best.pt`
- FreeSolv 水化自由能（回归，642mol）→ `/data/oih/models/admet/freesolv/model_0/best.pt`
- Lipophilicity logD（回归，MSE=0.43，4200mol）→ `/data/oih/models/admet/lipophilicity/model_0/best.pt`
- BBBP 血脑屏障（分类，AUC=0.89，2039mol）→ `/data/oih/models/admet/bbbp/model_0/best.pt`
- Tox21 NR-AhR 毒性（分类，AUC=0.90，6542mol）→ `/data/oih/models/admet/tox21/model_0/best.pt`
- 路径已写入 `qwen_tools.py` 工具描述，Qwen 可直接选用

### binder_design_pipeline ✅ 完整 7-step 实现（含 ADC 构建）
- Step 1-2: RFdiffusion → ProteinMPNN（原有）
- Step 3: AF3 验证 — top5 MPNN → AF3 复合物 → ipTM 分级（≥0.75 high / ≥0.6 low_confidence）
- Step 4: FreeSASA — nanobody 表面 Lys/Cys SASA>40Å² 过滤，取 top 3 偶联位点
- Step 5: Linker Select — 根据位点类型选 maleimide(Cys) / NHS(Lys)，cleavable，compatible MMAE
- Step 6: Fetch Payload — PubChem 获取 MMAE SMILES + MW
- Step 7: RDKit Conjugate — antibody + linker + payload → ADC，DAR=4
- 每步 try/except，单步失败标记 `partial: true` 不阻塞后续
- `dry_run=true` 返回完整 mock 结构含 `adc_design` 字段
- AF3 任务间隔 5 秒避免 GPU OOM，timeout 1800s
- `BinderDesignPipelineRequest`：`num_designs`（alias）、`dry_run`、`hotspot_residues: list[str]`

### GROMACS 蛋白配体 MD 修复（3 个关键 bug）
1. **tc-grps 动态检测**：`make_ndx` 后解析 `index.ndx` 找实际合并组名（如 `Protein_UNL`），不再硬编码 `Protein_LIG`
2. **MDP 延迟生成**：NVT/NPT/MD 的 MDP 在 `_run()` 内 `make_ndx` 之后才写入，确保 tc-grps 正确
3. **每步文件检查**：em.gro → nvt.gro → npt.gro → md.xtc，缺失立即 raise + 附 .log 最后 20 行
4. **废弃参数删除**：`dispdivcorr` → `DispCorr`，删除 `ns_type = grid`（GROMACS 2024 已废弃）

### RAG 知识库初始化
- 6 轮 ADC 文献检索（HER2/CD30/TROP2/DAR/Linker），29 篇入库 ChromaDB
- 本地 embedding 查询验证通过（BAAI/bge-m3 ONNX）

### dashboard.html 前端（单文件 760 行）
- 深色毛玻璃 UI，Space Grotesk + IBM Plex Mono，canvas 粒子背景
- 顶栏：自然语言输入 → POST `/api/v1/agent/chat`，4 个 example prompt
- 左栏：Agent Pipeline Timeline（2s polling，工具节点动画）
- 右栏：9 种动态 Tab（Chat/Protein3D/Molecule2D/Docking/ADMET/Sites/Linker/Payload/ADC Assembly/MD）
- Agent Chat：底部输入框 + markdown 渲染 + 600s 超时
- 服务地址：`http://192.168.31.23:9099/dashboard.html`

### sync_claude_to_skills.py 文档同步脚本
- 5 个同步目标：skills/*.md + routers/*.py + qwen_tools.py + skills_loader + CLAUDE.md（只读源）
- 幂等可重复执行，用标记包裹（`AUTO_SYNC_FROM_CLAUDE_MD` / `SYNC_NOTES`）
- Router 注释安全：换行符/引号清理，不破坏 Python 语法
- 用法：`python scripts/sync_claude_to_skills.py --apply`

### ADC 工具链验证
- freesasa / linker_select / rdkit_conjugate 三处注册完整（router + tool_definitions + TOOL_MAP）

---

## Current Outstanding Work

1. **100 ns MD 进行中** — task_id `2add68d5`，EM 已通过，NVT/NPT/MD 进行中（tc-grps 修复后第三次提交）
2. **MM/PBSA** — 100ns MD 完成后 post-process md.xtc 计算结合自由能
3. **AutoDock pipeline integration** — complete `pipeline.py` autodock preprocessing chain
4. ~~**Frontend**~~ — ✅ dashboard.html 完成，Agent Chat 联通，实时任务追踪正常
5. ~~**RAG**~~ — ✅ 已完成
6. ~~**chemprop Dockerfile**~~ — ✅ Dockerfile 已有 `ln -sf`，容器 python3.11 正常
7. ~~**ADMET 模型**~~ — ✅ 5 个标准 benchmark 模型训练完成
8. ~~**RAG 知识库初始化**~~ — ✅ 29 篇 ADC 文献入库
9. **端到端测试** — 用真实任务验证 dashboard 动态 Tab 解锁（蛋白结构/对接/ADMET/ADC）
10. **chemprop 容器镜像重建** — Dockerfile 已修，等空闲时 `docker build` 固化到镜像

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

## 环境关键事项（血泪经验，遇到问题先看这里）

### 两个独立 FastAPI 服务（不要搞混！）

| 服务 | 入口 | 端口 | Python | 管理方式 |
|------|------|------|--------|----------|
| OIH平台 | `/data/oih/oih-api/main.py` | 8080 | `/data/oih/miniconda/bin/python` | systemd `oih-api` |
| Gemini前端 | `/opt/oih-agent/app.py` (`app:app`) | 8001 | `/opt/oih-agent/fastapi/bin/python3` | 手动 |

**OIH平台**（本项目）：
- 重启（统一用这条，sudo 在非交互终端需要密码所以不用 systemctl）：
  ```bash
  NO_PROXY='*' no_proxy='*' nohup /data/oih/miniconda/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8080 --workers 1 > /tmp/fastapi.log 2>&1 &
  ```
- 查日志：`tail -f /tmp/fastapi.log` 或 `journalctl -u oih-api -n 30 --no-pager`
- pip安装：`/data/oih/miniconda/bin/pip install <包名>`
- Python：`/data/oih/miniconda/bin/python`
- 验证：`curl -s --noproxy '*' http://localhost:8080/health` → 返回 `{"status":"ok", "containers":{...}}`
- 不要用 `/opt/oih-agent/fastapi/` 下的任何东西，那是 Gemini 前端的 venv

**Gemini前端**（不要动！）：
- 进程：`/opt/oih-agent/fastapi/bin/uvicorn app:app --host 127.0.0.1 --port 8001`
- 验证：`curl -s --noproxy '*' http://127.0.0.1:8001/health` → 返回 `{"ok":true, "model":"gemini-2.5-flash"}`
- 不要 kill、不要重启、不要修改、不要往它的 venv 装包

### 代理
- Clash 端口 7890
- localhost请求必须加 --noproxy '*' 否则502
- curl内网：curl -s --noproxy '*' http://127.0.0.1:8001/...

### GPU规则
- 所有容器 NVIDIA_VISIBLE_DEVICES=1，容器内永远用 device=0
- JAX容器（AF3、BindCraft）不要挂载宿主机 cuda lib64

### 三处注册规则（新工具必须同时改三处）
1. routers/ 下对应文件加路由
2. tool_definitions/qwen_tools.py 加tool定义
3. qwen_agent.py 的 TOOL_MAP 加条目

## Claude Code 安全提示处理
以下命令模式会触发安全确认，直接选Yes：
- `nohup ... > /tmp/fastapi.log 2>&1 &` → Yes（标准后台启动）
- `sudo systemctl ...` → Yes
- `/data/oih/miniconda/bin/pip install ...` → Yes（已设置不再询问）

---

## oih-api.service 陷阱
- systemd 有 oih-api.service（enabled），服务器重启会自动启动
- 但平时用 nohup 手动启动，两者冲突会导致 kill 后被 systemd 拉起
- 每次重启前先：`systemctl stop oih-api`
- 正式部署时考虑把 nohup 命令写进 service 文件统一管理

---

## 2026-03-16 完成项

### 前端功能
- dashboard.html 新增附件上传UI（794→959行）
  - 📎 按钮弹出菜单（PDB上传/FASTA粘贴/SMILES粘贴）
  - 附件标签预览（输入框上方，可删除）
  - PTM检测结果卡片（✅/⚠️/📁 分类显示）
  - 发送逻辑扩展：携带 pdb_content/fasta_sequence/smiles/filename
- 终止任务按钮（running任务旁 ⏹）+ 终止对话按钮（🗑 清除会话）

### 后端功能
- AgentChatRequest 扩展4个Optional字段（pdb_content/fasta_sequence/smiles/filename）
- detect_ptm() 函数：自动识别糖基化/磷酸化/二硫键/乙酰化
- generate_tool_inputs() 函数：自动生成 af3_input.json / gromacs_ptm_notes.json / adc_input.json
- is_simple_message() 动态 thinking_budget（问候类=0，复杂任务=1024+）
- /api/v1/tasks/{task_id}/cancel 端点

### 知识文件
- skills/PTM_UPLOAD_PARADIGM.md：Qwen PTM决策范式
- docs/paper/OIH_manuscript.md：bioRxiv论文草稿框架

### 测试验证
- 测试1 纯文本 hello：✅ 200，turns=2
- 测试2 SMILES阿司匹林：✅ chemprop自动调用，Tox21=0.0154
- 测试3 PDB+PTM：✅ af3_input.json(NAG糖基化)+gromacs_ptm_notes.json(二硫键)生成正确
- is_simple_message：hi→budget=0 ✅，预测ADMET→budget=1024+ ✅

### 系统状态
- 100ns MD：22.74ns/100ns（22.7%），轨迹24GB，性能19.2ns/day，自然退出
- 11容器全部running

### 待完成
- Case 2：纳米抗体设计实验（Fig 4数据）
- Case 3：HER2-ADC设计实验（Fig 5数据）
- Fig 6：各工具运行时间统计
- MD动画渲染（MDAnalysis→gif）
- 100ns MD重新提交（从22.74ns checkpoint续跑）
- MM/PBSA结合自由能计算

---

## 2026-03-17–18 完成项

### binder_design_pipeline 完整 7-step + ADC
- Step 1-3: RFdiffusion → ProteinMPNN → AF3 验证（ipTM ≥0.75 high / ≥0.6 low_confidence）
- Step 4-7: FreeSASA 偶联位点 → Linker 选择 → MMAE payload → RDKit 偶联（DAR=4）
- 每步 try/except，单步失败标记 partial，不阻塞后续
- `pdb_id` 参数支持（自动 fetch_pdb）
- `dry_run` 参数支持
- `_parse_mpnn_fasta()` 修复 FASTA 解析
- `hotspot_residues` 改为 `list[str]`
- HER2 nanobody ADC v3 已提交（task: 6a34f32f）

### Task 持久化 ✅
- `core/task_manager.py`：状态变更时写入 `data/tasks/{task_id}.json`
- 启动时扫描目录恢复历史任务（running/pending 标记为 failed）
- 字段：task_id, tool, status, progress, progress_msg, result, error, created_at, updated_at

### Cancel 接口 ✅
- `DELETE /api/v1/tasks/{task_id}`：取消 pending/running 任务
- `core/task_manager.py` cancel_task 扩展为支持 running 状态
- 已完成/已失败任务返回 400

### Dashboard Results Hub ✅
- 常驻 Results Hub tab，按工具分组展示所有历史任务
- 左侧工具树 + 右侧工具类型特定渲染（12种）
- 路径字段带复制按钮
- Pipeline 结果内联 6 统计卡片 + AF3 表格 + ADC 摘要 + 下载按钮

### Dashboard Pipeline Preview ✅
- 输入时前端关键词匹配，显示流程预览（6种 pipeline）
- 横向流程图：emoji + 工具名 + 预计时间

### Dashboard 子任务折叠 ✅
- Pipeline 触发的子任务按 created_at 时间窗口归组
- 左侧只显示 pipeline 卡片 + 子任务摘要（ProteinMPNN ×20 ✓18 ⏳2）

### Dashboard Tab 高亮 ✅
- running → cyan 脉冲动画，completed → 绿色 3 秒

### Dashboard 3Dmol.js 修复 ✅
- data 验证 + addModel try/catch + 友好错误提示

### Qwen 同步 ✅
- qwen_tools.py：binder_design_pipeline 描述更新（7步 + pdb_id + adc_design）
- skills_loader.py：新增 ADC 关键词映射
- ADC_WORKFLOW.md：完整 7 步流程 + 偶联化学规则 + DAR 说明

---

## 2026-03-18 完成项

### AF3 超时修复 — pipeline.py `_wait_for_af3_task()`
- **根因**：v3 任务 5 个 AF3 设计全部失败。rank1 超时 1800s（但实际已跑完，ipTM=0.48），rank2-5 被路由到 DEGRADED 队列 OOM exit 1
- **修复 1**：新增 `_wait_for_af3_task()` 无限等待函数，每 30s poll，仅在 OOM/exit1/cancelled/连续10次同错 时判定失败
- **修复 2**：binder_pipeline + drug_discovery_pipeline 的 AF3 调用改用 `_wait_for_af3_task()`
- **修复 3**：`_wait_for_task()` 支持 `timeout <= 0` 表示无限等待

### AF3 不走 DEGRADED 队列 — task_manager.py `_NO_DEGRADED_TOOLS`
- 新增 `_NO_DEGRADED_TOOLS = {"alphafold3", "bindcraft"}`
- `_resolve_queue()` 中这些工具 VRAM 不足时每 60s 重试检查，等到 VRAM 足够再进 GPU 队列
- 不再降级到 DEGRADED 导致 OOM crash

### 静态文件 /outputs 挂载 — main.py
- `app.mount("/outputs", StaticFiles(directory="/data/oih/outputs"), name="outputs")`
- dashboard.html 3D 查看器和下载按钮路径改为 `/outputs/...`（之前 `/static/outputs/...` 404）
- 验证：val_0 CIF 文件 HTTP 200

### HER2 ADC v4 提交
- task_id: `b07f91e0-004f-4ec8-ad00-d7a84f826859`
- 参数：1N8Z, hotspot S310/T311/Q313/L317, 10 designs
- AF3 现在会无限等待完成，不会超时

### 知识蒸馏系统 (2026-03-18)
- `skills/SELF_DIAGNOSIS_WORKFLOW.md` — 自我诊断手册（AF3/VRAM/RFdiffusion/GROMACS）
- `qwen_tools.py` system prompt — 新增自主诊断修复指令（4步闭环：诊断→修复→重跑→报告）
- `data/distillation/` — 蒸馏训练数据目录
- `scripts/collect_distillation_data.py` — 自动收集任务案例
- 当前案例数：**67条**（4条手工 + 35条历史提取 + 28条自动收集）
- 分类覆盖：gpu(6) / container(3) / tool(4) / pipeline(16) / proxy(2) / dashboard(1) / 自动(28) / 手工(4) / abandoned(2) / reference(1)
- 下一步：积累到100条后做 LoRA 微调

---

## 2026-03-20 完成项

### pocket_guided_binder_pipeline 重构 — 多维口袋评分系统（14步）

**旧流程**（11步）：P2Rank top pocket → 直接取 top 6 残基 → DiffDock 盲对接交叉验证 → RFdiffusion
**新流程**（14步）：多维评分 + Qwen 选择 → DiffDock 仅作成药性参考

**口袋评分公式**：
```
composite = p2rank_prob × 0.2 + sasa_score × 0.2 + conservation × 0.2 + rag_score × 0.3 + electrostatics × 0.1
```

| 维度 | 计算方法 | 权重 |
|------|---------|------|
| P2Rank | P2Rank ML probability（已归一化到 0-1） | 0.2 |
| SASA | FreeSASA 对 target PDB 计算每个 pocket 残基的 mean_SASA / 150，cap 1.0 | 0.2 |
| Conservation | B-factor 作为柔性代理：1 - mean_normalised_bfactor（低B=刚性=保守=好靶点） | 0.2 |
| RAG | 文献残基重叠度：0(无) / 0.5(1-2残基) / 1.0(3+残基) | 0.3 |
| Electrostatics | 口袋内带电残基(K/R/E/D/H)比例 | 0.1 |

**14步详细流程**：
1. fetch_pdb — 下载靶标
2. RAG 文献检索 — `{target} {pdb_id} binding site epitope domain experimental validation`
3. fpocket + P2Rank — 并行检测口袋（top 5）
4. FreeSASA per pocket — 靶标 PDB 表面暴露度
5. B-factor conservation + electrostatics — 保守性和静电分析
6. 复合评分 + Qwen 结构生物学家口袋选择 — 返回 pocket_id + 6 个 hotspot + 选择理由
7. DiffDock — 仅对选中口袋做成药性参考（标记 `small molecule druggability reference`）
8. RFdiffusion — 用 Qwen 选择的 hotspot 做 binder backbone 设计
9. ProteinMPNN — 序列设计
10. AF3 验证 — ipTM ≥ 0.6
11-14. ADC 构建 — FreeSASA 偶联位点 → Linker → MMAE → RDKit (DAR=4)

**新增 helper 函数**（`routers/pipeline.py`）：
- `_parse_p2rank_residues()` — P2Rank 残基解析（提取自内联代码）
- `_compute_bfactor_conservation()` — B-factor 保守性评分
- `_compute_electrostatics_from_pdb()` — 带电残基比例
- `_compute_sasa_score_for_pocket()` — 口袋 SASA 评分
- `_compute_rag_score()` — 文献残基重叠评分
- `_extract_residue_numbers_from_text()` — 正则提取文献中残基编号
- `_rag_search_pocket_context()` — 直接调用 `HybridRetriever.retrieve()`（不走 HTTP）
- `_compute_freesasa_per_residue()` — 直接调用 FreeSASA C 库（不走 task）
- `_qwen_select_pocket()` — Qwen3-14B 口袋选择（返回 JSON）

**返回结果新增字段**：
- `pocket_scores` — 每个口袋 5 维分数 + composite
- `selected_pocket` — `{id, center, hotspots, reason, composite_score}`
- `diffdock_reference` — `{label, confidence, pose_path}`

### AF3 PDB 解析修复 — 3 层 fallback
- **原因**：CIF→PDB 转换失败时 `af3_pdb=None` → `"No PDB for FreeSASA"` 错误
- **修复**：
  1. `convert_af3_cif_to_pdb()`（gemmi + pdbfixer 完整原子补全）
  2. Fallback: 纯 `gemmi.write_pdb()`（跳过 pdbfixer）
  3. Fallback: 扫描 AF3 目录和 seed 子目录找现有 `.pdb` 文件
  4. 已转换的 `_for_sasa.pdb` 文件跳过重复转换

### 口袋评分运行时 3 个 bug 修复（2026-03-21）

**Bug 1 — RAG self-HTTP 死锁**：
- `_rag_search_pocket_context()` 通过 `httpx.get("http://127.0.0.1:8080/api/v1/rag/search")` 调用本机 RAG 接口
- `--workers 1` 的 uvicorn 唯一 worker 阻塞在 pipeline 协程上，无法处理 RAG 请求 → 死锁
- **修复**：改为直接调用 `from retrieval.rag_router import get_retriever; retriever.retrieve(...)` 跳过 HTTP

**Bug 2 — Qwen3 thinking mode 返回空 content**：
- Qwen3-14B 默认开启 thinking mode，`content` 字段为 `null`，思考过程在 `reasoning` 字段
- `_qwen_select_pocket()` 取 `content` 后为 None → 后续 `re.sub()` 报 TypeError，被 except 吞掉
- **修复**：请求体加 `"chat_template_kwargs": {"enable_thinking": False}` 禁用 thinking，直接返回 JSON
- **防御**：增加 `content = msg.get("content") or ""` null 保护 + `exc_info=True` 完整 traceback 日志

**Bug 3 — RAG 残基编号匹配失败**：
- RAG 从文献提取纯数字残基编号 `['42', '310']`
- P2Rank 残基格式为 `['A42', 'A310']`（带链前缀）
- 旧代码尝试给 RAG 残基加链前缀，但逻辑不可靠
- **修复**：`_compute_rag_score()` 比较时双方都 `lstrip` 字母字符，统一为纯数字比较

### RAG ChromaDB 知识库配置与扩充

**ChromaDB 路径**：`/data/oih/knowledge/chroma`（collection: `oih_knowledge`）
- 配置位置：`/data/oih/retrieval/config.py` → `cfg.chroma_path`
- 环境变量：`OIH_CHROMA_PATH`（默认已指向正确路径）
- `/data/oih/retrieval/chroma_db` 是空库，不使用
- Embedding: BAAI/bge-m3 (ONNX)

**文档数量**：80 篇（50 原有 ADC 文献 + 26 PubMed HER2 结构论文 + 4 curated 残基数据）

**HER2 残基级别数据**（4 条 curated entries）：
- `curated_her2_pertuzumab_residues` — Domain II 二聚化臂：S267, L269, T271, K273, E280, G282 (PDB 1S78)
- `curated_her2_trastuzumab_residues` — Domain IV：K505, F506, P557, E558, D560, E561 (PDB 1N8Z)
- `curated_her2_domain_hotspots` — 4 个治疗口袋汇总 + Kd 值
- `curated_her2_2a91_pocket1` — P2Rank pocket 1 残基 ↔ Domain I/III 界面映射

**重要**：PubMed 摘要通常不含残基编号，需要从全文提取或手动 curate。`rag_score > 0` 需要知识库中有残基级数据。对新靶标需按此模式补充 curated entries。

**入库方法**：
- PubMed 批量：`POST /api/v1/rag/local/add-papers` + PMID 列表
- 手动 curated：`chromadb.PersistentClient(path).get_collection('oih_knowledge').upsert()`
- PDF 上传：`POST /api/v1/rag/local/upload-pdf`（需 `OIH_PDF_INGEST=true`）

### BindCraft 并行路径（2026-03-21）
- `pocket_guided_binder_pipeline` 从 14 步扩展到 **15 步**
- Step 8 拆分为 8a (RFdiffusion) + 8b (BindCraft)，用 `asyncio` 并行执行
- BindCraft 失败不阻塞 pipeline（`try/except`，标记 `status: failed`）
- BindCraft 结果提取序列（`_extract_sequence_from_pdb`），最多 2 条合并到 AF3 候选列表
- AF3 验证从合并后的候选（MPNN + BindCraft）中取 top 5
- BindCraft 参数：`num_designs=10`（不能<10, pydantic ge=10），使用 `default_4stage_multimer.json`

---

## 2026-03-21 完成项

### DiscoTope3 集成（第14个容器 oih-discotope3）
- 3处注册完成，B细胞表位预测
- 验证：1N8Z HER2 → 195表位/1015残基
- DT3 raw score 范围 0.001-0.5（不同结构不同），**不要用固定阈值0.7**
- `calibrated_score_epi_threshold` 默认 0.5

### IgFold 集成（第15个容器 oih-igfold）
- 抗体/纳米抗体序列→3D结构快速预测（~2s/seq GPU）
- 基于 proteinmpnn:latest，igfold 0.4.0 + antiberty
- 输出 pRMSD（不是pLDDT），转换：`pseudo_plddt = 100 - prmsd * 20`
- `do_renum=False`（anarcii≠anarci 包名冲突）

### ESM2 新工具
- `esm2_score_sequences`：伪困惑度评分（PPL<15=合格序列）
- `esm2_mutant_scan`：ESM-1v 点突变扫描（ΔΔG proxy）
- ESM-1v 模型 7.3GB，首次加载需下载

### IEDB + SAbDab 接入 RAG
- IEDB: `https://query-api.iedb.org/bcell_search` PostgREST API
- SAbDab: TSV全量下载 + antigen名过滤（8MB, 20K行）
- `extract_antigen_name()` 正则提取靶标名
- RAG gather 总超时 30s，PubMed efetch 10s+15s 超时保护

### pocket_guided_binder_pipeline 升级 16 步 6D 评分
- 评分公式：`p2rank(0.15) + sasa(0.15) + conservation(0.15) + rag(0.25) + electrostatics(0.10) + epitope(0.20)`
- DiscoTope3 与 fpocket/P2Rank 并行（step 3）
- `_compute_epitope_score_for_pocket()`：口袋残基8Å内DT3表位比例
- ESM2 PPL filter → IgFold pLDDT filter → AF3（漏斗）

### known_epitope_override（血泪经验）
- 触发条件：DT3 高分残基 ∩ RAG 文献残基 ≥ 2
- DT3 阈值：**adaptive = max(top 20% score, 0.10)**，不要用固定 0.7
- 编号对齐：PDB 与 UniProt 编号可能有偏移，**自动检测 offset(0, ±22, ±23)**
- 模糊匹配：`abs(dt3_num + offset - rag_num) <= 3`
- **空间聚集**：`_cluster_hotspots()` centroid 距离 ≤ 15Å，**最多 5 个 hotspot**
- 不聚集的后果：8个分散hotspot→RFdiffusion 每设计 30+ 分钟→超时

### RFdiffusion 大靶标注意事项
- HER2 ECD 506 残基 → 每设计 20-40 分钟（正常 ~2 分钟）
- **timeout 必须 7200s**（不是 3600s）
- **num_designs 用 10**（不要 20，太慢）
- **hotspot 必须空间聚集**（<= 15Å centroid），分散的会极慢
- 超时后恢复已生成 PDB（`_wait_for_task` try/except + scan output dir）

### 模型缓存 Volume Mount
| 容器 | 宿主机路径 | 容器路径 |
|------|-----------|---------|
| oih-esm | /data/oih/model_cache/esm_torch_hub | /root/.cache/torch/hub |
| oih-discotope3 | /data/oih/model_cache/torch/hub | /root/.cache/torch/hub |
| oih-discotope3 | /data/oih/model_cache/discotope3_models | /app/discotope3/models |
| oih-igfold | /data/oih/model_cache/igfold | /root/.cache |

### 当前工具总数：30 个，容器数：15 个
- 新增 5 工具：discotope3_predict, igfold_predict, esm2_score_sequences, esm2_mutant_scan, extract_interface_residues
- 新增 2 容器：oih-discotope3, oih-igfold
- RAG 5 源：PubMed + bioRxiv + IEDB + SAbDab + LocalDB

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

### pLDDT Quality Expectations by Tier
- Tier 1 hotspots → RFdiffusion backbone quality: pLDDT > 70 expected
- Tier 3 hotspots → RFdiffusion backbone quality: pLDDT 40-70, need more designs
