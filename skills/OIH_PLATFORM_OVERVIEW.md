# OIH 平台总览与关键规则

## 服务器信息
- IP：192.168.31.23
- CPU：AMD EPYC 7763，128线程
- RAM：251GB（230GB可用）
- GPU0：RTX 4090 24GB → Qwen2.5/Qwen3 推理专用（vLLM）
- GPU1：RTX 4090 45GB → 所有生物计算工具
- 端口：FastAPI=8080，vLLM=8002，Nginx=8000

## ⚠️ GPU映射关键规则
**所有计算容器内永远用 gpu_id=0 / device=0**
- docker-compose 中 `NVIDIA_VISIBLE_DEVICES=1` 将宿主机GPU1映射为容器内GPU0
- 容器内不存在gpu_id=1，用1会报 "Device ID 1 did not correspond to any of the 1 detected device(s)"
- 适用：gromacs、diffdock、rfdiffusion、proteinmpnn、bindcraft、alphafold3、esm、gnina

## 容器清单

| 容器名 | 镜像 | 用途 | 状态 |
|--------|------|------|------|
| oih-gromacs | gromacs:2024.4 | MD模拟 | ✅已验证 |
| oih-autodock-gpu | autodock-gpu:latest | 分子对接 | ✅已验证 |
| oih-gnina | gnina:latest | GPU对接 | ✅已验证 |
| oih-fpocket | fpocket:latest | 口袋检测 | ✅已验证 |
| oih-p2rank | p2rank:latest | ML口袋预测 | ✅已验证 |
| oih-rfdiffusion | rfdiffusion:latest | 蛋白骨架设计 | ✅已验证 |
| oih-proteinmpnn | proteinmpnn:latest | 序列设计 | ✅已验证 |
| oih-diffdock | diffdock:latest | 柔性对接 | ✅已验证 |
| oih-bindcraft | bindcraft:latest | Binder设计 | ✅已验证 |
| oih-alphafold3 | alphafold3:latest | 结构预测 | ✅已验证 |
| oih-esm | esm:latest | 序列嵌入 | ✅已验证 |
| oih-chemprop | chemprop:latest | 小分子性质预测 | ✅已验证（CPU模式） |

## 模型文件位置（宿主机）

| 工具 | 模型路径 |
|------|---------|
| RFdiffusion | /data/rfdiffusion/models/ |
| ProteinMPNN | /data/proteinmpnn/ProteinMPNN/ |
| AlphaFold3 | /data/af3/models/（待确认）|
| Qwen3-14B | /data/oih/models/Qwen3-14B-AWQ |

## 关键路径

| 用途 | 路径 |
|------|------|
| OIH API | /data/oih/oih-api/ |
| docker-compose | /data/oih/oih-api/docker-compose.yml |
| 输入数据 | /data/oih/inputs/ |
| 输出数据 | /data/oih/outputs/ |
| Skills文档 | /data/oih/oih-api/skills/ |
| AutoDock可执行文件 | /usr/local/bin/autodock_gpu_128wi（容器内）|

## Python命令注意事项
- oih-rfdiffusion：用 `python3`（没有python）
- oih-gromacs：用 `gmx`
- oih-autodock-gpu：用 `python3`，prody已安装

## 工具在容器外/宿主机运行的命令
```bash
obabel           # 配体/受体格式转换，生成pdbqt
autogrid4        # AutoDock grid生成
```

## API路由结构
```
/api/v1/structure  → structure_prediction.py  (AF3, ESM)
/api/v1/design     → protein_design.py        (RFdiffusion, ProteinMPNN, BindCraft)
/api/v1/pocket     → pocket_analysis.py       (fpocket, p2rank)
/api/v1/docking    → molecular_docking.py     (AutoDock-GPU, gnina, DiffDock, Vina)
/api/v1/md         → md_simulation.py         (GROMACS)
/api/v1/ml         → ml_tools.py              (ESM embed, Chemprop)
/api/v1/adc        → adc.py                   (ADC设计: linker_select, rdkit_conjugate)
/api/v1/pipeline   → pipeline.py              (完整流程)
/api/v1/tasks      → tasks.py                 (任务管理)
```

## 任务队列（已实现）
- CPU队列 Semaphore(8)：fpocket/p2rank/chemprop/chemprop_predict/freesasa/rdkit_conjugate
- GPU队列 Semaphore(3)：gnina/autodock/af3/rfdiffusion/proteinmpnn/bindcraft/diffdock/gromacs/esm
- 降级队列 Semaphore(4)：GPU VRAM不足时自动降级到CPU+内存
- **永不降级**：AF3/BindCraft 不进降级队列（OOM crash），等待 GPU 空闲

## 任务持久化与取消
- 状态变更时写入 `data/tasks/{task_id}.json`
- 服务启动时扫描目录恢复历史任务（running/pending 标记为 failed）
- 字段：task_id, tool, status, progress, progress_msg, result, error, created_at, updated_at
- 取消接口：`DELETE /api/v1/tasks/{task_id}`（pending/running → cancelled）

## 系统状态工具
- `GET /api/v1/system/status` → containers[], gpu_queue[], task_summary{}
- Qwen 可调用 `get_system_status` 回答"有多少容器在运行"等问题

## 静态文件服务
- `/outputs/` 挂载 `/data/oih/outputs/`，HTTP 直接下载 CIF/PDB/SDF 文件
- Dashboard 3D 查看器和下载按钮使用 `/outputs/...` 路径

## ADC 工具链（2026-03-15 新增）

| 工具 | 路由 | 功能 | 队列 |
|------|------|------|------|
| freesasa | /api/v1/design/freesasa | 抗体偶联位点预测（Lys/Cys SASA>40Å²分析） | CPU |
| linker_select | /api/v1/adc/linker_select | ADC linker筛选（20条临床验证库，approved优先） | 同步 |
| rdkit_conjugate | /api/v1/adc/rdkit_conjugate | ADC payload-linker共价偶联（7类反应化学） | CPU |

**ADC设计流程**：fetch_pdb → freesasa(偶联位点) → fetch_molecule(payload) → linker_select(选linker) → rdkit_conjugate(偶联) → chemprop(ADMET)

## 显存预估
| 工具 | 显存 |
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

## ⚠️ JAX/BindCraft GPU关键规则
- JAX使用pip安装的自带CUDA库，**不需要**挂载宿主机cuda lib64
- 挂载`/usr/local/cuda-12.6/lib64`会导致LD_LIBRARY_PATH冲突，JAX无法识别GPU
- BindCraft容器**不挂载**任何cuda路径，JAX自己管理CUDA
- 验证命令：`unset LD_LIBRARY_PATH && python3 -c 'import jax; print(jax.devices())'`
- 其他容器（gromacs/gnina等）用PyTorch，需要挂载cuda lib64；JAX容器不需要
