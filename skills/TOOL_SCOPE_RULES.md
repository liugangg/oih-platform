# Tool Scope Rules — 工具适用范围规则

## 核心原则
每个工具都有特定的适用范围。错误使用不会报错，但输出无意义。
Qwen 必须在调用前判断工具是否适用。

## 工具适用范围表

| 工具 | 适用输入 | 禁止输入 | 失败模式 |
|------|---------|---------|---------|
| igfold_predict | 抗体/纳米抗体序列 | de novo binder序列 | pLDDT=5-40，全部过滤 |
| discotope3_predict | 蛋白质抗原PDB | 小分子靶点 | 无意义表位预测 |
| extract_interface_residues | 有已知复合物PDB | 无结构靶点 | 文件不存在报错 |
| fpocket | 任意蛋白PDB | 抗体结合面预测 | 找到小分子口袋非抗体位点 |
| esm2_score_sequences | 任意蛋白序列 | 无 | 通用，安全 |
| esm2_mutant_scan | 任意蛋白序列 | 无 | 通用，安全 |
| rfdiffusion_design | 任意蛋白靶点 | 无 | 通用 |
| proteinmpnn_design | 任意backbone PDB | 无 | 通用 |
| alphafold3_predict | 任意序列/复合物 | 无 | 通用，最终验证 |

## binder_type 判断规则

| 用户说的 | binder_type | IgFold |
|---------|-------------|--------|
| "纳米抗体/nanobody/VHH/单域抗体" | nanobody | ✅ 启用 |
| "抗体/antibody/IgG" | antibody | ✅ 启用 |
| "设计binder/结合蛋白/de novo" | de_novo | ❌ 跳过 |
| 未指定 | de_novo（默认） | ❌ 跳过 |

## binder 设计路径选择

### 有已知复合物PDB（Tier 1，如 HER2+trastuzumab=1N8Z）
```
extract_interface_residues → RFdiffusion → ProteinMPNN → ESM2 → AF3
```

### 无已知结构（Tier 3，新靶点）
```
DiscoTope3 + IEDB + RAG → known_epitope_override → RFdiffusion → ProteinMPNN → ESM2 → AF3
```

### nanobody 设计（IgFold 适用）
```
... → ProteinMPNN → ESM2(PPL<15) → IgFold(pLDDT>70) → AF3
```

### de novo binder 设计（IgFold 不适用）
```
... → ProteinMPNN → ESM2 perplexity 过滤(top 10) → AF3 直接验证
```

## 验证结果（2026-03-22）

### HER2 Tier 1 + de_novo + Domain 截取
- extract_interface_residues → C558-C573 (Domain IV) → RFdiffusion → AF3
- val_0: ipTM=0.84 (全长1015aa), val_2: ipTM=**0.86** (Domain4 202aa)
- **Domain 截取比全长更优** — AF3 对短序列复合物更准确
- 3/3 通过 ipTM≥0.6 (100% 通过率)

## RAG 优先级规则（所有靶点通用）

**PPI interface > epitope prediction**

当 RAG 返回已知蛋白-蛋白相互作用界面（共晶结构、突变验证的结合残基）时，
必须优先使用这些残基做 hotspot，覆盖 DiscoTope3/IEDB 的表位预测。

- B 细胞表位预测的是免疫原性，不是最佳 binder 设计位点
- PPI 界面残基是实验验证的蛋白结合位点，直接适用于 binder 设计
- RAG 搜索分两层：Layer 1 (PPI共晶/突变) → Layer 2 (epitope fallback)
- CD36 教训：DiscoTope3 选了表面暴露残基 A397/A400，但应该用 CLESH domain (93-120)

## MPNN Chain 检测（2026-03-26 最终修复）

RFdiffusion binder_design 输出中链顺序不固定：
- 最短链 = binder scaffold (60-120aa)
- 最长链 = target protein (100-600aa)

**chains_to_design 已改为默认 "auto"**。Router 自动用 gemmi 检测最短链。
调用 `proteinmpnn_sequence_design` 时不需要指定 chains_to_design，留默认即可。

| 靶点 | RFdiff chain A | RFdiff chain B | MPNN auto 检测 |
|------|---------------|---------------|----------------|
| HER2 (原始C链) | binder 214aa | target 76aa | A ✅ (最短) |
| CD36 (原始A链) | target 400aa | binder 78-95aa | B ✅ (最短) |
| EGFR (原始A链) | target 613aa | binder 74aa | B ✅ (最短) |
| Trop2 (原始A链) | target 274aa | binder ~80aa | B ✅ (最短) |

验证方法：MPNN FASTA designed sequence 长度 60-120aa = 正确(binder)，>200aa = 错误(target)

## ipSAE 接口验证（2026-03-26 新增）

AF3验证完成后，**必须调用 ipsae_score** 检查接口质量：
- ipSAE > 0.15 = 真阳性（binder确实结合了antigen）
- ipSAE = 0.000 = 假阳性（AF3给了ipTM但实际无接口接触）
- 调用方式：`ipsae_score(af3_output_dir="AF3输出目录路径")`
- CD36 DT3路线教训：21个design的ipTM=0.12-0.43，但ipSAE全部=0.000

## 序列提取安全规则（2026-03-27 bug fix）

从 PDB 提取序列时，**必须指定 chain ID**，否则会把多条链拼接成一条：
- 错误：`_extract_sequence_from_pdb(pdb)` → 400aa target + 69aa binder = 469aa 拼接
- 正确：`_extract_sequence_from_pdb(pdb, chain_id='B')` → 69aa binder
- 影响：61 个 pre-fix 设计全部提交了错误序列给 AF3
- 验证：AF3 input JSON 中 binder 链应为 60-120aa，如果 >200aa 一定是拼接错误

## GPU 资源泄漏防护（2026-03-27 关键修复）

`docker exec` 在容器内创建的进程 parent 是 containerd-shim，**不是 FastAPI**。
task_manager 完成任务后释放 semaphore，但容器内的 python/jackhmmer 进程可能还在跑，占 44GB VRAM。

**已修复**：每个 GPU 任务完成后（无论成功/失败），自动 kill 容器内匹配进程。
两层防护：
1. `_cleanup_container_after_task()` — 任务完成后立即清理
2. `_kill_orphaned_gpu_processes()` — FastAPI 重启时清理历史残留

如果发现 GPU 满但没有 running task → `nvidia-smi` 看 PID → `docker exec <container> kill -9 <pid>`

### 历史教训
- IgFold 误用于 de novo binder → pLDDT 全部 0.4-12.6 → 0 个候选进入 AF3
- 根因：IgFold 使用 AntiBERTy，只在抗体数据集上训练
- 修复：pipeline 加入 binder_type 参数，de_novo 时跳过 IgFold
- FreeSASA 不接受 CIF → 需先转 PDB
- GPU Semaphore 必须=1，否则两个大模型并发 OOM
