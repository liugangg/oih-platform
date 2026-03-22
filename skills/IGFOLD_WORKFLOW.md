# IgFold 抗体结构预测工作流

## 基本信息
- 容器：oih-igfold（基于 proteinmpnn:latest）
- 软件：igfold 0.4.0 + antiberty + abnumber
- GPU：NVIDIA_VISIBLE_DEVICES=1 → 容器内 cuda:0
- 队列：GPU（VRAM ~4GB）
- 超时：120秒/序列
- 模型：打包在 pip 包内，docker commit 固定

## 功能
从抗体/纳米抗体序列快速预测 3D 结构（~2-4秒/条 GPU）。
主要用作 ProteinMPNN → AF3 之间的预筛选漏斗。

## 参数
- `job_name`: 任务标识符
- `sequences`: 链序列字典，如 `{"H": "EVQLVE..."}`（纳米抗体只需 H 链）
- `do_refine`: OpenMM 结构精修（更慢但几何更好，仅用于最终候选）
- `do_renum`: ANARCI CDR 编号（当前禁用，anarci 包名冲突待修复）

## 输出
- `output_pdb`: 预测结构 PDB 文件路径
- `mean_plddt`: 平均 pLDDT 分数（0-100，越高越好）
- `mean_prmsd`: 平均预测 RMSD（Å，越低越好）

## 上下游关系
- **上游**: proteinmpnn_sequence_design → 序列
- **下游**: alphafold3_predict（仅对 IgFold pLDDT > 70 的候选做复合物验证）

## Pipeline 中的角色（Step 10b/16）
```
ProteinMPNN (50-100 sequences) → ESM2 PPL filter (PPL < 15) → 50
→ IgFold pLDDT filter (pLDDT > 70) → 10-15 → AF3 validation (ipTM ≥ 0.6)
```

## pLDDT 质量期望（按 Target Tier）
- **Tier 1 hotspots**（已知复合物界面残基）→ pLDDT > 70 预期，通过率高
- **Tier 3 hotspots**（计算预测表位）→ pLDDT 40-70 常见，需更多设计数量

## ⚠️ IgFold 只适用于抗体/纳米抗体序列
- IgFold 内部使用 AntiBERTy，只在抗体序列上训练
- **de novo binder（RFdiffusion 设计）不是抗体 → pLDDT 无意义**（实测 pLDDT 0.4-12.6）
- pipeline 中 `binder_type='de_novo'`（默认）时自动跳过 IgFold
- `binder_type='nanobody'` 或 `'antibody'` 时才启用 IgFold 过滤
- de novo binder 走 ESM2 PPL filter → 直接 AF3（无结构预筛选）

## 注意事项
- `do_renum=False`（ANARCI 包名冲突 anarci vs anarcii，后续修复）
- 容器重启时模型不会丢失（docker commit 固定 + /root/.cache volume mount）
- 批量调用时每个序列用单独 task（避免 OOM）
- `mean_plddt` 可能返回 None → pipeline 中用 `or 0` 防御
