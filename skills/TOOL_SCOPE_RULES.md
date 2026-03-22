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

### 历史教训
- IgFold 误用于 de novo binder → pLDDT 全部 0.4-12.6 → 0 个候选进入 AF3
- 根因：IgFold 使用 AntiBERTy，只在抗体数据集上训练
- 修复：pipeline 加入 binder_type 参数，de_novo 时跳过 IgFold
- FreeSASA 不接受 CIF → 需先转 PDB
- GPU Semaphore 必须=1，否则两个大模型并发 OOM
