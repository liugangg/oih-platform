# DiscoTope3 B细胞表位预测工作流

## 基本信息
- 容器：oih-discotope3
- 软件路径：/app/discotope3/discotope3/main.py
- GPU：NVIDIA_VISIBLE_DEVICES=1 → 容器内 cuda:0
- 模型：XGBoost ensemble + ESM embeddings
- 队列：GPU（VRAM ~6GB）
- 超时：600秒

## 功能
预测蛋白质结构上的B细胞表位倾向性（哪些残基最可能被抗体识别）。

## 参数
- `input_pdb`: 蛋白质PDB文件路径（抗原或抗体结构）
- `struc_type`: `solved`（实验结构）或 `alphafold`（AF2/AF3预测结构）
- `calibrated_score_epi_threshold`: 表位阈值，low=0.40（敏感）/ moderate=0.90（默认）/ high=1.50（严格）
- `multichain_mode`: 预测整个复合物所有链（默认 false）
- `cpu_only`: 强制CPU推理（默认 false，使用GPU）

## 输出
- CSV文件：每个残基的表位倾向性评分（calibrated score）
- PDB文件：高评分残基标注

## 上下游关系
- **上游**：fetch_pdb → input_pdb | alphafold3_predict → CIF→PDB转换后的文件
- **下游**：
  - 高评分表位残基 → rfdiffusion_design 的 hotspot_residues 参数
  - 表位信息 → binder_design_pipeline 的设计参考
  - 与 fpocket/P2Rank 口袋结果交叉验证

## 使用场景
1. **抗体靶点分析**：预测抗原表面哪些区域可被抗体靶向
2. **疫苗抗原设计**：选择高免疫原性的表位用于疫苗
3. **Binder设计指导**：用表位残基作为RFdiffusion的hotspot
4. **AF3验证**：比较AF3预测的interface与DiscoTope3预测的表位重叠度

## 注意事项
- CLI必须从 `/app/discotope3/discotope3/` 目录运行（bare imports）
- 容器内用 `python3`（不是 `python`）
- AlphaFold结构必须设 `struc_type=alphafold`（影响B-factor处理）
- 大型复合物（>1000残基）自动切换到CPU embedding

## DT3 分数分布（血泪经验）
- `DiscoTope-3.0_score` 原始分数范围通常 **0.001 ~ 0.5**（不同结构差异大）
- `calibrated_score` 经校准后范围更大（可到 6+），但 pipeline 用的是原始分数
- **绝对不要用固定阈值 0.7** — 大多数结构的最高分都到不了 0.5
- 正确做法：**adaptive threshold = max(scores[top_20%], 0.10)**

| 结构 | 残基数 | 最高DT3分 | top-20%阈值 | ≥阈值残基数 |
|------|--------|-----------|-------------|-------------|
| 2A91 chain A (HER2 ECD) | 506 | 0.361 | 0.142 | 102 |
| 1N8Z chain C (HER2+Ab) | 581 | 0.513 | ~0.15 | ~95 |

## known_epitope_override 关键参数（2026-03-21 验证）

### 阈值设置
- 不要用固定阈值（如0.7）— 不同结构的DT3分数范围差异很大
- 正确做法：`adaptive threshold = max(top 20% 分数, 0.10)`
- 2A91 实测：max score=0.361，固定0.7→0个高置信残基，override永远不触发

### 编号对齐（PDB vs UniProt）
- IEDB/RAG 返回的残基编号可能是 **UniProt 编号**，与 PDB 编号有偏移
- 2A91：PDB编号与UniProt编号有 **+22 偏移**（PDB res 245 ≈ UniProt res 267）
- 解决方案：尝试多个常见偏移量 `[0, ±22, ±23]`，取匹配最多的
- 每个比较允许 **±3 容差**（`abs(dt3_num + offset - rag_num) <= 3`）

### override 触发条件
- `overlap >= 2` 残基（不是3，避免漏判）
- 触发后直接用 overlap 残基作为 hotspot，跳过 6D 评分（步骤 4-6）
- 如果未触发，正常走 6D 复合评分 + Qwen 选择

### 2A91 HER2 验证结果
- DT3 adaptive threshold: **0.142**（top 20%）
- offset=0: 7 matches（直接匹配）
- offset=+22: **18 matches**（最佳，PDB→UniProt偏移）
- 最终 hotspots: `A245, A246, A157, A258, A256, A286, A255, A260`
- 已知命中：A245/A246 (Domain I/III界面), A286 (Domain II近端)

### 6D 评分公式（override 未触发时使用）
```
composite = p2rank(0.15) + sasa(0.15) + conservation(0.15) + rag(0.25) + electrostatics(0.10) + epitope(0.20)
```
epitope 维度：口袋残基在 8Å 内有 DT3 高分残基的比例（_compute_epitope_score_for_pocket）

## Target Tier Classification (2026-03-22)
- **Tier 1 (Known complex)**: Use `extract_interface_residues` → most reliable, experimentally validated
  - HER2 = Tier 1, use 1N8Z (trastuzumab complex) → skip DiscoTope3
- **Tier 2 (Homolog)**: Use homology transfer → moderate confidence
- **Tier 3 (Novel)**: Use DiscoTope3 + IEDB + RAG → computational prediction
  - DiscoTope3 is only used for Tier 3 targets (novel, no known complex)
  - For Tier 1, interface extraction from known complex is far more reliable

## ⚠️ 适用范围限制

DiscoTope3 预测抗原表面的 B细胞表位，只适用于抗体设计场景。

### 正确使用场景
- 靶点是蛋白质抗原 → 用 DiscoTope3
- 设计抗体/纳米抗体/binder → 用 DiscoTope3
- 无已知复合物结构（Tier 3）→ 用 DiscoTope3

### 禁止使用场景
- 小分子药物对接 → 用 fpocket/P2Rank
- 已有共晶结构（Tier 1）→ 直接用 extract_interface_residues
- 酶活性位点预测 → 用 fpocket + 保守性分析
