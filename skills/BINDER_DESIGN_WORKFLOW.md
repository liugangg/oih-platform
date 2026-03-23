# Binder Design Workflow — 统一决策链

## 完整流程
靶标输入 → Tier分类 → RAG 2层搜索 → PeSTo PPI预测 →
Hotspot选择 → 域截取 → RFdiffusion/BindCraft → ProteinMPNN →
ESM2 → AF3验证 → ADC组装

## 1. Tier 分级
- Tier1: KNOWN_COMPLEXES命中 或 RAG Layer1搜到共晶/mutagenesis界面
  → extract_interface_residues → 直接做hotspot
- Tier3: 无已知界面 → PeSTo + RAG + conservation 多工具共识

## 2. RAG 两层搜索（所有靶点必须先执行）
Layer 1 (PPI interface — 最高优先级):
  "{target} protein complex co-crystal binding partner interface residues"
  → 搜到实验验证残基 → 直接用，跳过计算预测

Layer 2 (epitope fallback — 仅当Layer1无结果):
  "{target} epitope binding site"

关键规则：B-cell epitope ≠ binder design hotspot
- DiscoTope3/IEDB 预测免疫原性，不适合binder hotspot
- CD36实证: DiscoTope3 A397-400 → ipTM=0.33全败
- Baker lab从未用epitope工具选hotspot

## 3. PeSTo PPI 界面预测（替代P2Rank+DiscoTope3）
- ROC AUC 0.92（vs MaSIF-site 0.80）
- 输入: PDB单链 → 输出: 每个残基PPI interface概率(0-1)
- 部署: proteinmpnn容器内 /app/pesto/
- 重要: 对复合物PDB必须提取靶蛋白单链再跑
  （复合物跑会压低已占界面的分数，如PD-L1从0.44→0.99）

### PeSTo 已验证靶点难度表
| 靶点 | PDB | Chain | Max PPI | >0.5残基 | 难度 |
|------|-----|-------|---------|----------|------|
| TrkA | 1HE7 | A | 0.999 | 59 | 极容易 |
| PD-L1 | 4ZQK | A单链 | 0.994 | 23 | 容易 |
| Nectin-4 | 4GJT | A | 0.966 | 56 | 容易 |
| CD36 | 5LGD | A | 0.865 | 8 | 中等 |
| EGFR | 1YY9 | A单链 | 0.759 | 19 | 中等 |
| HER2 | 1N8Z | full | 0.668 | 3 | Tier1已完成 |
| TROP2 | 7PEE | full | 0.422 | 0 | 困难 |

## 4. Hotspot 选择规则
来源优先级:
1. 文献实验验证（共晶/mutagenesis）>>> 所有计算预测
2. PeSTo PPI interface (score > 0.5) > 单一计算工具
3. 保守+表面+中等凹度 > 高暴露但平坦

空间聚类: _cluster_hotspots()
- Cα距离 ≤ 15Å，每组最多5个残基
- 分散热点 → _multi_cluster_hotspots() 分组独立设计
- 至少含1带电残基 + 1疏水残基

## 5. 评分公式（PPI优化版，替代旧6D）
rag(0.30) + pesto_ppi(0.25) + conservation(0.20) +
sasa(0.10) + electrostatics(0.15)

已移除:
- P2Rank (小分子口袋工具，不适合PPI)
- DiscoTope3 epitope (B-cell epitope ≠ PPI interface)
- DiffDock (小分子对接工具，已从pipeline移除)

## 6. 域截取（DOMAIN_REGISTRY）
已注册: HER2 Domain IV / CD36 CLESH region / EGFR Domain III / PDL1 IgV
规则: 全长>500aa必须截取到~200aa
好处: AF3速度5-8x，精度持平或更高

## 7. 双路并行设计
PathA: RFdiffusion (hotspot指定) → ProteinMPNN → ESM2
PathB: BindCraft (无hotspot，AF2自由探索)
两路都进AF3 validation，择优

## 8. AF3 验证
- num_seeds=3
- 当前阈值 ipTM >= 0.6
- 动态超时: <500aa 1200s, 500-1000aa 2400s, >1000aa 3600s

## 9. ADC Assembly（仅通过AF3的设计）
FreeSASA → 表面暴露Lys → linker_select → rdkit_conjugate
DAR=4, NHS-PEG4-MMAE, 7条NHS-amine SMARTS

## 10. 工具适用范围
- DiffDock: 已移除，仅小分子
- P2Rank/fpocket: 仅小分子docking pipeline
- DiscoTope3: 不参与binder评分，仅作辅助参考
- IgFold: 仅nanobody/antibody，de novo binder跳过
- PeSTo: PPI interface预测，替代P2Rank+DiscoTope3

## 案例参考
HER2 (Tier1): C558-C573 → ipTM=0.86, 3/10 pass, ADC成功
CD36 DiscoTope3: A397-400 → ipTM=0.33, 0/10 全败
CD36 PeSTo: A187-194 → 待验证
CD36 CLESH文献: E101/D106/E108/D109 → 待验证
