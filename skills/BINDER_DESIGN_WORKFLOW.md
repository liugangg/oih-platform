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

## 11. 多靶点并行调度策略

### GPU 资源模型 (RTX 4090, 44GB)
| 工具 | VRAM | 时间 | 可并行 |
|------|------|------|--------|
| RFdiffusion | 4-10GB | 20min | 3个并行 (共~20GB) |
| ProteinMPNN | 4GB | 2min | 可与RFdiff并行 |
| ESM2 | 6GB | 5min | 可与MPNN并行 |
| AF3 | 20GB | 15min | 必须独占 |
| BindCraft | 16GB | 30min | 最多+1个小任务 |
| PeSTo | 0 (CPU) | 10s | 不占GPU |
| FreeSASA | 0 (CPU) | 5s | 不占GPU |

### 最优调度：pipeline式流水线
```
GPU时间线:
─────────────────────────────────────────────────
RFdiff(靶1) + RFdiff(靶2) + RFdiff(靶3)  ← 3并行
    ↓              ↓              ↓
MPNN(靶1) + RFdiff(靶4) + MPNN(靶2)      ← 穿插
    ↓                              ↓
AF3(靶1_val0) + MPNN(靶3)                ← AF3独占+小任务
    ↓
AF3(靶1_val1)
    ↓
AF3(靶2_val0) + MPNN(靶4)
    ...
─────────────────────────────────────────────────
CPU时间线（完全并行，不等GPU）:
PeSTo(所有靶点) | RAG(所有靶点) | FreeSASA | ipSAE计算
─────────────────────────────────────────────────
```

### Qwen 调度规则
1. **Pipeline 在 CPU 队列**：pipeline 是编排器，不占 GPU slot
2. **GPU semaphore=3**：最多3个GPU任务并行（VRAM-based routing 自动判断）
3. **RFdiffusion 可3并行**：每个4-10GB，3个共20-30GB < 44GB
4. **AF3 必须独占**：20GB，同时最多跑1个AF3 + 1个小任务(MPNN/ESM2)
5. **CPU任务随时跑**：PeSTo/FreeSASA/RAG/RDKit 不受GPU限制
6. **失败恢复**：RFdiffusion 完成的 backbone 保留在 outputs/，MPNN 可以手动补跑

### 批量提交策略
当用户要求多个靶点时：
```python
# 正确：一次提交所有，让调度器自动排队
for target in targets:
    submit_pipeline(target)  # 全部 pending，按 VRAM 自动调度

# 错误：等一个完了再提下一个
for target in targets:
    submit_pipeline(target)
    wait_until_complete()  # 浪费GPU空闲时间
```

### 失败恢复模式
如果 pipeline 中途失败（如 MPNN timeout）：
1. 检查哪一步完成了（RFdiffusion backbone 在 outputs/ 里）
2. 手动补跑失败步骤（不需要重跑 RFdiffusion）
3. 示例：CLESH v4 RFdiffusion ✅ → MPNN timeout → 手动提交 9 个 MPNN 任务

## 案例参考
HER2 (Tier1): C558-C573 → ipTM=0.86, 3/10 pass, ADC成功
CD36 DiscoTope3: A397-400 → ipTM=0.33, ipSAE=0.000, 0/10 全败
CD36 PeSTo v5: A187-194 → ipTM=0.55, ipSAE=0.193, 最佳CD36设计
CD36 PeSTo v6 n=50: ipTM=0.18, 更多≠更好（稀释了探索聚焦）
CD36 CLESH: E101/D106/E108/D109 → MPNN timeout，手动补跑中

## 域截取自主决策规则（不依赖硬编码 DOMAIN_REGISTRY）

当 DOMAIN_REGISTRY 没有该靶点时，Qwen 应自主确定截取范围：

### Step 1: 查 UniProt domain 注释
- Signal peptide（截掉）
- Transmembrane（截掉）
- Topological domain: Extracellular（保留）
- Domain: Ig-like D1/D2/D3

### Step 2: 确定 hotspot 所在的最小结构域
- 截取该 domain ± 30 残基 buffer

### Step 3: 大小验证
- 目标: 100-250aa（AF3 最高效区间）
- <100aa: 扩大 buffer; >500aa: 必须截取

### Step 4: 结构完整性
- 不切断 beta-sheet/alpha-helix
- 二硫键 Cys-Cys 配对都要在范围内
- 边界选 loop/linker 区域

### 截取公式
center = mean(hotspots); range = (min(hotspots)-50, max(hotspots)+50)
如果 <80aa，扩大到 center ± 60

### 案例
- HER2 C558-C573 → Domain IV (488-630) 142aa → ipTM=0.86
- CD36 PeSTo A187-194 → (140-240) 100aa → 待验证
- CD36 全长 469aa → ipTM=0.33 失败
