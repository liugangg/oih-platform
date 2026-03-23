# Binder Design Workflow — 统一决策链

## 完整流程（16步）
靶标输入 → Tier分类 → RAG检索 → 热点选择 → 域截取 →
RFdiffusion/BindCraft → ProteinMPNN → ESM2 → IgFold(仅抗体) →
AF3验证 → ADC组装

## 1. Tier 分级（pipeline.py KNOWN_COMPLEXES, _classify_target_tier）
- Tier1: KNOWN_COMPLEXES 命中（HER2/EGFR/PDL1等）→ extract_interface_residues
- Tier1-equivalent: RAG Layer1 搜到共晶/mutagenesis 结合界面
- Tier3: 无已知界面 → 多工具共识

## 2. RAG 两层搜索（pipeline.py _rag_search_pocket_context）
Layer 1 (PPI interface — 最高优先级):
  "{target} protein complex co-crystal structure binding partner"
  → 搜到实验验证残基 → 直接做 hotspot，跳过 DiscoTope3

Layer 2 (epitope fallback — 仅当 Layer1 无结果):
  "{target} epitope binding site"

关键规则：B-cell epitope ≠ binder design hotspot
- Epitope 预测免疫原性位点（抗体在免疫中结合哪里）
- Binder 设计需要 PPI 兼容界面（形状互补+疏水核心）
- 实证：CD36 DiscoTope3 A397-A400 → ipTM 0.33 全失败
  文献 CLESH domain E101/D106/E108/D109 才是正确的 TSP-1 界面

## 3. 6D Pocket Scoring（pipeline.py:837）
composite = p2rank(0.15) + sasa(0.15) + conservation(0.15)
          + rag(0.25) + electrostatics(0.10) + epitope(0.20)

注意 RAG 权重最高(0.25)，文献证据优先于计算预测

相关函数：
- _compute_sasa_score_for_pocket (line 414)
- _compute_epitope_score_for_pocket (line 562)
- _compute_rag_score (line 636)

## 4. Hotspot 选择规则
来源优先级：
1. 文献实验验证（共晶/mutagenesis）>>> 所有计算预测
2. 多工具共识（DiscoTope3 ∩ P2Rank ∩ FreeSASA ≥3 agree）> 单工具
3. 保守+表面+中等凹度 > 高暴露但平坦

空间聚类（pipeline.py _cluster_hotspots line 427）：
- Cα 距离 ≤ 15Å
- 每组最多 5 个残基
- 分散热点 → _multi_cluster_hotspots (line 482) 分组
- 每组独立跑 RFdiffusion
- 至少含 1 带电残基 + 1 疏水残基

## 5. 域截取（pipeline.py DOMAIN_REGISTRY line 37）
已注册靶点自动截取：
- HER2: Domain IV ~202aa
- CD36: extracellular_loop (30-439)
- EGFR: domain3 ~180aa
- PDL1: IgV ~168aa

规则：
- 全长 > 500aa 必须截取
- 截取到 ~200aa，hotspot 必须在范围内
- padding = 30aa 保留边界柔性
- AF3 速度提升 5-8x，精度持平或更高
  （HER2: 全长 ipTM=0.84 vs 截取 ipTM=0.86）

## 6. 双路并行设计
PathA: RFdiffusion (hotspot指定) → ProteinMPNN (8 seq/backbone) → ESM2
PathB: BindCraft (无hotspot，AF2自由探索)

两路都进 AF3 validation，择优
- RFdiffusion 擅长：有明确结构化 hotspot 的靶点
- BindCraft 擅长：无已知结合位点，表面探索

## 7. AF3 验证
- num_seeds=3
- 通过阈值 ipTM ≥ 0.6
- 域截取版本优先（更快更准）
- 动态超时：<500aa 1200s, 500-1000aa 2400s, >1000aa 3600s

## 8. ADC Assembly（仅通过 AF3 的设计）
FreeSASA → 表面暴露 Lys/Cys → linker_select → rdkit_conjugate
- Cys → maleimide_thiol
- Lys only → nhs_amine（7 条 SMARTS 覆盖伯胺+仲胺）
- DAR=4（标准）
- Linker library: /data/oih/oih-api/data/linker_library.json
- 3D embedding fallback: 大分子返回 2D SMILES (embedding_status="2d_only")

## 工具适用范围（避免误用）
- DiffDock: 仅小分子对接，已从 binder pipeline 移除
- IgFold: 仅 nanobody/antibody fold 预测，de novo binder 跳过
- GNINA/AutoDock: 仅小分子，不参与 binder 设计
- DiscoTope3: epitope 预测，作为 6D 评分的一个维度(0.20)，不单独决定 hotspot

## 案例参考
HER2 (Tier1): interface extraction → C558/C560/C571-C573
  → ipTM 0.86 (3/10 pass) → ADC DAR=4 covalent NHS-amine 成功
CD36 (Tier3 失败): DiscoTope3 A397-A400 → ipTM 0.33 (0/10)
CD36 (RAG-first 修正): CLESH E101/D106/E108/D109 → 待验证
