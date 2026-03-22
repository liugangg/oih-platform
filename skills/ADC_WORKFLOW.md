# ADC 药物设计工作流

## 概述
ADC（抗体偶联药物）= 抗体（靶向性）+ Linker（连接子）+ Payload（细胞毒素）
平台支持从靶点PDB到完整ADC小分子结构的全流程设计。

## 工具链

### 1. freesasa — 抗体偶联位点预测
- 路由：/api/v1/design/freesasa
- 队列：CPU
- 输入：抗体PDB路径（来自rfdiffusion/proteinmpnn输出）
- 输出：暴露Lys/Cys列表，含SASA值（>40Å²为推荐偶联位点）
- 上游：rfdiffusion → proteinmpnn → af3验证后的抗体PDB
- 下游：rdkit_conjugate的conjugation_site参数

### 2. linker_select — Linker筛选
- 路由：/api/v1/adc/linker_select
- 队列：同步返回
- 输入：payload_smiles, cleavable(bool), reaction_type, compatible_payload, clinical_status
- 输出：推荐linker列表（含SMILES/DAR/approved_adcs/stability_note）
- 库文件：/data/oih/oih-api/data/linker_library.json（20条临床验证linker）
- 默认优先级：approved > clinical > research
- 上游：fetch_molecule获取payload_smiles
- 下游：rdkit_conjugate的linker_smiles/linker_name参数

### 3. rdkit_conjugate — 共价偶联
- 路由：/api/v1/adc/rdkit_conjugate
- 队列：CPU
- 输入：job_name, antibody_pdb, conjugation_site, linker_smiles, linker_name, payload_smiles, reaction_type
- 输出：adc_smiles, covalent(bool), reaction_type_used, dar_range, output_sdf, atom_count
- reaction_type="auto"时自动检测
- covalent=false时fallback到dot-disconnected（仍可用于ADMET）
- 上游：freesasa(conjugation_site) + linker_select(linker_smiles) + fetch_molecule(payload_smiles)
- 下游：chemprop ADMET预测 + gnina脱靶评估

## 7类偶联反应化学
| 反应类型 | linker端 | payload端 | 代表linker | 代表ADC |
|---------|---------|----------|-----------|--------|
| maleimide_thiol | 马来酰亚胺 | 巯基/Cys | MC-VC-PABC, SMCC | Adcetris, Kadcyla |
| nhs_amine | NHS酯 | 赖氨酸/胺基 | VC-PABC | 多数lysine偶联ADC |
| hydrazone | 酰肼 | 酮基 | Hydrazone, AcBut | Mylotarg |
| oxime | 羟胺 | 醛/酮 | aminooxy-PEG | 在研 |
| disulfide | 吡啶二硫 | 巯基 | SPDP, CL2 | DM4类ADC |
| dbco_azide | DBCO/炔 | 叠氮 | DBCO-PEG4 | 在研定点偶联 |
| transglutaminase | 酶促Q295 | 胺基 | TGase-cadaverine | 在研 |

## ADC设计完整流程 — binder_design_pipeline（一键7步）

**推荐方式：直接调用 binder_design_pipeline，支持 pdb_id 参数自动下载PDB**

```
Step 0: fetch_pdb(pdb_id="1N8Z") → 自动下载靶蛋白PDB
Step 1: RFdiffusion → 生成蛋白骨架（binder_design模式）
Step 2: ProteinMPNN → 为每个骨架设计序列（8 seq/backbone）
Step 3: AlphaFold3 → top5序列复合物预测，ipTM≥0.75=high，≥0.6=low_confidence
Step 4: FreeSASA → 最佳AF3结构的表面SASA分析，SASA>40Å²的Lys/Cys为偶联位点
Step 5: Linker Select → 根据位点类型选择：Cys→maleimide_thiol / Lys→NHS_ester，cleavable，compatible MMAE
Step 6: Fetch Payload → PubChem获取MMAE SMILES + MW
Step 7: RDKit Conjugate → antibody + linker + payload → ADC SMILES，DAR=4
```

输出 adc_design 字段：nanobody_sequence, iptm, conjugation_site, linker, payload, dar, adc_smiles, adc_structure_path

**偶联化学选择规则：**
- top偶联位点是 Cys → preferred_chemistry = maleimide_thiol
- top偶联位点是 Lys → preferred_chemistry = NHS_ester
- DAR=4 是ADC标准药物抗体比
- MMAE（monomethyl auristatin E）为标准payload，DM1为备选

## 常用Payload
- MMAE：微管蛋白抑制剂，Adcetris使用，需maleimide/PABC linker
- DM1/DM4：美登素，Kadcyla使用，需SMCC/SPDB linker
- SN-38：拓扑异构酶抑制剂，Trodelvy使用，需CL2A linker
- DXd：Enhertu使用，需GGFG linker
- Calicheamicin：Mylotarg使用，需hydrazone linker

## 注意事项
- maleimide端连抗体Cys，不是payload
- MMAE与VC-PABC：PABC通过氨基甲酸酯连MMAE N端胺基
- rdkit_conjugate生成的是linker-payload片段，用于ADMET
- covalent=false不影响后续ADMET预测流程

## Pipeline 实现细节

### 进度映射
| Steps | Progress % | 阶段 |
|-------|-----------|------|
| Step 0 (fetch_pdb) | 2% | 获取靶蛋白 |
| Step 1 (RFdiffusion) | 5-50% | 骨架生成 |
| Step 2 (ProteinMPNN) | 50-75% | 序列设计 |
| Step 3 (AF3 validation) | 75-95% | 结构验证 |
| Step 4-7 (ADC) | 80-100% | ADC构建 |

### 错误隔离
每步 try/except，单步失败设置 `adc_design.partial=true`，不阻塞后续步骤。
最终返回所有完成步骤的结果 + 失败步骤的错误信息。

### AF3 验证策略
- 从 MPNN 所有设计中按 score 排序取 top 5
- 逐个提交 AF3 复合物预测（binder + target 双链）
- 每个 AF3 任务间隔 5s 避免 GPU OOM
- ipTM 分级：≥0.75 = high，≥0.6 = low_confidence，<0.6 = low
- 最终按 ipTM 降序排列，选最优进入 ADC 步骤

### RFdiffusion PDB 预处理
binder_design 模式自动对靶蛋白 PDB 进行：
- 去除 HETATM（水/配体/糖基化修饰）
- 顺序重编号消除残基间隙（gap）
- hotspot 残基号自动映射到新编号

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## ⚠️ 注意事项（自动同步自 CLAUDE.md）

- Step 3: AF3 验证 — top5 MPNN → AF3 复合物 → ipTM 分级（≥0.75 high / ≥0.6 low_confidence）
- 每步 try/except，单步失败标记 `partial: true` 不阻塞后续
- AF3 任务间隔 5 秒避免 GPU OOM，timeout 1800s
- Step 1-3: RFdiffusion → ProteinMPNN → AF3 验证（ipTM ≥0.75 high / ≥0.6 low_confidence）
- 每步 try/except，单步失败标记 partial，不阻塞后续

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->

## NHS-amine Linker 支持（2026-03-22）
de novo binder 通常无游离 Cys → 用 NHS-amine (Lys 偶联)

| Linker | 类型 | 裂解 | DAR | 适用 |
|--------|------|------|-----|------|
| NHS-PEG4 | nhs_amine | 不可裂解 | 2-4 | Lys 偶联，类似 T-DM1 |
| NHS-PEG2-Val-Cit-PABC | nhs_amine | cathepsin B | 2-4 | Lys 偶联 + 可裂解 |

### 选择规则
- FreeSASA 检测到 Cys (SASA>40) → maleimide_thiol
- 只有 Lys → nhs_amine
- De novo binder 通常只有 Lys → 默认 nhs_amine

## FreeSASA 注意事项
- **不接受 CIF 文件** — 必须先转 PDB（BioPython MMCIFParser → PDBIO）
- pipeline 中已自动处理 CIF→PDB 转换

## RDKit 3D Embedding 限制
- 大分子 ADC EmbedMolecule 可能失败
- fallback: 返回 2D SMILES (`embedding_status="2d_only"`)，任务标记 completed
