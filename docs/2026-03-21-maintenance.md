# 2026-03-21 维修记录

## Bug修复

### 1. DiscoTope3 模型路径问题
- 症状: Path models/ does not exist + Found 0/100 XGBoost files
- 原因: models.zip 解压后产生嵌套目录 models/models/
- 修复: mv /app/discotope3/models/models/*.json /app/discotope3/models/
- 持久化: /data/oih/model_cache/discotope3_models → /app/discotope3/models

### 2. DiscoTope3 threshold 过高
- 症状: 0 epitope residues returned
- 原因: 默认 threshold=0.9，实际分数普遍<0.5
- 修复: threshold 默认改为 0.5

### 3. known_epitope_override 不触发
- 症状: override 始终不触发，hotspot 仍用几何口袋
- 原因1: DT3 阈值 0.7 过严，2A91 max score ~0.4
- 修复1: 改为 adaptive threshold = max(top 20%, 0.10)
- 原因2: RAG 用 UniProt 编号，DT3 用 PDB 编号，offset=22
- 修复2: 模糊匹配 ±3 + 自动检测 offset(0, ±22, ±23)
- 原因3: overlap 要求 >=3 过严
- 修复3: 改为 >=2

### 4. RFdiffusion 超时（1小时）
- 症状: oih-rfdiffusion timed out after 3600s
- 原因1: 8个 hotspot 跨越不同 domain（A157+A245-A286），空间分散40+Å
- 原因2: HER2 ECD 506残基，每设计计算量大
- 修复1: timeout 3600→7200s
- 修复2: 空间聚集过滤 _cluster_hotspots()，centroid距离<=15Å，最多5个hotspot
- 修复3: 超时后恢复已生成PDB，不整体失败

### 5. BindCraft exit 1 误判
- 症状: BindCraft failed (exit 1)
- 原因1: BindCraft 对难靶标0设计通过内部AF2过滤，exit 1 是正常行为
- 原因2: router 把任何 exit 1 当硬错误
- 修复: 有输出文件就返回partial results，也搜索 Accepted/Ranked/MPNN 子目录

### 6. BindCraft num_designs pydantic 验证失败
- 症状: 1 validation error for BindCraftRequest
- 原因: schema要求 ge=10，pipeline传了5
- 修复: num_designs=5 → 10

### 7. IgFold 依赖链问题
- 症状: ModuleNotFoundError: pytorch_lightning / transformers / abnumber
- 修复1: transformers>=4.28,<4.40 + torchmetrics>=0.11,<1.4（与torch 2.1兼容）
- 修复2: do_renum=False（跳过ANARCI，anarcii≠anarci包名冲突）
- 修复3: IgFoldOutput.prmsd（不是plddt），转换为pseudo-pLDDT
- 修复4: save_pdb无.pdb后缀 → 手动rename
- 模型: 打包在pip包内（igfold/trained_models/），docker commit固定

### 8. ESM 模型缓存丢失
- 症状: 每次重建容器重新下载2.5-7.3GB模型
- 修复: volume mount /data/oih/model_cache/esm_torch_hub:/root/.cache/torch/hub
- 涉及容器: oih-esm, oih-discotope3

### 9. ESM-1v mutant scan 超时
- 症状: task timed out after 600s
- 原因: ESM-1v 首次加载需下载7.3GB模型
- 修复: TIMEOUT_ESM 600→1800s，模型下载后缓存到host volume

### 10. pipeline selected_id 变量作用域
- 症状: cannot access local variable 'selected_id'
- 原因: override分支不设置selected_id，但后续日志在两个分支外引用
- 修复: logger.info移入else分支

### 11. PubMed efetch 阻塞 RAG
- 症状: /rag/search 请求挂起30+秒
- 原因: PubMed efetch无超时保护
- 修复: asyncio.wait_for(search, 10s) + asyncio.wait_for(fetch, 15s)

### 12. SAbDab 8MB TSV 阻塞事件循环
- 症状: RAG搜索超时
- 原因: SAbDab 下载全量TSV(20K行)并同步解析
- 修复: asyncio.gather 总超时30s，SAbDab解析中定期yield

## 模型缓存清单
| 模型 | 大小 | 宿主机路径 |
|------|------|-----------|
| ESM-IF1 (DiscoTope3) | 1.6GB | /data/oih/model_cache/torch/hub/checkpoints/ |
| ESM2-650M | 2.5GB | /data/oih/model_cache/esm_torch_hub/checkpoints/ |
| ESM-1v | 7.3GB | /data/oih/model_cache/esm_torch_hub/checkpoints/ |
| DiscoTope3 XGBoost | 101文件 | /data/oih/model_cache/discotope3_models/ |
| IgFold + AntiBERTy | pip内置 | docker commit固定（oih-igfold:latest） |

## Pipeline 运行记录
| 任务ID | 配置 | 结果 | 失败原因 |
|--------|------|------|---------|
| decafe6b | 6D评分首次，20设计 | failed | 重启中断AF3 4/5 |
| 72bcadc9 | override threshold修复 | failed | override触发但selected_id未定义 |
| ffe89ad3 | selected_id修复 | failed | 同上（修复前代码） |
| 60b4d10a | override触发，8个hotspot | failed | RFdiffusion超时3600s（18/20 PDB已生成） |
| 45ff6582 | 6个hotspot，2h超时 | cancelled | RFdiffusion仍太慢（3/10 PDB in 2h） |
| 17449435 | 5个聚集hotspot，10设计 | running | 空间聚集修复，等待结果 |

## 关键参数总结
- DiscoTope3 threshold: 0.5（calibrated_score_epi_threshold，不用0.9）
- DT3 raw score range: 0.001-0.5（不同结构不同，2A91 max=0.36）
- override 触发: adaptive top20% + fuzzy±3 + offset检测 + overlap>=2
- HER2 2A91 PDB↔UniProt offset: +22
- hotspot 空间聚集: centroid<=15Å，max 5 residues
- RFdiffusion timeout: 7200s
- RFdiffusion num_designs: 10（HER2大靶标用10不用20）
- BindCraft: exit 1正常（0设计过滤），不abort pipeline
