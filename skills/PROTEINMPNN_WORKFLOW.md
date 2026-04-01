# ProteinMPNN 完整工作流程（已验证）

## 环境信息
- 容器名：oih-proteinmpnn
- Python：python3（3.10.12）
- 主脚本：/app/ProteinMPNN/protein_mpnn_run.py
- 模型权重：容器内 /app/ProteinMPNN/vanilla_model_weights/（已内置，无需外部挂载）
- 示例：/app/ProteinMPNN/examples/

## ⚠️ GPU注意事项
- NVIDIA_VISIBLE_DEVICES=1 映射宿主机GPU1为容器内GPU0
- ProteinMPNN 自动使用 CUDA，无需手动指定 device
- 显存需求约 4GB

## 模型权重说明
| 路径 | 模型 | 用途 |
|------|------|------|
| vanilla_model_weights/v_48_002.pt | noise=0.02 | 高精度，多样性低 |
| vanilla_model_weights/v_48_010.pt | noise=0.10 | 平衡 |
| vanilla_model_weights/v_48_020.pt | noise=0.20 | 推荐，平衡精度和多样性 |
| vanilla_model_weights/v_48_030.pt | noise=0.30 | 高多样性 |
| ca_model_weights/v_48_020.pt | CA-only | RFdiffusion骨架输出专用 |
| soluble_model_weights/v_48_020.pt | soluble | 可溶性蛋白优化 |

## 完整推理命令

### 1. 标准序列设计（全链）✅已验证
```bash
docker exec oih-proteinmpnn python3 /app/ProteinMPNN/protein_mpnn_run.py \
  --pdb_path /data/oih/inputs/1IVO.pdb \
  --out_folder /data/oih/outputs/proteinmpnn/ \
  --num_seq_per_target 4 \
  --sampling_temp 0.1 \
  --model_name v_48_020
```
验证结果：4条序列，长度1115残基，11秒完成 ✅

### 2. 指定设计链（多链复合物只设计部分链）

**⚠️ 2026-03-26 规则：使用API调用时，chains_to_design 留默认 "auto"。**
Router会自动检测最短链（=binder）。不要硬编码 "A"。

```bash
# 直接docker exec时，必须手动确认binder是哪条链：
# 先检查：python3 -c "import gemmi; [print(c.name, sum(1 for r in c)) for c in gemmi.read_structure('complex.pdb')[0]]"
# 最短链 = binder = 要设计的链
docker exec oih-proteinmpnn python3 /app/ProteinMPNN/protein_mpnn_run.py \
  --pdb_path /data/oih/inputs/complex.pdb \
  --pdb_path_chains "B" \
  --out_folder /data/oih/outputs/proteinmpnn/ \
  --num_seq_per_target 8 \
  --sampling_temp 0.1
# 验证：FASTA第一条序列长度应为60-120aa(binder)，不是几百aa(target)
```

### 3. 固定部分残基（保留活性位点/关键残基）
```bash
# 先用helper脚本生成fixed_positions文件
docker exec oih-proteinmpnn python3 /app/ProteinMPNN/helper_scripts/make_fixed_positions_dict.py \
  --input_path /data/oih/inputs/ \
  --output_path /data/oih/outputs/proteinmpnn/fixed.jsonl \
  --chain_list "A" \
  --position_list "30 31 32 33 55 56"

# 再运行设计
docker exec oih-proteinmpnn python3 /app/ProteinMPNN/protein_mpnn_run.py \
  --pdb_path /data/oih/inputs/target.pdb \
  --fixed_positions_jsonl /data/oih/outputs/proteinmpnn/fixed.jsonl \
  --out_folder /data/oih/outputs/proteinmpnn/ \
  --num_seq_per_target 8 \
  --sampling_temp 0.1
```

### 4. 处理RFdiffusion输出（CA-only骨架）
```bash
docker exec oih-proteinmpnn python3 /app/ProteinMPNN/protein_mpnn_run.py \
  --pdb_path /data/oih/outputs/rfdiffusion/design_0.pdb \
  --ca_only \
  --out_folder /data/oih/outputs/proteinmpnn/ \
  --num_seq_per_target 8 \
  --sampling_temp 0.1 \
  --model_name v_48_020
```

### 5. 可溶性蛋白优化设计
```bash
docker exec oih-proteinmpnn python3 /app/ProteinMPNN/protein_mpnn_run.py \
  --pdb_path /data/oih/inputs/target.pdb \
  --use_soluble_model \
  --out_folder /data/oih/outputs/proteinmpnn/ \
  --num_seq_per_target 8 \
  --sampling_temp 0.15
```

### 6. 对已有序列打分（验证序列与骨架匹配度）
```bash
docker exec oih-proteinmpnn python3 /app/ProteinMPNN/protein_mpnn_run.py \
  --pdb_path /data/oih/inputs/target.pdb \
  --score_only 1 \
  --path_to_fasta /data/oih/inputs/sequences.fasta \
  --out_folder /data/oih/outputs/proteinmpnn/ \
  --save_score 1
```

## 关键参数说明
| 参数 | 说明 | 推荐值 |
|------|------|--------|
| --sampling_temp | 序列多样性，越大越多样 | 0.1~0.2 |
| --num_seq_per_target | 每个结构生成序列数 | 4~20 |
| --model_name | 模型版本 | v_48_020 |
| --ca_only | 仅用CA原子，用于RFdiffusion输出 | 视情况 |
| --use_soluble_model | 可溶性蛋白优化 | 视情况 |
| --batch_size | 批大小，GPU显存不足时调小 | 1 |
| --omit_AAs | 排除某些氨基酸，默认排除X | 'CX'可排除Cys |
| --save_score | 保存每个位置的log概率分数 | 1（调试时用）|

## 输出文件结构
```
/data/oih/outputs/proteinmpnn/
├── seqs/
│   └── 1IVO.fa          # 设计的序列（FASTA格式）
├── scores/              # 如果开启--save_score
│   └── 1IVO.npy
└── probs/               # 如果开启--save_probs
    └── 1IVO.npy
```

## FASTA输出格式
```
>1IVO, score=1.2345, global_score=1.2345, seq_recovery=0.xx
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLVLNSPPSRENRVSSRQFVNHEEMVHHAETREAQHNAQARAAVTGSQPTIRRSAQN
```

### Pipeline 中解析 FASTA（_parse_mpnn_fasta）
Router 返回 `{fasta_file, output_dir}`，不返回结构化 `sequences` 列表。
Pipeline 用 `_parse_mpnn_fasta(fasta_path)` 解析 FASTA 文件：
- 读取 header 中 `score=X` 和 `global_score=Y`
- 返回 `[{"sequence": "...", "score": 1.23, "global_score": 1.23}, ...]`
- 按 score 升序排列（越低越好），取 top 5 送 AF3 验证

## 完整蛋白设计流程
```
RFdiffusion（骨架设计）
    ↓ backbone.pdb
ProteinMPNN（序列设计）--ca_only
    ↓ sequences.fa
AlphaFold3（结构验证，看序列能否折叠回目标骨架）
    ↓ predicted.pdb
RMSD分析（设计骨架 vs AF3预测结构）
```

## Helper脚本列表（/app/ProteinMPNN/helper_scripts/）
```bash
docker exec oih-proteinmpnn ls /app/ProteinMPNN/helper_scripts/
# make_fixed_positions_dict.py  — 生成固定残基字典
# make_tied_positions_dict.py   — 生成绑定残基字典（对称设计）
# make_bias_AA.py               — 生成氨基酸偏好字典
# parse_multiple_chains.py      — 批量解析多链PDB
# assign_fixed_chains.py        — 指定固定链
```

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## ⚠️ 注意事项（自动同步自 CLAUDE.md）

- 容器内 NVIDIA_VISIBLE_DEVICES=1，永远用 device=0 / gpu_id=0（不要用1）

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->
