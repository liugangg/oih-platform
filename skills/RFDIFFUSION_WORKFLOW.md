# RFdiffusion 完整工作流程（已验证）

## 环境信息
- 容器名：oih-rfdiffusion
- 镜像：rfdiffusion:latest
- Python：python3（3.10.12），注意不是 python，没有 python 命令
- 推理脚本：/app/RFdiffusion/scripts/run_inference.py
- 示例脚本：/app/RFdiffusion/examples/

## ⚠️ 关键Bug修复（必须在新容器部署后执行）

### Bug：unconditional设计时默认input_pdb路径错误
model_runners.py 中默认 pdb 路径用相对路径，导致找不到文件：
```
FileNotFoundError: .../rfdiffusion/inference/../../examples/input_pdbs/1qys.pdb
```

**修复命令：**
```bash
# 备份原文件
docker exec oih-rfdiffusion bash -c "cp \
  /usr/local/lib/python3.10/dist-packages/rfdiffusion/inference/model_runners.py \
  /usr/local/lib/python3.10/dist-packages/rfdiffusion/inference/model_runners.py.bak"

# 修复路径
docker exec oih-rfdiffusion bash -c "sed -i \
  's|../../examples/input_pdbs/1qys.pdb|/app/RFdiffusion/examples/input_pdbs/1qys.pdb|' \
  /usr/local/lib/python3.10/dist-packages/rfdiffusion/inference/model_runners.py"

# 验证
docker exec oih-rfdiffusion bash -c "grep '1qys' \
  /usr/local/lib/python3.10/dist-packages/rfdiffusion/inference/model_runners.py"
```

## ⚠️ 模型路径问题（docker-compose.yml已修复）
- 模型实际位置（宿主机）：/data/rfdiffusion/models/
- 容器内挂载路径：/data/models/rfdiffusion
- docker-compose.yml 正确配置：
  ```yaml
  volumes:
    - /data/rfdiffusion/models:/data/models/rfdiffusion:ro
  ```

## ⚠️ GPU注意事项
- NVIDIA_VISIBLE_DEVICES=1 映射宿主机GPU1为容器内GPU0
- RFdiffusion 自动使用 CUDA device 0，无需手动指定
- 显存需求约 8GB

## 可用模型文件（/data/rfdiffusion/models/）
| 文件 | 用途 |
|------|------|
| Base_ckpt.pt | 基础蛋白骨架设计（最常用）|
| Base_epoch8_ckpt.pt | Base模型早期checkpoint |
| Complex_base_ckpt.pt | 蛋白-蛋白复合物/Binder设计 |
| Complex_Fold_base_ckpt.pt | 复合物+fold conditioning |
| Complex_beta_ckpt.pt | Complex beta版本 |
| ActiveSite_ckpt.pt | 活性位点设计 |
| InpaintSeq_ckpt.pt | 序列inpainting |
| InpaintSeq_Fold_ckpt.pt | 序列inpainting+fold |
| RF_structure_prediction_weights.pt | 结构预测权重（内部使用）|

## 示例脚本位置
/app/RFdiffusion/examples/ 内有完整官方示例，使用前先查阅：
```bash
docker exec oih-rfdiffusion bash -c "ls /app/RFdiffusion/examples/"
docker exec oih-rfdiffusion bash -c "cat /app/RFdiffusion/examples/design_ppi.sh"
```

## 完整推理命令

### 1. Unconditional设计（无条件从头设计）✅已验证
```bash
docker exec oih-rfdiffusion python3 /app/RFdiffusion/scripts/run_inference.py \
  inference.output_prefix=/data/oih/outputs/rfdiffusion/unconditional \
  inference.model_directory_path=/data/models/rfdiffusion \
  'contigmap.contigs=[100-200]' \
  inference.num_designs=10
```
注意：不需要 inference.input_pdb，bug修复后自动使用默认1qys.pdb

### 2. Motif Scaffolding（固定活性位点设计支架）
```bash
docker exec oih-rfdiffusion python3 /app/RFdiffusion/scripts/run_inference.py \
  inference.output_prefix=/data/oih/outputs/rfdiffusion/motif \
  inference.model_directory_path=/data/models/rfdiffusion \
  inference.input_pdb=/data/oih/inputs/5TPN.pdb \
  'contigmap.contigs=[10-40/A163-181/10-40]' \
  inference.num_designs=10
```

### 3. Binder设计（蛋白-蛋白相互作用）
```bash
docker exec oih-rfdiffusion python3 /app/RFdiffusion/scripts/run_inference.py \
  inference.output_prefix=/data/oih/outputs/rfdiffusion/binder \
  inference.model_directory_path=/data/models/rfdiffusion \
  inference.input_pdb=/data/oih/inputs/target.pdb \
  'contigmap.contigs=[A1-150/0 70-100]' \
  'ppi.hotspot_res=[A59,A83,A91]' \
  denoiser.noise_scale_ca=0 \
  denoiser.noise_scale_frame=0 \
  inference.num_designs=10
```

### 4. Partial Diffusion（对已有结构做局部扩散）
```bash
docker exec oih-rfdiffusion python3 /app/RFdiffusion/scripts/run_inference.py \
  inference.output_prefix=/data/oih/outputs/rfdiffusion/partial \
  inference.model_directory_path=/data/models/rfdiffusion \
  inference.input_pdb=/data/oih/inputs/input.pdb \
  'contigmap.contigs=[A1-100]' \
  diffuser.partial_T=10 \
  inference.num_designs=10
```

## Contig语法说明（从官方示例提取）
| 语法 | 含义 |
|------|------|
| `[100-200]` | 设计100-200残基新蛋白（随机长度）|
| `[100]` | ❌错误，会被解析成整数，必须用 100-100 |
| `[A163-181]` | 固定A链163-181残基（motif）|
| `[10-40/A163-181/10-40]` | 10-40新设计 + 固定motif + 10-40新设计 |
| `[A1-150/0 70-100]` | A链1-150 + 链断开(0) + 70-100新binder |
| `ppi.hotspot_res=[A59,A83,A91]` | 指定binder必须接触的热点残基（建议3-6个）|

## 性能（RTX 4090 45GB）
- unconditional 100-200残基：约0.40分钟/设计
- 建议 num_designs=10 批量生成，筛选最优

## 与ProteinMPNN配合使用
RFdiffusion只生成骨架（无序列），需配合ProteinMPNN设计序列：
```
RFdiffusion → backbone.pdb → ProteinMPNN → sequences.fa → AlphaFold3验证
```

## PDB 预处理（自动）
Router 在 binder_design 模式下自动处理靶蛋白 PDB：
1. **去除 HETATM**：水/配体/糖基化修饰会干扰 ContigMap
2. **顺序重编号**：消除残基间隙（gap），例如 PDB 2A91 原始编号 102→107 跳跃，重编号后 102→103 连续
3. **Hotspot 映射**：hotspot 残基号自动从原始编号转换到新编号
4. 输出文件：`{input}_renum.pdb`

### Hotspot 格式
- 用户/Qwen 可输入：`S310,F311`（氨基酸名+编号）或 `310,311`（纯编号）
- Router 自动归一化为 RFdiffusion 格式：`[A306,A307]`（chain ID + 重编号后的编号）

### 为什么需要重编号
RFdiffusion ContigMap 遍历 `A1-{max_resid}`，逐个检查残基是否存在于 PDB。
如果 PDB 有 gap（如晶体学缺失残基），ContigMap 会抛出：
`AssertionError: ('A', 103) is not in pdb file!`

## 错误排查
| 错误 | 原因 | 解决 |
|------|------|------|
| python: command not found | 容器内只有python3 | 用 python3 |
| FileNotFoundError: 1qys.pdb | model_runners.py路径bug | 执行上方bug修复命令 |
| AttributeError: 'int' has no strip | contig写成[50]整数 | 改成[50-50]或[100-200] |
| AssertionError: ('A', N) not in pdb | PDB有残基gap | 自动重编号已修复；如仍出现检查HETATM |
| 模型加载失败 | 挂载路径错误 | 检查docker-compose.yml挂载 |
| CUDA OOM | 显存不足 | 减少num_designs或缩短contig长度 |

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## ⚠️ 注意事项（自动同步自 CLAUDE.md）

- 容器内 NVIDIA_VISIBLE_DEVICES=1，永远用 device=0 / gpu_id=0（不要用1）
- **timeout 必须 7200s**（不是 3600s）
- **num_designs 用 10**（不要 20，太慢）
- **hotspot 必须空间聚集**（<= 15Å centroid），分散的会极慢

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->
