# ESM工作流文档

## 基本信息
- 容器：oih-esm
- 模型：ESM2-650M (esm2_t33_650M_UR50D)
- 模型路径：/root/.cache/torch/hub/checkpoints/
- GPU：NVIDIA_VISIBLE_DEVICES=1 → 容器内cuda:0

## 主要功能
- ESM2：蛋白质序列嵌入提取（结构/功能预测下游任务）
- ESM-1v：零样本变体效应预测
- ESM-IF1：逆折叠（给定结构设计序列）

## 嵌入提取（Python API）
```python
import torch
import esm

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval().to('cuda:0')

data = [("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQL...")]
batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_tokens = batch_tokens.to('cuda:0')

with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=False)

embeddings = results['representations'][33]
# shape: [batch, seq_len, 1280]
```

## 输出说明
- repr_layers=[33]：最后一层（第33层），1280维
- embeddings[:,0,:]：[CLS] token，代表整条序列
- embeddings[:,1:-1,:]：每个氨基酸残基的嵌入

## 命令行批量提取（FASTA）
```bash
docker exec oih-esm bash -c "
python3 /app/esm/scripts/extract.py \
  esm2_t33_650M_UR50D \
  /data/oih/inputs/sequences.fasta \
  /data/oih/outputs/esm_embeddings/ \
  --repr_layers 33 \
  --include mean per_tok"
```

## 注意事项
- 模型已缓存，无需重新下载
- 长序列（>1000aa）显存占用较大，建议batch_size=1
- 显存预估：650M模型约6GB（GPU1 45GB足够）
- ESMFold需python<=3.9，当前环境不支持，用AlphaFold3代替结构预测

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## ⚠️ 注意事项（自动同步自 CLAUDE.md）

- 容器内 NVIDIA_VISIBLE_DEVICES=1，永远用 device=0 / gpu_id=0（不要用1）

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->
