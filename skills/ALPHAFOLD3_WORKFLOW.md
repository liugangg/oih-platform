# AlphaFold3工作流文档

## 基本信息
- 容器：oih-alphafold3
- 软件路径：/app/alphafold/run_alphafold.py
- 模型路径：/data/alphafold3_models/（af3.bin）
- 数据库路径：/data/alphafold3_db/
- 输入/输出：/data/af3/input/ 和 /data/af3/output/
- GPU：NVIDIA_VISIBLE_DEVICES=1 → 容器内cuda:0
- 推理时间：~82秒（100aa单链）

## 输入JSON格式
```json
{
  "name": "job_name",
  "dialect": "alphafold3",
  "version": 1,
  "sequences": [
    {
      "protein": {
        "id": ["A"],
        "sequence": "SEQUENCE..."
      }
    }
  ],
  "modelSeeds": [1]
}
```

## 支持的输入类型
- protein：蛋白质序列
- rna/dna：核酸序列
- ligand：小分子（SMILES）
- 多链复合物：多个sequences条目

## 运行命令
```bash
docker exec oih-alphafold3 bash -c "
python3 /app/alphafold/run_alphafold.py \
  --json_path=/data/af3/input/<input>.json \
  --model_dir=/data/alphafold3_models \
  --db_dir=/data/alphafold3_db \
  --output_dir=/data/af3/output/<job_name> \
  --flash_attention_implementation=triton"
```

## 输出文件
- *_model.cif：最佳结构（mmCIF格式）
- *_confidences.json：每残基pLDDT置信度
- *_summary_confidences.json：整体指标（ipTM、pTM等）
- *_ranking_scores.csv：多seed排名
- seed-X_sample-Y/：各采样结构

## 注意事项
- data pipeline（MSA搜索）CPU密集，128核全速运行
- --run_data_pipeline=false 跳过MSA（需已有MSA缓存）
- --run_inference=false 只跑MSA不推理
- 多链复合物预测ipTM>0.6为可信结合
- ⚠️ 不挂载宿主机cuda lib64（同BindCraft，JAX自带CUDA）

## 任务调度规则

### 永不降级（_NO_DEGRADED_TOOLS）
AF3 和 BindCraft 永远不进 DEGRADED 队列（CPU fallback 会 OOM crash）。
VRAM 不足时每 60s 重试检查 GPU1，等到空闲后进 GPU 队列。

### 超时处理（_wait_for_af3_task）
Pipeline 中 AF3 使用无限等待：每 30s poll 任务状态。
仅以下情况判定失败：
- OOM（exit code 1 + memory 关键词）
- 容器 exit 1
- 任务被 cancel
- 连续 10 次返回相同错误

**超时误判诊断**：如果 AF3 "超时"但可能已完成，检查输出目录：
```bash
find /data/oih/outputs -name "*model.cif" | grep af3
cat <output_dir>/*ranking_scores.csv   # 读取真实 ipTM
```
有 CIF 文件 → 超时是误判，用真实 ipTM 继续 pipeline。

### Pipeline 中 AF3 调用间隔
多个 AF3 任务连续提交时，每个之间间隔 5 秒，避免 GPU OOM。

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## ⚠️ 注意事项（自动同步自 CLAUDE.md）

- 容器内 NVIDIA_VISIBLE_DEVICES=1，永远用 device=0 / gpu_id=0（不要用1）
- **根因**：v3 任务 5 个 AF3 设计全部失败。rank1 超时 1800s（但实际已跑完，ipTM=0.48），rank2-5 被路由到 DEGRADED 队列 OOM exit 1
- **修复 1**：新增 `_wait_for_af3_task()` 无限等待函数，每 30s poll，仅在 OOM/exit1/cancelled/连续10次同错 时判定失败
- 不再降级到 DEGRADED 导致 OOM crash
- **原因**：CIF→PDB 转换失败时 `af3_pdb=None` → `"No PDB for FreeSASA"` 错误

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->
