# OIH 平台自我诊断与修复手册

## 核心原则
当任何计算任务失败时，不等待用户指令，自主执行：
诊断 → 修复参数 → 重试 → 验证 → 报告结果

## AF3 失败诊断树

### ipTM=None（超时或崩溃）
1. 检查输出目录是否有 CIF 文件：
   find /data/oih/outputs -name "*model.cif" | grep -i af3 | head -5
2. 有 CIF → 超时误判，读取真实分数：
   cat <output_dir>/*ranking_scores.csv
   → 用真实 ipTM 继续 pipeline
3. 无 CIF → 真正失败，检查原因：
   docker logs oih-alphafold3 --tail 30
   nvidia-smi --query-gpu=memory.used,memory.free --format=csv
   → OOM：等待 GPU 空闲后重试
   → 其他错误：查 /tmp/fastapi.log

### ipTM < 0.5（设计质量差）
原因：hotspot 残基选错或设计数量不足
修复步骤：
1. 换 hotspot 到真实结合界面：
   - HER2 Domain II (pertuzumab表位): T144, R150, S175, R177
   - HER2 Domain IV (trastuzumab表位): S310, T311, N344, R375
   - TROP2 结合位点: K65, R87, D110
2. 增加 num_designs: 10 → 50
3. 降低验证阈值: 0.6 → 0.5（先跑通再优化）

### ipTM 0.5-0.75（低置信度通过）
- 继续 pipeline，标记为 low_confidence
- 在报告中注明需要湿实验验证

## VRAM 问题诊断

### 快速检查
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

### 路由规则
- GPU0 (23GB): Qwen3-14B 常驻，剩余空间不稳定
- GPU1 (46GB): 所有生物计算工具，容器内 device=0
- AF3/BindCraft 需要 ≥20GB → 只能用 GPU1
- 如果 GPU1 不足 → 等待，不降级

## RFdiffusion 设计质量优化

### 自动参数调整策略
第一轮：num_designs=10，快速验证 hotspot
→ 全部失败：换 hotspot 重试
→ 部分通过 (ipTM>0.5)：增加 num_designs=50
→ 无通过：降低 AF3 阈值到 0.5

### HER2 靶点最优 hotspot
- 避开 trastuzumab 表位（竞争现有药物）
- 推荐新表位：Domain II K505, E558, D561
  或 Domain III: H473, N476

## GROMACS 常见问题

### nvt.gro 不存在
→ NVT 步骤静默失败，检查：
   cat /data/oih/outputs/<job>/nvt.log | tail -20

### tc-grps 错误
→ 动态检测蛋白+配体组名，不要硬编码

## RFdiffusion ContigMap 崩溃

### AssertionError: ('A', N) is not in pdb file!
原因：PDB 有残基间隙（晶体学缺失残基），ContigMap 遍历连续范围时找不到缺失的残基。
修复：Router 自动重编号 PDB（去 HETATM + 顺序编号），已在 `_renumber_pdb_for_rfdiffusion()` 实现。
如仍出现：检查是否有非标准残基、PDB 文件是否被截断。

## Pipeline 自动重试策略

失败后自动重试规则：
1. 超时类错误 → 直接重试（最多3次）
2. OOM 错误 → 等待60s再重试
3. 参数错误 → 调整参数后重试
4. 文件不存在 → 检查上一步是否完成

## 蒸馏训练数据

位置：`data/distillation/`
格式：JSONL，每行一个案例（task_id, error_msg, tool, fix_applied, outcome）
分类：gpu / container / tool / pipeline / proxy / dashboard / abandoned / reference
目标：积累 100 条后做 LoRA 微调 Qwen3-14B
收集脚本：`scripts/collect_distillation_data.py`（自动从任务历史提取失败模式）
