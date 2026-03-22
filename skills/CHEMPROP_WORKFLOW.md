# Chemprop工作流文档

## 基本信息
- 容器：oih-chemprop
- Python：3.11（容器内 /usr/bin/python3.11，symlink /usr/bin/python3 → python3.11）
- 版本：chemprop 2.2.2
- **推理用CPU**（--accelerator cpu），GPU加速主要用于训练大数据集
- 队列：CPU queue（task_manager 中 chemprop/chemprop_predict 归入 _CPU_TOOLS）

## CLI命令
chemprop {train, predict, convert, fingerprint, hpopt}

## 训练（GPU加速，大数据集推荐）
```bash
docker exec oih-chemprop bash -c "
chemprop train \
  -i /data/oih/inputs/train.csv \
  --smiles-columns smiles \
  --target-columns <目标列名> \
  --accelerator gpu \
  --devices 1 \
  --epochs 50 \
  --save-dir /data/oih/outputs/<task>/"
```

## 预测（CPU模式，避免GPU OOM）
```bash
docker exec oih-chemprop bash -c "
chemprop predict \
  -i /data/oih/inputs/test.csv \
  --model-paths /data/oih/outputs/<task>/model_0/best.pt \
  --accelerator cpu \
  -o /data/oih/outputs/<task>/predictions.csv"
```

## 注意事项
- 训练时 --devices 1 表示"使用1个GPU"（不是device index）
- 预测时用 --accelerator cpu，小分子推理CPU足够（3分子<5秒）
- 容器内 NVIDIA_VISIBLE_DEVICES=1，宿主机GPU1映射为容器内GPU0
- 模型保存：<save-dir>/model_0/best.pt
- 建议加 --num-workers 8 提升数据加载速度
- 容器内测试模型：/opt/chemprop/tests/data/example_model_v2_regression_mol.pt

## ADC ADMET 评估用法

chemprop predict 可用于评估 ADC payload/linker-payload 偶联物的 ADMET 性质。

**上游**：rdkit_conjugate 输出的 `adc_smiles` 字段（dot-disconnected 或 covalent SMILES 均可）

**API调用**：
```bash
curl -s --noproxy '*' -X POST http://localhost:8080/api/v1/ml/chemprop/predict \
  -H "Content-Type: application/json" \
  -d '{
    "job_name": "adc_admet",
    "smiles": ["<adc_smiles_from_rdkit_conjugate>"],
    "model_path": "/path/to/admet_model/best.pt",
    "task": "predict"
  }'
```

**输出**：每个分子的预测值（毒性/溶解性/渗透性等，取决于训练模型）

**典型 ADC 工作流**：
1. `fetch_molecule` → 获取 payload SMILES (如 MMAE)
2. `linker_select` → 选择 linker (如 MC-VC-PABC)
3. `rdkit_conjugate` → 生成 linker-payload 偶联物 adc_smiles
4. `chemprop_predict` → 用 adc_smiles 做 ADMET 预测

## 典型应用
- 溶解度/logS预测
- ADMET性质预测（吸收/分布/代谢/排泄/毒性）
- 生物活性预测（IC50、Ki等）
- 抗生素活性筛选（参考Cell 2020 Halicin）
- ADC payload 毒性/选择性评估

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## ⚠️ 注意事项（自动同步自 CLAUDE.md）

- 1. **Dockerfile 必须有**：`RUN ln -sf /usr/bin/python3.11 /usr/bin/python3`（容器内 python3 默认指向 3.10，但所有包装在 3.11 路径下）
- 2. **predict 调用必须加**：`--accelerator cpu`（小批量走 GPU 会 OOM），归属 `_CPU_TOOLS` 队列
- 3. **`--devices 1` = "使用 1 个 GPU 设备"**，不是 device index=1。train 用 `--accelerator gpu --devices 1`，predict 用 `--accelerator cpu`
- **根因**：容器内 `python3` symlink 指向 3.10，但所有包（torch/numpy/chemprop）装在 python3.11 路径下 → `No module named 'numpy'`
- **修复**：`docker exec oih-chemprop bash -c "rm /usr/bin/python3 && ln -s /usr/bin/python3.11 /usr/bin/python3"`（容器重建会丢失，需写入 Dockerfile）
- **router 修复**：`routers/ml_tools.py` 加 `--accelerator cpu`，避免 GPU OOM；`task_manager.py` 把 `chemprop_predict` 加入 `_CPU_TOOLS`
- **验证**：3 分子预测 completed，CPU queue，5 秒完成

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->
