# BindCraft工作流文档

## 基本信息
- 容器：oih-bindcraft
- 功能：AF2反向传播+MPNN+PyRosetta从头设计蛋白结合剂
- GPU：NVIDIA_VISIBLE_DEVICES=1 → 容器内cuda:0
- 显存需求：~16GB
- ⚠️ 不挂载宿主机cuda lib64（JAX自带CUDA，挂载会冲突）

## target配置文件（JSON）
```json
{
    "design_path": "/data/oih/outputs/<task>/",
    "binder_name": "<名称>",
    "starting_pdb": "/data/oih/inputs/<target>.pdb",
    "chains": "A",
    "target_hotspot_residues": null,
    "lengths": [50, 80],
    "number_of_final_designs": 1
}
```

## 运行命令
```bash
docker exec oih-bindcraft bash -c "
cd /app/BindCraft &&
python3 -u bindcraft.py \
  --settings '/data/oih/inputs/bindcraft_target.json' \
  --filters './settings_filters/default_filters.json' \
  --advanced './settings_advanced/default_4stage_multimer.json'
"
```

## 注意事项
- hotspot_residues设null让AF2自动选择结合位点
- 建议至少100个final designs用于实验筛选
- 每个trajectory约需数分钟，难靶标可能需要数千次
- 输出：PDB结构+CSV统计（ipTM、pLDDT等）
- ipTM是结合预测的二元指标，不直接反映亲和力

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## ⚠️ 注意事项（自动同步自 CLAUDE.md）

- 容器内 NVIDIA_VISIBLE_DEVICES=1，永远用 device=0 / gpu_id=0（不要用1）

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->
