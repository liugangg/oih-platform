# Vina-GPU 2.1 & GNINA 对接工作流（已验证）

## 工具信息
- Vina-GPU容器：oih-vina-gpu，命令：vina_gpu（AutoDockVina-GPU 2.1）
- GNINA容器：oih-gnina，命令：gnina（深度学习CNN打分）
- GPU：NVIDIA_VISIBLE_DEVICES=1 → 容器内device 0

## 何时用Vina-GPU vs GNINA vs AutoDock-GPU
| 工具 | 适用场景 |
|------|------|
| Vina-GPU 2.1 | 快速筛选，已知口袋坐标，标准对接 |
| GNINA | 通用最佳，CNN重打分，精度更高，输出SDF |
| AutoDock-GPU | 需要精确自由能，大规模虚拟筛选 |

## Vina-GPU命令
```bash
docker exec oih-vina-gpu vina_gpu \
  --receptor /data/oih/inputs/<job>/receptor.pdbqt \
  --ligand <lig.pdbqt> \
  --out /data/oih/outputs/<job>/vina_gpu/out.pdbqt \
  --center_x <x> --center_y <y> --center_z <z> \
  --size_x 25 --size_y 25 --size_z 25 \
  --num_modes 9 \
  --exhaustiveness 8 \
  --thread 8000
```
输出格式：PDBQT，解析affinity行

## GNINA命令
```bash
docker exec oih-gnina gnina \
  --receptor /data/oih/inputs/<receptor>.pdb \
  --ligand <lig.pdbqt> \
  --out /data/oih/outputs/<job>/gnina/poses.sdf \
  --center_x <x> --center_y <y> --center_z <z> \
  --size_x 25 --size_y 25 --size_z 25 \
  --num_modes 9 \
  --exhaustiveness 8 \
  --autobox_add 4 \
  --device 0
```
输出格式：SDF，解析CNNscore和affinity

## 配体准备（两者相同）
```bash
# SMILES → pdbqt（容器内obabel）
obabel -:"<SMILES>" --gen3d -O ligand.pdbqt -h
```

## ⚠️ 注意事项
- Vina-GPU：receptor需预先转换为pdbqt格式
- GNINA：receptor直接用PDB，无需转换 ✅
- GNINA输出SDF比PDBQT更易解析和可视化
- smart_dock路由：有坐标→Vina-GPU，无坐标→DiffDock盲对接

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## ⚠️ 注意事项（自动同步自 CLAUDE.md）

- 容器内 NVIDIA_VISIBLE_DEVICES=1，永远用 device=0 / gpu_id=0（不要用1）

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->
