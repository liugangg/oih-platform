# Fpocket & P2Rank 结合口袋检测工作流（已验证）

## 工具信息
- fpocket容器：oih-fpocket
- p2rank容器：oih-p2rank
- 用途：分子对接前检测蛋白质结合位点坐标

## 何时用fpocket vs p2rank
- fpocket：通用蛋白，速度快，返回drug_score + 坐标
- p2rank：AlphaFold预测结构推荐用alphafold模式，精度更高

## fpocket调用命令
```bash
docker exec oih-fpocket fpocket \
  -f /data/oih/outputs/<job>/<input>.pdb \
  -m <min_sphere_size>
```
输出目录：`/data/oih/outputs/<job>/<pdb_stem>_out/`
解析文件：`**/*_info.txt`
输出字段：pocket_id, drug_score, volume, x_centroid, y_centroid, z_centroid

## p2rank调用命令
```bash
docker exec oih-p2rank bash -c "
/app/p2rank/prank predict [-c alphafold] \
  -f /data/oih/inputs/<input>.pdb \
  -o /data/oih/outputs/<job>/p2rank
"
```
model参数：default | alphafold | conservation
输出文件：`*_predictions.csv`
输出字段：name, rank, score, probability, center_x, center_y, center_z

## ⚠️ 注意事项
- fpocket：input_pdb先从inputs/复制到outputs/再运行
- p2rank：input_pdb直接读inputs/，输出写outputs/
- CSV字段名有前导空格，解析时需strip

## 流程：口袋检测→对接
1. 调用fpocket或p2rank获取口袋列表
2. 取pockets[0]的center_x/y/z作为对接box中心
3. box_size默认25x25x25 Å
4. 传入dock_ligand的center_x/y/z参数

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## ⚠️ 注意事项（自动同步自 CLAUDE.md）

- **旧流程**（11步）：P2Rank top pocket → 直接取 top 6 残基 → DiffDock 盲对接交叉验证 → RFdiffusion
- 10. AF3 验证 — ipTM ≥ 0.6

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->
