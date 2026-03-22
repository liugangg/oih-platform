# AutoDock-GPU 完整前处理流程（已验证）

## 工具分布
- obabel, autogrid4: 宿主机 /usr/bin/
- autodock_gpu_128wi: 容器 oih-autodock-gpu /usr/local/bin/
- prody: 容器 oih-autodock-gpu 内 Python

## 完整流程

### 步骤1: prody 清理 PDB（容器内）
```python
from prody import parsePDB, writePDB
struct = parsePDB('input.pdb', altloc='first')
protein = struct.select('protein and chain A B')  # 只保留蛋白链
writePDB('protein_clean.pdb', protein)
```

### 步骤2: obabel 生成带电荷 pdbqt（宿主机）
```bash
obabel protein_clean.pdb -O receptor.pdbqt -xr --partialcharge gasteiger
```
注意：-xr 表示rigid receptor，必须加 --partialcharge gasteiger 否则电荷为0导致autogrid4报错

### 步骤3: 获取口袋坐标（从fpocket结果自动获取）
```python
# 从fpocket输出解析 box_center 和 box_size
# box_center = pocket['x_cent'], pocket['y_cent'], pocket['z_cent']
# box_size 默认 25 25 25（可根据口袋体积调整）
```

### 步骤4: 生成 GPF 文件（宿主机）
```bash
cat > receptor.gpf << GPF
npts 60 60 60
gridfld receptor.maps.fld
spacing 0.375
receptor_types A C N NA OA S
ligand_types A C HD N NA OA SA
receptor receptor.pdbqt
gridcenter {cx} {cy} {cz}
smooth 0.5
map receptor.A.map
map receptor.C.map
map receptor.HD.map
map receptor.N.map
map receptor.NA.map
map receptor.OA.map
map receptor.SA.map
elecmap receptor.e.map
dsolvmap receptor.d.map
dielectric -0.1465
GPF
```
注意：receptor_types 必须与 pdbqt 中实际原子类型一致

### 步骤5: autogrid4 生成 grid（宿主机）
```bash
cd /workdir && autogrid4 -p receptor.gpf -l receptor.glg
```
生成文件：receptor.maps.fld, receptor.*.map

### 步骤6: autodock_gpu_128wi 对接（容器内）
```bash
autodock_gpu_128wi \
  --ffile receptor.maps.fld \
  --lfile ligand.pdbqt \
  --nrun 20 \
  --devnum 1 \
  --resnam result
```

### 配体准备（从SMILES）
```bash
# 宿主机 obabel 从 SMILES 生成 3D pdbqt
obabel -:"SMILES_STRING" -opdbqt --gen3d -O ligand.pdbqt --partialcharge gasteiger
```

## 非标准残基处理策略
- 检测 HETATM：grep HETATM input.pdb | awk '{print $4}' | sort -u
- NAG/糖基化等：prody select 时用 'protein' 自动排除
- 如果 obabel 报 kekulize 警告：通常不影响结果，可忽略

## 错误处理
- autogrid4 "atom types不匹配"：检查 pdbqt 实际原子类型重新写 gpf
- autogrid4 "no partial charges"：重新用 --partialcharge gasteiger 生成 pdbqt
- meeko HasQuery报错：meeko版本与rdkit不兼容，改用obabel替代

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## ⚠️ 注意事项（自动同步自 CLAUDE.md）

- 容器内 NVIDIA_VISIBLE_DEVICES=1，永远用 device=0 / gpu_id=0（不要用1）

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->
