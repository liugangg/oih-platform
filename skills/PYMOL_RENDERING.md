# PyMOL Rendering -- 结构可视化操作规程

## 环境
- 安装: `pip install pymol-open-source` (已装在 `/data/oih/miniconda/`)
- 命令行无头渲染: `/data/oih/miniconda/bin/pymol -cq script.py`
- 不需要 X11/display，`-cq` = command-line + quiet

## 标准发布设置 (Nature Communications)
```python
from pymol import cmd

# 白色背景，无阴影
cmd.set("cartoon_fancy_helices", 1)
cmd.set("ray_opaque_background", 1)
cmd.bg_color("white")
cmd.set("ray_shadows", 0)
cmd.set("antialias", 2)

# 渲染
cmd.orient("object_name")
cmd.ray(2400, 1800)
cmd.png("/data/oih/outputs/paper_figures/output.png", dpi=300)
cmd.quit()
```

## 常用操作

### 加载与清理
```python
cmd.load("/data/oih/outputs/fetch_pdb/XXXX.pdb", "protein")
cmd.remove("resn HOH")       # 移除水分子
cmd.remove("hetatm")         # 移除配体/离子
cmd.remove("not chain A")    # 只保留目标链
```

### 显示模式
```python
cmd.show("cartoon", "protein")       # cartoon 主体
cmd.show("sticks", "selection")      # 侧链棍棒
cmd.show("spheres", "selection")     # 球体高亮
cmd.show("surface", "protein")       # 表面
cmd.hide("everything", "selection")  # 隐藏
```

### 颜色方案
- 基底色: `gray80` (浅灰) 或 `gray70` (深灰)
- 高亮色: `marine` (蓝), `red`, `yellow`, `cyan`, `orange`, `lightpink`
- 错误标记: `red` (wrong), `marine` (correct)

```python
cmd.color("gray80", "protein")
cmd.color("marine", "domain_selection")
cmd.color("yellow", "hotspot_selection")
```

### 选择残基
```python
# 按残基编号
cmd.select("hotspots", "resi 558+560+571+572+573 and chain C")
# 按残基范围
cmd.select("domain4", "resi 510-630 and chain C")
# 按残基类型
cmd.select("lysines", "resn LYS and resi 188")
```

### 对齐叠加
```python
cmd.align("mobile_object and chain X", "target_object and chain Y")
```

### 透明幽灵叠加
```python
cmd.set("cartoon_transparency", 0.65, "ghost_object")
cmd.color("lightpink", "ghost_object")
```

### 球体热点可视化
```python
cmd.show("spheres", "hotspots")
cmd.set("sphere_scale", 0.4, "hotspots")  # 0.4-0.6 适合热点
```

### 标签
```python
cmd.set("label_size", 18)
cmd.set("label_color", "red", "selection")
```

## 实战模板

### 模板 1: 单蛋白域高亮 (EGFR fig2c 模式)
用途: 展示蛋白不同域，标记正确/错误热点
```python
from pymol import cmd

cmd.load("/data/oih/outputs/fetch_pdb/1YY9.pdb", "egfr")
cmd.remove("resn HOH")
cmd.remove("hetatm")
cmd.remove("not chain A")

cmd.show("cartoon", "egfr")
cmd.color("gray80", "egfr")

# 错误热点 - 红色
cmd.select("wrong", "resi 86-89 and chain A")
cmd.color("red", "wrong")
cmd.show("sticks", "wrong")

# 正确热点 - 蓝色
cmd.select("correct", "resi 353+355+382+384+408+410 and chain A")
cmd.color("marine", "correct")
cmd.show("sticks", "correct")

cmd.set("cartoon_fancy_helices", 1)
cmd.set("ray_opaque_background", 1)
cmd.bg_color("white")
cmd.set("ray_shadows", 0)
cmd.set("antialias", 2)
cmd.orient("egfr")
cmd.ray(2400, 1800)
cmd.png("/data/oih/outputs/paper_figures/output.png", dpi=300)
cmd.quit()
```

### 模板 2: 域高亮 + 热点球体 (HER2 fig2d 模式)
用途: 展示蛋白域 + 球体标记药物结合热点
```python
from pymol import cmd

cmd.load("/data/oih/outputs/fetch_pdb/1N8Z.pdb", "her2")
cmd.remove("resn HOH")
cmd.remove("not chain C")

cmd.show("cartoon", "her2")
cmd.color("gray70", "her2")

# Domain IV 高亮
cmd.select("domain4", "resi 510-630 and chain C")
cmd.color("marine", "domain4")

# Hotspot 球体
cmd.select("hotspots", "resi 558+560+571+572+573 and chain C")
cmd.color("yellow", "hotspots")
cmd.show("sticks", "hotspots")
cmd.show("spheres", "hotspots")
cmd.set("sphere_scale", 0.4, "hotspots")

cmd.set("cartoon_fancy_helices", 1)
cmd.set("ray_opaque_background", 1)
cmd.bg_color("white")
cmd.set("ray_shadows", 0)
cmd.set("antialias", 2)
cmd.orient("her2")
cmd.ray(2400, 1800)
cmd.png("/data/oih/outputs/paper_figures/output.png", dpi=300)
cmd.quit()
```

### 模板 3: 复合物叠加对比 (fig5d 模式)
用途: 设计的 binder 与已知抗体叠加比较
```python
import glob
from pymol import cmd

# 加载设计复合物
cif = glob.glob("/data/oih/outputs/job_name/alphafold3/job_name/*_model.cif")
cmd.load(cif[0], "designed")

# 加载参考抗体
cmd.load("/data/oih/outputs/fetch_pdb/1N8Z.pdb", "reference")

# 设计复合物着色
cmd.show("cartoon", "designed")
cmd.select("des_binder", "designed and chain A")
cmd.select("des_antigen", "designed and chain B")
cmd.color("cyan", "des_binder")
cmd.color("orange", "des_antigen")

# 参考抗体: 半透明粉色幽灵
cmd.show("cartoon", "reference and chain A")
cmd.show("cartoon", "reference and chain B")
cmd.color("lightpink", "reference")
cmd.set("cartoon_transparency", 0.65, "reference")

# 隐藏参考中不需要的链
cmd.hide("everything", "reference and chain C")
cmd.hide("everything", "reference and chain D")

# 对齐: 参考抗原链对齐到设计抗原链
cmd.align("reference and chain C", "des_antigen")

# 关键残基标记 (如偶联位点 K188)
cmd.select("k188", "des_binder and resn LYS and resi 188")
cmd.show("spheres", "k188")
cmd.color("red", "k188")
cmd.set("sphere_scale", 0.6, "k188")

cmd.set("cartoon_fancy_helices", 1)
cmd.set("ray_opaque_background", 1)
cmd.bg_color("white")
cmd.set("ray_shadows", 0)
cmd.set("antialias", 2)
cmd.orient("designed")
cmd.ray(2400, 1800)
cmd.png("/data/oih/outputs/paper_figures/output.png", dpi=300)
cmd.quit()
```

## 输出
- 所有图片保存到 `/data/oih/outputs/paper_figures/`
- 命名约定: `fig{N}{letter}_{description}.png`

## Tips
- 加载后**先**移除 HOH 和 HETATM，再做选择
- 渲染前**必须** `cmd.orient()` 调整视角
- `sphere_scale` 0.4-0.6 适合热点可视化，太大会遮挡 cartoon
- `cartoon_transparency` 0.5-0.7 适合幽灵叠加，太低看不清后面
- 多链 PDB 先确认链编号（用 gemmi 或 PyMOL `cmd.get_chains()`）
- CIF 文件直接 `cmd.load()` 即可，PyMOL 支持 mmCIF
- 脚本最后**必须** `cmd.quit()` 否则进程不退出
