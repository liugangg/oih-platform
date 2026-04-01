# Matplotlib Figures -- 论文图表生成操作规程

## 环境
- Python: `/data/oih/miniconda/bin/python`
- 无头模式 (必须在 import pyplot 前设置):
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

## Nature Communications 排版规范
- 单栏宽: 89mm (3.5in), 双栏宽: 183mm (7.2in)
- 字号: 标签 7-8pt, 标题 8-9pt, 注释 5.5-6.5pt
- DPI: 300 (栅格), 同时输出 PDF (矢量)
- 字体: Arial > Liberation Sans > DejaVu Sans

### 标准 rcParams
```python
plt.rcParams.update({
    'font.family': 'Arial',     # 或 Liberation Sans
    'font.size': 8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'savefig.dpi': 300,
})
```

## Okabe-Ito 色盲友好调色板 (全项目统一)
```python
C = {
    'HER2':    '#0072B2',   # 蓝
    'Nectin4': '#E69F00',   # 橙
    'EGFR':    '#009E73',   # 绿
    'CD36':    '#D55E00',   # 红橙
    'TROP2':   '#999999',   # 灰
    'DT3':     '#CC79A7',   # 粉紫 (DiscoTope3)
    'PeSTo':   '#56B4E9',   # 天蓝
    'Agent':   '#F0E442',   # 黄
    'thresh':  '#D55E00',   # 阈值虚线
    'text2':   '#666666',   # 辅助文字
}
```

## 保存函数 (同时输出 PNG + PDF)
```python
OUT = '/data/oih/outputs/paper_figures'
os.makedirs(OUT, exist_ok=True)

def savefig(fig, name):
    for fmt in ['pdf', 'png']:
        fig.savefig(f'{OUT}/{name}.{fmt}', dpi=300, bbox_inches='tight')
    plt.close(fig)
```

## 数据规范
- **只用 post-fix 验证数据** (修复 MPNN chains_to_design bug 后的数据)
- ipTM 阈值: 0.6 (虚线), ipSAE 阈值: 0.15 (虚线)
- 成功标准: ipSAE >= 0.15
- 阈值线统一用 `C['thresh']` 颜色, `ls='--'`, `lw=0.8`, `alpha=0.5-0.7`

## 常用图表类型

### 1. Strip/Swarm Plot (ipTM/ipSAE 对比)
```python
fig, ax = plt.subplots(figsize=(70/25.4, 55/25.4))
jitter = 0.08
np.random.seed(42)
x = np.ones(n) * group_idx + np.random.uniform(-jitter, jitter, n)
ax.scatter(x, values, c=color, s=50, zorder=3, edgecolors='white', linewidth=0.5)
# 均值线
ax.plot([group_idx-.15, group_idx+.15], [mean]*2, c=color, lw=2)
```

### 2. Scatter Plot (ipTM vs ipSAE)
```python
fig, ax = plt.subplots(figsize=(100/25.4, 85/25.4))
ax.scatter(iptm, ipsae, c=color, marker='o', s=45,
           edgecolors='white', linewidth=0.4, label=target, zorder=3)
ax.axhline(0.15, color=C['thresh'], ls='--', lw=0.8, alpha=0.5)
ax.axvline(0.5, color=C['thresh'], ls='--', lw=0.8, alpha=0.5)
# 象限着色
ax.axhspan(0.15, ymax, xmin=0.5/xlim, alpha=0.04, color='#009E73')
```

### 3. Horizontal Bar Chart
```python
fig, ax = plt.subplots(figsize=(90/25.4, 50/25.4))
y = np.arange(len(names))
ax.barh(y, values, color=colors, edgecolor='white', height=0.65)
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

### 4. Stacked Area Plot (per-residue profiles)
```python
fig, ax = plt.subplots(figsize=(183/25.4, 50/25.4))
ax.fill_between(residues, 0, profile1, alpha=0.4, color=C['PeSTo'], label='PeSTo')
ax.fill_between(residues, 0, profile2, alpha=0.4, color=C['DT3'], label='DT3')
```

### 5. Heatmap
```python
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('custom', ['white', '#0072B2'])
im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
fig.colorbar(im, ax=ax, shrink=0.8)
```

### 6. Funnel Chart (pipeline filtering)
```python
stages = ['RFdiffusion', 'MPNN', 'ESM2 PPL<15', 'AF3 ipTM>0.6', 'ipSAE>0.15']
counts = [100, 80, 60, 20, 5]
for i, (stage, count) in enumerate(zip(stages, counts)):
    width = count / max(counts)
    ax.barh(i, width, height=0.6, left=(1-width)/2, color=gradient_color)
    ax.text(0.5, i, f'{stage}: {count}', ha='center', va='center', fontsize=7)
```

## 标准样式规则

### 子面板标记
```python
fig.text(0.01, 0.98, 'a', fontsize=12, fontweight='bold', va='top')
```

### 轴清理 (所有图必须)
```python
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

### 尺寸约定 (mm -> inches)
```python
figsize=(width_mm/25.4, height_mm/25.4)
# 单栏: 89mm, 1.5栏: 120mm, 双栏: 183mm
```

### 注释箭头
```python
ax.annotate('label', xy=(x, y), xytext=(tx, ty),
            fontsize=5.5, arrowprops=dict(arrowstyle='->', lw=0.5, color=C['text2']),
            color=C['text2'])
```

## 字体注意事项
- 避免 Unicode 符号 (mu, alpha 等)，用 ASCII: `mu=`, `>=`, `<=`
- 如果必须用希腊字母，用 matplotlib mathtext: `r'$\mu$'`
- 中文注释不要放在图里（期刊不接受），放 figure legend

## 输出
- 所有图保存到 `/data/oih/outputs/paper_figures/`
- 命名: `fig{N}{letter}_{description}.png` 和 `.pdf`
- 同时保存 PNG (栅格) 和 PDF (矢量)

## 完整模板
```python
#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = '/data/oih/outputs/paper_figures'
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'savefig.dpi': 300,
})

C = {
    'HER2': '#0072B2', 'Nectin4': '#E69F00', 'EGFR': '#009E73',
    'CD36': '#D55E00', 'TROP2': '#999999',
    'DT3': '#CC79A7', 'PeSTo': '#56B4E9', 'Agent': '#F0E442',
    'thresh': '#D55E00', 'text2': '#666666',
}

def savefig(fig, name):
    for fmt in ['pdf', 'png']:
        fig.savefig(f'{OUT}/{name}.{fmt}', dpi=300, bbox_inches='tight')
    plt.close(fig)

# --- Your figure ---
fig, ax = plt.subplots(figsize=(89/25.4, 60/25.4))
# ... plot code ...
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.text(0.01, 0.98, 'a', fontsize=12, fontweight='bold', va='top')
savefig(fig, 'figX_description')
```
