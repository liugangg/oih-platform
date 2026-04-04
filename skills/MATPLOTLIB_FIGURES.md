# Matplotlib Figures -- Paper Figure Generation Protocol

## Environment
- Python: `/data/oih/miniconda/bin/python`
- Headless mode (must set before importing pyplot):
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

## Nature Communications Typesetting Standards
- Single column width: 89mm (3.5in), Double column width: 183mm (7.2in)
- Font size: Labels 7-8pt, Titles 8-9pt, Annotations 5.5-6.5pt
- DPI: 300 (raster), also output PDF (vector)
- Font: Arial > Liberation Sans > DejaVu Sans

### Standard rcParams
```python
plt.rcParams.update({
    'font.family': 'Arial',     # or Liberation Sans
    'font.size': 8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'savefig.dpi': 300,
})
```

## Okabe-Ito Colorblind-Friendly Palette (Project-wide)
```python
C = {
    'HER2':    '#0072B2',   # blue
    'Nectin4': '#E69F00',   # orange
    'EGFR':    '#009E73',   # green
    'CD36':    '#D55E00',   # red-orange
    'TROP2':   '#999999',   # gray
    'DT3':     '#CC79A7',   # pink-purple (DiscoTope3)
    'PeSTo':   '#56B4E9',   # sky blue
    'Agent':   '#F0E442',   # yellow
    'thresh':  '#D55E00',   # threshold dashed line
    'text2':   '#666666',   # secondary text
}
```

## Save Function (Output both PNG + PDF)
```python
OUT = '/data/oih/outputs/paper_figures'
os.makedirs(OUT, exist_ok=True)

def savefig(fig, name):
    for fmt in ['pdf', 'png']:
        fig.savefig(f'{OUT}/{name}.{fmt}', dpi=300, bbox_inches='tight')
    plt.close(fig)
```

## Data Standards
- **Only use post-fix validation data** (data after fixing MPNN chains_to_design bug)
- ipTM threshold: 0.6 (dashed line), ipSAE threshold: 0.15 (dashed line)
- Success criterion: ipSAE >= 0.15
- Threshold lines: use `C['thresh']` color, `ls='--'`, `lw=0.8`, `alpha=0.5-0.7`

## Common Chart Types

### 1. Strip/Swarm Plot (ipTM/ipSAE comparison)
```python
fig, ax = plt.subplots(figsize=(70/25.4, 55/25.4))
jitter = 0.08
np.random.seed(42)
x = np.ones(n) * group_idx + np.random.uniform(-jitter, jitter, n)
ax.scatter(x, values, c=color, s=50, zorder=3, edgecolors='white', linewidth=0.5)
# mean line
ax.plot([group_idx-.15, group_idx+.15], [mean]*2, c=color, lw=2)
```

### 2. Scatter Plot (ipTM vs ipSAE)
```python
fig, ax = plt.subplots(figsize=(100/25.4, 85/25.4))
ax.scatter(iptm, ipsae, c=color, marker='o', s=45,
           edgecolors='white', linewidth=0.4, label=target, zorder=3)
ax.axhline(0.15, color=C['thresh'], ls='--', lw=0.8, alpha=0.5)
ax.axvline(0.5, color=C['thresh'], ls='--', lw=0.8, alpha=0.5)
# quadrant shading
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

## Standard Style Rules

### Sub-panel Labels
```python
fig.text(0.01, 0.98, 'a', fontsize=12, fontweight='bold', va='top')
```

### Axis Cleanup (Required for all plots)
```python
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

### Size Convention (mm -> inches)
```python
figsize=(width_mm/25.4, height_mm/25.4)
# Single column: 89mm, 1.5 column: 120mm, Double column: 183mm
```

### Annotation Arrows
```python
ax.annotate('label', xy=(x, y), xytext=(tx, ty),
            fontsize=5.5, arrowprops=dict(arrowstyle='->', lw=0.5, color=C['text2']),
            color=C['text2'])
```

## Font Notes
- Avoid Unicode symbols (mu, alpha, etc.), use ASCII: `mu=`, `>=`, `<=`
- If Greek letters are needed, use matplotlib mathtext: `r'$\mu$'`
- Do not put non-English annotations in figures (journals don't accept), place in figure legend

## Output
- All figures saved to `/data/oih/outputs/paper_figures/`
- Naming: `fig{N}{letter}_{description}.png` and `.pdf`
- Save both PNG (raster) and PDF (vector)

## Complete Template
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
