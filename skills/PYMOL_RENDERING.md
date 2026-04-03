# PyMOL Rendering -- Structure Visualization Protocol

## Environment
- Installation: `pip install pymol-open-source` (installed in `/data/oih/miniconda/`)
- Command-line headless rendering: `/data/oih/miniconda/bin/pymol -cq script.py`
- No X11/display needed, `-cq` = command-line + quiet

## Standard Publication Settings (Nature Communications)
```python
from pymol import cmd

# White background, no shadows
cmd.set("cartoon_fancy_helices", 1)
cmd.set("ray_opaque_background", 1)
cmd.bg_color("white")
cmd.set("ray_shadows", 0)
cmd.set("antialias", 2)

# Render
cmd.orient("object_name")
cmd.ray(2400, 1800)
cmd.png("/data/oih/outputs/paper_figures/output.png", dpi=300)
cmd.quit()
```

## Common Operations

### Load and Clean
```python
cmd.load("/data/oih/outputs/fetch_pdb/XXXX.pdb", "protein")
cmd.remove("resn HOH")       # Remove water molecules
cmd.remove("hetatm")         # Remove ligands/ions
cmd.remove("not chain A")    # Keep only the target chain
```

### Display Modes
```python
cmd.show("cartoon", "protein")       # Cartoon backbone
cmd.show("sticks", "selection")      # Side chain sticks
cmd.show("spheres", "selection")     # Sphere highlights
cmd.show("surface", "protein")       # Surface
cmd.hide("everything", "selection")  # Hide
```

### Color Scheme
- Base color: `gray80` (light gray) or `gray70` (dark gray)
- Highlight color: `marine` (blue), `red`, `yellow`, `cyan`, `orange`, `lightpink`
- Error markers: `red` (wrong), `marine` (correct)

```python
cmd.color("gray80", "protein")
cmd.color("marine", "domain_selection")
cmd.color("yellow", "hotspot_selection")
```

### Select Residues
```python
# By residue number
cmd.select("hotspots", "resi 558+560+571+572+573 and chain C")
# By residue range
cmd.select("domain4", "resi 510-630 and chain C")
# By residue type
cmd.select("lysines", "resn LYS and resi 188")
```

### Alignment/Superposition
```python
cmd.align("mobile_object and chain X", "target_object and chain Y")
```

### Transparent Ghost Overlay
```python
cmd.set("cartoon_transparency", 0.65, "ghost_object")
cmd.color("lightpink", "ghost_object")
```

### Sphere Hotspot Visualization
```python
cmd.show("spheres", "hotspots")
cmd.set("sphere_scale", 0.4, "hotspots")  # 0.4-0.6 suitable for hotspots
```

### Labels
```python
cmd.set("label_size", 18)
cmd.set("label_color", "red", "selection")
```

## Practical Templates

### Template 1: Single Protein Domain Highlighting (EGFR fig2c style)
Purpose: Display different protein domains, mark correct/incorrect hotspots
```python
from pymol import cmd

cmd.load("/data/oih/outputs/fetch_pdb/1YY9.pdb", "egfr")
cmd.remove("resn HOH")
cmd.remove("hetatm")
cmd.remove("not chain A")

cmd.show("cartoon", "egfr")
cmd.color("gray80", "egfr")

# Incorrect hotspots - red
cmd.select("wrong", "resi 86-89 and chain A")
cmd.color("red", "wrong")
cmd.show("sticks", "wrong")

# Correct hotspots - blue
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

### Template 2: Domain Highlighting + Hotspot Spheres (HER2 fig2d style)
Purpose: Display protein domains + sphere markers for drug binding hotspots
```python
from pymol import cmd

cmd.load("/data/oih/outputs/fetch_pdb/1N8Z.pdb", "her2")
cmd.remove("resn HOH")
cmd.remove("not chain C")

cmd.show("cartoon", "her2")
cmd.color("gray70", "her2")

# Domain IV highlight
cmd.select("domain4", "resi 510-630 and chain C")
cmd.color("marine", "domain4")

# Hotspot spheres
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

### Template 3: Complex Superposition Comparison (fig5d style)
Purpose: Overlay designed binder with known antibody for comparison
```python
import glob
from pymol import cmd

# Load designed complex
cif = glob.glob("/data/oih/outputs/job_name/alphafold3/job_name/*_model.cif")
cmd.load(cif[0], "designed")

# Load reference antibody
cmd.load("/data/oih/outputs/fetch_pdb/1N8Z.pdb", "reference")

# Designed complex coloring
cmd.show("cartoon", "designed")
cmd.select("des_binder", "designed and chain A")
cmd.select("des_antigen", "designed and chain B")
cmd.color("cyan", "des_binder")
cmd.color("orange", "des_antigen")

# Reference antibody: semi-transparent pink ghost
cmd.show("cartoon", "reference and chain A")
cmd.show("cartoon", "reference and chain B")
cmd.color("lightpink", "reference")
cmd.set("cartoon_transparency", 0.65, "reference")

# Hide unneeded chains from reference
cmd.hide("everything", "reference and chain C")
cmd.hide("everything", "reference and chain D")

# Align: reference antigen chain to designed antigen chain
cmd.align("reference and chain C", "des_antigen")

# Key residue markers (e.g. conjugation site K188)
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

## Output
- All images saved to `/data/oih/outputs/paper_figures/`
- Naming convention: `fig{N}{letter}_{description}.png`

## Tips
- **First** remove HOH and HETATM after loading, then make selections
- **Must** call `cmd.orient()` before rendering to adjust the view
- `sphere_scale` 0.4-0.6 is suitable for hotspot visualization; too large will obscure cartoon
- `cartoon_transparency` 0.5-0.7 is suitable for ghost overlays; too low hides the background
- For multi-chain PDBs, confirm chain IDs first (using gemmi or PyMOL `cmd.get_chains()`)
- CIF files can be loaded directly with `cmd.load()`, PyMOL supports mmCIF
- **Must** call `cmd.quit()` at the end of scripts, otherwise the process won't exit
