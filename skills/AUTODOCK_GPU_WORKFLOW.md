# AutoDock-GPU Complete Preprocessing Workflow (Verified)

## Tool Distribution
- obabel, autogrid4: host machine /usr/bin/
- autodock_gpu_128wi: container oih-autodock-gpu /usr/local/bin/
- prody: Python inside the oih-autodock-gpu container

## Complete Workflow

### Step 1: Clean PDB with prody (inside container)
```python
from prody import parsePDB, writePDB
struct = parsePDB('input.pdb', altloc='first')
protein = struct.select('protein and chain A B')  # keep only protein chains
writePDB('protein_clean.pdb', protein)
```

### Step 2: Generate pdbqt with charges using obabel (host machine)
```bash
obabel protein_clean.pdb -O receptor.pdbqt -xr --partialcharge gasteiger
```
Note: -xr means rigid receptor; --partialcharge gasteiger is required, otherwise charges are 0 and autogrid4 will error

### Step 3: Obtain pocket coordinates (automatically extracted from fpocket results)
```python
# Parse box_center and box_size from fpocket output
# box_center = pocket['x_cent'], pocket['y_cent'], pocket['z_cent']
# box_size defaults to 25 25 25 (adjustable based on pocket volume)
```

### Step 4: Generate GPF file (host machine)
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
Note: receptor_types must match the actual atom types in the pdbqt file

### Step 5: Generate grid with autogrid4 (host machine)
```bash
cd /workdir && autogrid4 -p receptor.gpf -l receptor.glg
```
Output files: receptor.maps.fld, receptor.*.map

### Step 6: Docking with autodock_gpu_128wi (inside container)
```bash
autodock_gpu_128wi \
  --ffile receptor.maps.fld \
  --lfile ligand.pdbqt \
  --nrun 20 \
  --devnum 1 \
  --resnam result
```

### Ligand Preparation (from SMILES)
```bash
# Generate 3D pdbqt from SMILES using obabel on host machine
obabel -:"SMILES_STRING" -opdbqt --gen3d -O ligand.pdbqt --partialcharge gasteiger
```

## Non-standard Residue Handling Strategy
- Detect HETATM: grep HETATM input.pdb | awk '{print $4}' | sort -u
- NAG/glycosylation etc.: automatically excluded when using 'protein' selector in prody
- If obabel reports kekulize warnings: usually does not affect results, can be ignored

## Error Handling
- autogrid4 "atom types mismatch": check actual atom types in pdbqt and rewrite gpf
- autogrid4 "no partial charges": regenerate pdbqt with --partialcharge gasteiger
- meeko HasQuery error: meeko version incompatible with rdkit, use obabel as alternative

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## Notes (auto-synced from CLAUDE.md)

- Inside the container NVIDIA_VISIBLE_DEVICES=1; always use device=0 / gpu_id=0 (do not use 1)

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->
