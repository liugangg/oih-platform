# GROMACS MD Simulation Complete Workflow (Verified)

## Environment Information
- Container name: oih-gromacs
- Version: GROMACS 2024.4
- GPU command: gmx mdrun -gpu_id 0 (always use 0 inside the container)
- Working directory: /data/oih/outputs/test_gromacs/

## ⚠️ GPU Notes
Always use gpu_id **0** inside the container; never use 1.
Reason: NVIDIA_VISIBLE_DEVICES=1 maps host GPU1 to container GPU0.

## Complete 8-Step Workflow

### Step 1: pdb2gmx — Generate Topology Files
```bash
docker exec oih-gromacs bash -c "
cd /data/oih/outputs/test_gromacs &&
echo '6' | gmx pdb2gmx \
  -f /data/oih/inputs/1IVO.pdb \
  -o protein.gro \
  -water spce \
  -ignh 2>&1 | tail -5
"
```
- `echo '6'` → Select AMBER99SB-ILDN force field
- `-ignh` → Ignore hydrogen atoms (automatically added)
- Output: protein.gro, topol.top, posre.itp

### Step 2: editconf — Define Simulation Box
```bash
docker exec oih-gromacs bash -c "
cd /data/oih/outputs/test_gromacs &&
gmx editconf -f protein.gro -o protein_box.gro \
  -c -d 1.0 -bt cubic 2>&1 | tail -3
"
```
- `-d 1.0` → 1.0 nm distance from protein to box boundary
- `-bt cubic` → Cubic box

### Step 3: solvate — Add Water
```bash
docker exec oih-gromacs bash -c "
cd /data/oih/outputs/test_gromacs &&
gmx solvate -cp protein_box.gro -cs spc216.gro \
  -o protein_solv.gro -p topol.top 2>&1 | tail -3
"
```

### Step 4: genion — Add Ions (Neutralize Charge)
```bash
# First generate ions.tpr
docker exec oih-gromacs bash -c "
cd /data/oih/outputs/test_gromacs &&
cat > ions.mdp << 'MDP'
integrator  = steep
emtol       = 1000.0
nsteps      = 50000
nstlist     = 10
cutoff-scheme = Verlet
coulombtype = cutoff
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
MDP
gmx grompp -f ions.mdp -c protein_solv.gro -p topol.top -o ions.tpr -maxwarn 2 2>&1 | tail -3 &&
echo 'SOL' | gmx genion -s ions.tpr -o protein_ions.gro \
  -p topol.top -pname NA -nname CL -neutral 2>&1 | tail -5
"
```
- `echo 'SOL'` → Select solvent group to replace with ions
- `-neutral` → Automatically add sufficient ions to neutralize the system charge

### Step 5: em — Energy Minimization
```bash
docker exec oih-gromacs bash -c "
cd /data/oih/outputs/test_gromacs &&
cat > em.mdp << 'MDP'
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000
nstlist     = 10
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
MDP
gmx grompp -f em.mdp -c protein_ions.gro -p topol.top -o em.tpr -maxwarn 2 2>&1 | tail -3 &&
gmx mdrun -v -deffnm em -gpu_id 0 -nb gpu -ntmpi 1 -ntomp 16 2>&1 | tail -5
"
```
Verification: Converged in 2443 steps, potential energy -4.97×10⁶ kJ/mol, maximum force 810 kJ/mol/nm < 1000 threshold ✅

### Step 6: nvt — NVT Equilibration (Temperature Coupling, 100 ps)
```bash
docker exec oih-gromacs bash -c "
cd /data/oih/outputs/test_gromacs &&
cat > nvt.mdp << 'MDP'
integrator  = md
nsteps      = 50000
dt          = 0.002
nstxout     = 500
nstvout     = 500
nstenergy   = 500
nstlog      = 500
continuation = no
constraint_algorithm = lincs
constraints = h-bonds
nstlist     = 10
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
tcoupl      = V-rescale
tc-grps     = Protein Non-Protein
tau_t       = 0.1 0.1
ref_t       = 300 300
pcoupl      = no
gen_vel     = yes
gen_temp    = 300
gen_seed    = -1
MDP
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 2 2>&1 | tail -3 &&
gmx mdrun -v -deffnm nvt -gpu_id 0 -nb gpu -pme gpu -update gpu -ntmpi 1 -ntomp 16 2>&1 | tail -8
"
```
Verification: Performance 152 ns/day ✅

### Step 7: npt — NPT Equilibration (Temperature and Pressure Coupling, 100 ps)
```bash
docker exec oih-gromacs bash -c "
cd /data/oih/outputs/test_gromacs &&
cat > npt.mdp << 'MDP'
integrator  = md
nsteps      = 50000
dt          = 0.002
nstenergy   = 500
nstlog      = 500
continuation = yes
constraint_algorithm = lincs
constraints = h-bonds
nstlist     = 10
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
tcoupl      = V-rescale
tc-grps     = Protein Non-Protein
tau_t       = 0.1 0.1
ref_t       = 300 300
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = 1.0
compressibility = 4.5e-5
refcoord_scaling = com
gen_vel     = no
MDP
gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 2 2>&1 | tail -3 &&
gmx mdrun -v -deffnm npt -gpu_id 0 -nb gpu -pme gpu -update gpu -ntmpi 1 -ntomp 16 2>&1 | tail -8
"
```
Verification: Performance 162 ns/day ✅

### Step 8: md — Production MD Simulation (1 ns, Adjustable)
```bash
docker exec oih-gromacs bash -c "
cd /data/oih/outputs/test_gromacs &&
cat > md.mdp << 'MDP'
integrator  = md
nsteps      = 500000
dt          = 0.002
nstenergy   = 5000
nstlog      = 5000
nstxout-compressed = 5000
continuation = yes
constraint_algorithm = lincs
constraints = h-bonds
nstlist     = 10
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
tcoupl      = V-rescale
tc-grps     = Protein Non-Protein
tau_t       = 0.1 0.1
ref_t       = 300 300
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = 1.0
compressibility = 4.5e-5
gen_vel     = no
MDP
gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr -maxwarn 2 2>&1 | tail -3 &&
gmx mdrun -v -deffnm md -gpu_id 0 -nb gpu -pme gpu -update gpu -ntmpi 1 -ntomp 16 2>&1 | tail -8
"
```

## CPU Fallback Command (When GPU Is Unavailable)
```bash
gmx mdrun -v -deffnm md -nb cpu -pme cpu -ntmpi 1 -ntomp 32
```

## Small Molecule MD (Protein-Ligand Complex)
Pure protein MD does not require additional tools, but protein + small molecule requires:
1. acpype to generate small molecule GAFF2 force field parameters (not installed in the container; must be added to Dockerfile)
2. Merge protein + ligand into complex.pdb, then follow the standard workflow

Install acpype: Add `RUN pip install acpype` to the Dockerfile

## Notes
- Force field selection: AMBER99SB-ILDN (option 6) is suitable for proteins
- Water model: SPC/E (spce)
- 1IVO.pdb test system: 1021 protein residues, 89,255 water molecules, 283,376 atoms

## Small Molecule MD (Protein-Ligand Complex) Complete Workflow (Verified)

### Prerequisites
- Host machine: obabel (/usr/bin/obabel), acpype, AmberTools (tleap/parmchk2)
- Ligand input: SDF format (from gnina/vina docking output)

### Step 0: Ligand Parameterization (Host Machine)
```bash
# First pose from SDF → mol2
obabel <poses.sdf> -O ligand.mol2 -f 1 -l 1

# mol2 → GAFF2 force field parameters (using Gasteiger charges; BCC requires quantum calculation)
acpype -i ligand.mol2 -b LIG -c gas -a gaff2 -o gmx
# Output directory: LIG.acpype/
# Key files: LIG_GMX.gro, LIG_GMX.itp, posre_LIG.itp
```
⚠️ BCC charges (-c bcc) require sqm quantum calculation, which is slow and may fail; Gasteiger (-c gas) is sufficient for MD.

### Step 1: Merge Protein + Ligand Complex
```bash
# Remove END line from protein.pdb, then append ligand
grep -v "^END" protein.pdb > complex.pdb
grep "^HETATM\|^ATOM" LIG.acpype/LIG_GMX.gro >> complex.pdb  # or convert gro→pdb
echo "END" >> complex.pdb
```

### Step 2: Modify topol.top to Include Ligand
```
; Add at the end of topol.top:
#include "LIG.acpype/LIG_GMX.itp"
; Add to [ molecules ] section:
LIG    1
```

### Step 3: Follow Standard GROMACS 8-Step Workflow
Start from pdb2gmx, using AMBER99SB-ILDN force field (compatible with GAFF2)

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## ⚠️ Notes (Auto-synced from CLAUDE.md)

- Inside the container NVIDIA_VISIBLE_DEVICES=1; always use device=0 / gpu_id=0 (never use 1)
- 1. **tc-grps must use `Protein_LIG Water_and_ions`** (not the default `Protein Non-Protein`), otherwise NVT grompp reports "group not found"
- 2. **Verify output files exist after each mdrun step**: em.gro → nvt.gro → npt.gro → md.xtc; if any step is missing, raise immediately and attach the last 20 lines of the .log
- 3. **`gmx make_ndx` to create the Protein_LIG group**: Run `echo '1 | 13
q' | gmx make_ndx` after genion and before EM to merge the Protein and LIG groups
- 4. **Do not silently ignore mdrun return values**: The check `retcode != 0 and "WARNING" not in stderr` is unsafe; WARNING may mask real errors
- 1. **Dynamic tc-grps detection**: After `make_ndx`, parse `index.ndx` to find the actual merged group name (e.g., `Protein_UNL`); do not hardcode `Protein_LIG`
- 2. **Deferred MDP generation**: NVT/NPT/MD MDP files are written inside `_run()` after `make_ndx` to ensure tc-grps is correct
- 3. **Per-step file check**: em.gro → nvt.gro → npt.gro → md.xtc; raise immediately if missing + attach last 20 lines of .log

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->
