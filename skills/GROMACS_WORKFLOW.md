# GROMACS MD模拟完整流程（已验证）

## 环境信息
- 容器名：oih-gromacs
- 版本：GROMACS 2024.4
- GPU命令：gmx mdrun -gpu_id 0（容器内永远用0）
- 工作目录：/data/oih/outputs/test_gromacs/

## ⚠️ GPU注意事项
容器内 gpu_id 永远用 **0**，不能用 1。
原因：NVIDIA_VISIBLE_DEVICES=1 已将宿主机GPU1映射为容器内GPU0。

## 完整8步流程

### 步骤1: pdb2gmx — 生成拓扑文件
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
- `echo '6'` → 选择 AMBER99SB-ILDN 力场
- `-ignh` → 忽略氢原子（自动添加）
- 输出：protein.gro, topol.top, posre.itp

### 步骤2: editconf — 定义模拟盒子
```bash
docker exec oih-gromacs bash -c "
cd /data/oih/outputs/test_gromacs &&
gmx editconf -f protein.gro -o protein_box.gro \
  -c -d 1.0 -bt cubic 2>&1 | tail -3
"
```
- `-d 1.0` → 蛋白质距盒子边界1.0nm
- `-bt cubic` → 立方体盒子

### 步骤3: solvate — 加水
```bash
docker exec oih-gromacs bash -c "
cd /data/oih/outputs/test_gromacs &&
gmx solvate -cp protein_box.gro -cs spc216.gro \
  -o protein_solv.gro -p topol.top 2>&1 | tail -3
"
```

### 步骤4: genion — 加离子（中和电荷）
```bash
# 先生成 ions.tpr
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
- `echo 'SOL'` → 选择溶剂组替换为离子
- `-neutral` → 自动添加足够离子中和体系电荷

### 步骤5: em — 能量最小化
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
验证结果：2443步收敛，势能 -4.97×10⁶ kJ/mol，最大力 810 kJ/mol/nm < 1000阈值 ✅

### 步骤6: nvt — NVT平衡（控温，100ps）
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
验证结果：性能 152 ns/day ✅

### 步骤7: npt — NPT平衡（控温控压，100ps）
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
验证结果：性能 162 ns/day ✅

### 步骤8: md — 正式MD模拟（1ns，可调）
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

## CPU降级命令（GPU不可用时）
```bash
gmx mdrun -v -deffnm md -nb cpu -pme cpu -ntmpi 1 -ntomp 32
```

## 小分子MD（蛋白-配体复合物）
纯蛋白MD不需要额外工具，但蛋白+小分子需要：
1. acpype 生成小分子 GAFF2 力场参数（容器内未安装，需加入Dockerfile）
2. 合并蛋白+配体 complex.pdb 再走正常流程

安装acpype：在 Dockerfile 中加 `RUN pip install acpype`

## 注意事项
- 力场选择：AMBER99SB-ILDN（序号6）适合蛋白质
- 水模型：SPC/E（spce）
- 1IVO.pdb 测试体系：1021个蛋白残基，89255个水分子，283376原子

## 小分子MD（蛋白-配体复合物）完整流程（已验证）

### 前置条件
- 宿主机：obabel（/usr/bin/obabel）、acpype、AmberTools（tleap/parmchk2）
- 配体输入：SDF格式（来自gnina/vina对接输出）

### 步骤0: 配体参数化（宿主机）
```bash
# SDF第一个pose → mol2
obabel <poses.sdf> -O ligand.mol2 -f 1 -l 1

# mol2 → GAFF2力场参数（用gasteiger电荷，bcc需要量化计算）
acpype -i ligand.mol2 -b LIG -c gas -a gaff2 -o gmx
# 输出目录：LIG.acpype/
# 关键文件：LIG_GMX.gro, LIG_GMX.itp, posre_LIG.itp
```
⚠️ BCC电荷（-c bcc）需要sqm量化计算，较慢且可能失败；gasteiger（-c gas）足够用于MD。

### 步骤1: 合并蛋白+配体复合物
```bash
# 去掉protein.pdb的END行，拼接配体
grep -v "^END" protein.pdb > complex.pdb
grep "^HETATM\|^ATOM" LIG.acpype/LIG_GMX.gro >> complex.pdb  # 或转换gro→pdb
echo "END" >> complex.pdb
```

### 步骤2: 修改topol.top加入配体
```
; 在topol.top末尾加入：
#include "LIG.acpype/LIG_GMX.itp"
; 在[ molecules ]中加入：
LIG    1
```

### 步骤3: 走正常GROMACS 8步流程
从pdb2gmx开始，force_field用AMBER99SB-ILDN（与GAFF2兼容）

<!-- AUTO_SYNC_FROM_CLAUDE_MD -->
## ⚠️ 注意事项（自动同步自 CLAUDE.md）

- 容器内 NVIDIA_VISIBLE_DEVICES=1，永远用 device=0 / gpu_id=0（不要用1）
- 1. **tc-grps 必须用 `Protein_LIG Water_and_ions`**（不是默认的 `Protein Non-Protein`），否则 NVT grompp 报 "group not found"
- 2. **每步 mdrun 之后必须验证输出文件存在**：em.gro → nvt.gro → npt.gro → md.xtc，任何一步缺失立刻 raise 并附 .log 最后20行
- 3. **`gmx make_ndx` 创建 Protein_LIG 组**：在 genion 之后、EM 之前运行 `echo '1 | 13
q' | gmx make_ndx` 合并 Protein 和 LIG 组
- 4. **不要静默忽略 mdrun 返回值**：`retcode != 0 and "WARNING" not in stderr` 这种判断不安全，WARNING 可能掩盖真实错误
- 1. **tc-grps 动态检测**：`make_ndx` 后解析 `index.ndx` 找实际合并组名（如 `Protein_UNL`），不再硬编码 `Protein_LIG`
- 2. **MDP 延迟生成**：NVT/NPT/MD 的 MDP 在 `_run()` 内 `make_ndx` 之后才写入，确保 tc-grps 正确
- 3. **每步文件检查**：em.gro → nvt.gro → npt.gro → md.xtc，缺失立即 raise + 附 .log 最后 20 行

<!-- /AUTO_SYNC_FROM_CLAUDE_MD -->
