"""
MD Simulation Router
Tools: GROMACS (GPU-accelerated)
"""
# --- SYNC_NOTES (auto-generated from CLAUDE.md, do not edit) ---
# GROMACS 注意事项（来自 CLAUDE.md，勿手动编辑）：
#   - 容器内 NVIDIA_VISIBLE_DEVICES=1，永远用 device=0 / gpu_id=0（不要用1）
#   - 1. **tc-grps 必须用 `Protein_LIG Water_and_ions`**（不是默认的 `Protein Non-Protein`），否则 NVT grompp 报...
#   - 2. **每步 mdrun 之后必须验证输出文件存在**：em.gro → nvt.gro → npt.gro → md.xtc，任何一步缺失立刻 raise 并附 .log 最后20行
#   - 3. **`gmx make_ndx` 创建 Protein_LIG 组**：在 genion 之后、EM 之前运行 `echo '1 | 13 q' | gmx make_ndx` ...
#   - 4. **不要静默忽略 mdrun 返回值**：`retcode != 0 and "WARNING" not in stderr` 这种判断不安全，WARNING 可能掩盖真实错误
#   - 1. **tc-grps 动态检测**：`make_ndx` 后解析 `index.ndx` 找实际合并组名（如 `Protein_UNL`），不再硬编码 `Protein_LIG`
#   - 2. **MDP 延迟生成**：NVT/NPT/MD 的 MDP 在 `_run()` 内 `make_ndx` 之后才写入，确保 tc-grps 正确
#   - 3. **每步文件检查**：em.gro → nvt.gro → npt.gro → md.xtc，缺失立即 raise + 附 .log 最后 20 行
# --- /SYNC_NOTES ---

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)
from fastapi import APIRouter
from schemas.models import GROMACSRequest, TaskRef
from core.task_manager import task_manager
from core.docker_client import run_in_container_streaming, run_in_container
from core.config import settings

router = APIRouter()

# Pre-built MDP templates
MDP_MINIM = """
; Energy minimization
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000
nstlist     = 1
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
"""

def _make_nvt_mdp(temp: int, tc_grps: str = "Protein Non-Protein") -> str:
    return f"""
; NVT equilibration
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
lincs_iter  = 1
lincs_order = 4
cutoff-scheme = Verlet
nstlist     = 10
rcoulomb    = 1.0
rvdw        = 1.0
rvdw-switch = 0.9
coulombtype = PME
pme_order   = 4
fourierspacing = 0.16
tcoupl      = V-rescale
tc-grps     = {tc_grps}
tau_t       = 0.1 0.1
ref_t       = {temp} {temp}
pcoupl      = no
pbc         = xyz
DispCorr    = EnerPres
gen_vel     = yes
gen_temp    = {temp}
gen_seed    = -1
"""


def _make_npt_mdp(temp: int, tc_grps: str = "Protein Non-Protein") -> str:
    return f"""
; NPT equilibration
integrator  = md
nsteps      = 50000
dt          = 0.002
continuation = yes
constraint_algorithm = lincs
constraints = h-bonds
cutoff-scheme = Verlet
nstlist     = 10
rcoulomb    = 1.0
rvdw        = 1.0
coulombtype = PME
pme_order   = 4
fourierspacing = 0.16
tcoupl      = V-rescale
tc-grps     = {tc_grps}
tau_t       = 0.1 0.1
ref_t       = {temp} {temp}
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = 1.0
compressibility = 4.5e-5
pbc         = xyz
DispCorr    = EnerPres
gen_vel     = no
"""


def _make_md_mdp(nsteps: int, temp: int, tc_grps: str = "Protein Non-Protein") -> str:
    return f"""
; Production MD
integrator  = md
nsteps      = {nsteps}
dt          = 0.002
nstxout-compressed = 5000
nstvout     = 5000
nstenergy   = 5000
nstlog      = 5000
continuation = yes
constraint_algorithm = lincs
constraints = h-bonds
cutoff-scheme = Verlet
nstlist     = 10
rcoulomb    = 1.0
rvdw        = 1.0
coulombtype = PME
pme_order   = 4
fourierspacing = 0.16
tcoupl      = V-rescale
tc-grps     = {tc_grps}
tau_t       = 0.1 0.1
ref_t       = {temp} {temp}
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = 1.0
compressibility = 4.5e-5
pbc         = xyz
DispCorr    = EnerPres
gen_vel     = no
"""


@router.post("/gromacs", response_model=TaskRef, summary="Run GROMACS MD simulation")
async def run_gromacs(req: GROMACSRequest):
    """
    Run full GROMACS MD pipeline:
    1. PDB2GMX (topology)
    2. Box setup + solvation
    3. Energy minimization
    4. NVT equilibration
    5. NPT equilibration
    6. Production MD (GPU-accelerated)

    Supports protein-water, protein-ligand, and membrane-protein systems.
    """
    job_dir = os.path.join(settings.OUTPUT_DIR, req.job_name, "gromacs")
    os.makedirs(job_dir, exist_ok=True)

    # Write MDP files
    mdp_dir = os.path.join(settings.INPUT_DIR, req.job_name, "mdp")
    os.makedirs(mdp_dir, exist_ok=True)

    nsteps_prod = int(req.sim_time_ns * 1e6 / 2)  # 2fs timestep, ns→ps→steps

    # Only write minim.mdp now; NVT/NPT/MD MDPs are written inside _run()
    # after make_ndx so we know the actual tc-grps group name
    with open(os.path.join(mdp_dir, "minim.mdp"), "w") as f:
        f.write(MDP_MINIM)

    async def _run(task):
        workdir = f"/data/oih/outputs/{req.job_name}/gromacs"
        pdb_in = req.input_pdb if os.path.exists(req.input_pdb) else f"/data/oih/inputs/{os.path.basename(req.input_pdb)}"
        mdp_base = f"/data/oih/inputs/{req.job_name}/mdp"
        gpu_flag = "--gpu_id 0"  # 容器内NVIDIA_VISIBLE_DEVICES=1已映射为GPU0

        # ── pdbfixer: complete missing sidechain heavy atoms before pdb2gmx ──
        try:
            from pdbfixer import PDBFixer
            from openmm.app import PDBFile
            fixer = PDBFixer(filename=pdb_in)
            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixed_pdb = os.path.join(workdir, "input_fixed.pdb")
            os.makedirs(workdir, exist_ok=True)
            with open(fixed_pdb, "w") as f:
                PDBFile.writeFile(fixer.topology, fixer.positions, f)
            pdb_in = fixed_pdb
            logger.info(f"[gromacs] pdbfixer completed missing atoms → {fixed_pdb}")
        except Exception as e:
            logger.warning(f"[gromacs] pdbfixer failed ({e}), using raw PDB with -missing flag")

        # ── 小分子配体参数化（如果提供了ligand_sdf）──────────────────────────
        ligand_itp = None
        ligand_gro = None
        if req.ligand_sdf:
            task.progress = 3
            task.progress_msg = "Parameterizing ligand with GAFF2 (acpype)..."
            lig_sdf = req.ligand_sdf
            lig_workdir = f"/data/oih/outputs/{req.job_name}/ligand"
            os.makedirs(lig_workdir, exist_ok=True)

            # SDF → mol2（取第一个pose）
            mol2_path = f"{lig_workdir}/ligand.mol2"
            r = subprocess.run(
                ["obabel", lig_sdf, "-O", mol2_path, "-f", "1", "-l", "1"],
                capture_output=True, text=True
            )
            if not os.path.exists(mol2_path):
                raise RuntimeError(f"obabel SDF→mol2 failed: {r.stderr}")

            # acpype GAFF2参数化
            r = subprocess.run(
                ["acpype", "-i", mol2_path, "-b", "LIG", "-c", "gas", "-a", "gaff2", "-o", "gmx"],
                capture_output=True, text=True, cwd=lig_workdir
            )
            lig_acpype_dir = f"{lig_workdir}/LIG.acpype"
            if not os.path.exists(lig_acpype_dir):
                raise RuntimeError(f"acpype failed: {r.stderr}")

            ligand_itp = f"{lig_acpype_dir}/LIG_GMX.itp"
            ligand_gro = f"{lig_acpype_dir}/LIG_GMX.gro"
            posre_itp  = f"{lig_acpype_dir}/posre_LIG.itp"

            # pdb2gmx只处理蛋白质，配体通过itp单独引入
            # pdb_in already points to pdbfixer-cleaned PDB (set above)
            task.progress_msg = "Ligand parameterized ✅ — proceeding with complex MD..."

        # Step 1: pdb2gmx (protein topology only)
        task.progress = 10
        task.progress_msg = "Generating topology..."
        step1_cmd = (
            f"chmod -R 777 {workdir} && "
            f"cd {workdir} && gmx pdb2gmx -f {pdb_in} -o {workdir}/protein.gro "
            f"-water {req.water_model} -ff {req.forcefield} -ignh -missing -p {workdir}/topol.top -q {workdir}/clean.pdb && "
            f"chmod -R 777 {workdir}"
        )
        retcode, stdout, stderr = await run_in_container(
            settings.CONTAINER_GROMACS,
            ["bash", "-c", step1_cmd],
            timeout=settings.TIMEOUT_GROMACS,
        )
        if retcode != 0:
            raise RuntimeError(f"GROMACS step failed: Generating topology...\n{stderr[-2000:]}")

        # Step 1b: if ligand, merge protein.gro + LIG_GMX.gro → complex.gro BEFORE editconf
        # This ensures editconf -c centers the entire complex (protein + ligand) together.
        editconf_input = f"{workdir}/protein.gro"
        if ligand_itp:
            task.progress = 15
            task.progress_msg = "Merging protein + ligand coordinates..."
            import shutil
            shutil.copy(ligand_itp, f"{workdir}/LIG_GMX.itp")
            shutil.copy(posre_itp,  f"{workdir}/posre_LIG.itp")

            # Update topol.top: add LIG_GMX.itp include and LIG 1 to molecules
            import re as _re
            topol_path = f"{workdir}/topol.top"
            with open(topol_path) as f:
                topol = f.read()
            if '#include "LIG_GMX.itp"' not in topol:
                topol = topol.replace(
                    "; Include chain topologies\n",
                    "; Include chain topologies\n#include \"LIG_GMX.itp\"\n"
                )
                topol = _re.sub(
                    r"(\[ molecules \][^\[]*?)(\nSOL |\Z)",
                    lambda m: m.group(1) + "\nLIG                  1" + m.group(2),
                    topol, count=1, flags=_re.DOTALL
                )
            with open(topol_path, "w") as f:
                f.write(topol)

            # Merge GRO files: protein.gro + LIG_GMX.gro → complex.gro
            with open(f"{workdir}/protein.gro") as f:
                prot_lines = f.readlines()
            with open(ligand_gro) as f:
                lig_lines = f.readlines()
            prot_natoms = int(prot_lines[1].strip())
            lig_atoms   = lig_lines[2:-1]
            lig_natoms  = len(lig_atoms)
            new_natoms  = prot_natoms + lig_natoms
            complex_gro = [prot_lines[0], f"{new_natoms}\n"] + prot_lines[2:-1] + lig_atoms + [prot_lines[-1]]
            with open(f"{workdir}/complex.gro", "w") as f:
                f.writelines(complex_gro)
            editconf_input = f"{workdir}/complex.gro"

        steps = [
            # 2. Define box (center protein+ligand complex together)
            (20, "Setting up simulation box...",
             f"gmx editconf -f {editconf_input} -o {workdir}/box.gro "
             f"-c -d {req.box_padding_nm} -bt cubic"),

            # 3. Solvate
            (25, "Solvating...",
             f"chmod 777 {workdir} && "
             f"cd {workdir} && "
             f"gmx solvate -cp {workdir}/box.gro -cs spc216.gro "
             f"-o {workdir}/solv.gro -p {workdir}/topol.top && "
             f"chmod -R 777 {workdir}"),

            # 4. Add ions (grompp first, then genion)
            (30, "Adding ions...",
             f"cd {workdir} && "
             f"cp {workdir}/*.itp /tmp/ 2>/dev/null || true && "
             f"gmx grompp -f {mdp_base}/minim.mdp -c {workdir}/solv.gro "
             f"-p {workdir}/topol.top -o /tmp/ions.tpr -maxwarn 5 && "
             f"echo SOL | gmx genion -s /tmp/ions.tpr "
             f"-o {workdir}/solv_ions.gro -p {workdir}/topol.top -pname NA -nname CL -neutral && "
             f"chmod -R 777 {workdir}"),
        ]

        for progress_val, msg, bash_cmd in steps:
            task.progress = progress_val
            task.progress_msg = msg
            retcode, stdout, stderr = await run_in_container(
                settings.CONTAINER_GROMACS,
                ["bash", "-c", bash_cmd],
                timeout=settings.TIMEOUT_GROMACS,
            )
            if retcode != 0:
                raise RuntimeError(f"GROMACS step failed: {msg}\n{stderr[-2000:]}")

        # Detect tc-grps and write NVT/NPT/MD MDP files
        # For protein-ligand: create index group, detect actual merged group name
        tc_grps = "Protein Non-Protein"
        ndx_flag = ""

        if ligand_itp:
            task.progress = 35
            task.progress_msg = "Creating index groups for protein+ligand..."
            # Use make_ndx: merge Protein with the ligand group (often "Other" or "UNL" or "LIG")
            # Try "1 | 13" first (Protein | Other), fallback to "Protein | UNL"
            ndx_cmd = (
                f"cd {workdir} && "
                f"echo -e '1 | 13\\nq\\n' | gmx make_ndx -f {workdir}/solv_ions.gro "
                f"-o {workdir}/index.ndx 2>&1"
            )
            retcode, stdout, stderr = await run_in_container(
                settings.CONTAINER_GROMACS,
                ["bash", "-c", ndx_cmd],
                timeout=60,
            )
            logger.info("[gromacs] make_ndx retcode=%d", retcode)

            # Parse the ndx file to find the merged group name
            ndx_host = f"{job_dir}/index.ndx"
            if os.path.exists(ndx_host):
                ndx_flag = f"-n {workdir}/index.ndx"
                with open(ndx_host) as f:
                    ndx_groups = [line.strip("[] \n") for line in f if line.strip().startswith("[")]
                # Find the merged group: typically last one, named "Protein_UNL" or "Protein_Other" etc.
                merged = [g for g in ndx_groups if g.startswith("Protein_") and g != "Protein"]
                if merged:
                    tc_grps = f"{merged[-1]} Water_and_ions"
                    logger.info("[gromacs] Detected tc-grps: %s", tc_grps)
                else:
                    tc_grps = "Protein Non-Protein"
                    logger.warning("[gromacs] No merged Protein_* group found, falling back to Protein Non-Protein")
            else:
                logger.warning("[gromacs] index.ndx not created, using default tc-grps")

        # Now write NVT/NPT/MD MDP files with correct tc-grps
        with open(os.path.join(settings.INPUT_DIR, req.job_name, "mdp", "nvt.mdp"), "w") as f:
            f.write(_make_nvt_mdp(req.temperature_k, tc_grps))
        with open(os.path.join(settings.INPUT_DIR, req.job_name, "mdp", "npt.mdp"), "w") as f:
            f.write(_make_npt_mdp(req.temperature_k, tc_grps))
        with open(os.path.join(settings.INPUT_DIR, req.job_name, "mdp", "md.mdp"), "w") as f:
            f.write(_make_md_mdp(nsteps_prod, req.temperature_k, tc_grps))
        logger.info("[gromacs] MDP files written with tc-grps='%s'", tc_grps)

        def _read_log_tail(log_path: str, n: int = 20) -> str:
            """Read last n lines of a log file for error reporting."""
            host_path = log_path.replace("/data/oih/outputs", settings.OUTPUT_DIR)
            try:
                with open(host_path) as f:
                    lines = f.readlines()
                return "".join(lines[-n:])
            except Exception:
                return "(log file not readable)"

        # 5. Energy minimization
        task.progress = 40
        task.progress_msg = "Running energy minimization..."
        em_cmd = (
            f"cd {workdir} && gmx grompp -f {mdp_base}/minim.mdp -c {workdir}/solv_ions.gro "
            f"-p {workdir}/topol.top -o {workdir}/em.tpr -maxwarn 5 && "
            f"cd {workdir} && CUDA_VISIBLE_DEVICES=0 gmx mdrun -v -deffnm {workdir}/em -ntmpi 1 -ntomp 16 -gpu_id 0 -nb gpu"
        )
        retcode, stdout, stderr = await run_in_container(
            settings.CONTAINER_GROMACS, ["bash", "-c", em_cmd], timeout=settings.TIMEOUT_GROMACS)
        if retcode != 0:
            raise RuntimeError(f"EM failed:\n{stderr[-2000:]}")
        if not os.path.exists(f"{job_dir}/em.gro"):
            raise RuntimeError(f"EM failed: em.gro not generated.\n{_read_log_tail(f'{workdir}/em.log')}")

        # 6. NVT equilibration
        task.progress = 55
        task.progress_msg = "NVT equilibration..."
        nvt_cmd = (
            f"cd {workdir} && gmx grompp -f {mdp_base}/nvt.mdp -c {workdir}/em.gro "
            f"-r {workdir}/em.gro -p {workdir}/topol.top {ndx_flag} -o {workdir}/nvt.tpr -maxwarn 5 && "
            f"cd {workdir} && CUDA_VISIBLE_DEVICES=0 gmx mdrun -deffnm {workdir}/nvt -ntmpi 1 -ntomp 4 -gpu_id 0 -nb gpu -pme gpu"
        )
        retcode, stdout, stderr = await run_in_container(
            settings.CONTAINER_GROMACS, ["bash", "-c", nvt_cmd], timeout=settings.TIMEOUT_GROMACS)
        if retcode != 0:
            raise RuntimeError(f"NVT failed (check tc-grps, 蛋白配体体系需用 Protein_LIG Water_and_ions):\n{stderr[-2000:]}")
        if not os.path.exists(f"{job_dir}/nvt.gro"):
            raise RuntimeError(f"NVT failed: nvt.gro not generated. 检查 tc-grps 设置。\n{_read_log_tail(f'{workdir}/nvt.log')}")

        # 7. NPT equilibration
        task.progress = 70
        task.progress_msg = "NPT equilibration..."
        npt_cmd = (
            f"cd {workdir} && gmx grompp -f {mdp_base}/npt.mdp -c {workdir}/nvt.gro "
            f"-r {workdir}/nvt.gro -t {workdir}/nvt.cpt -p {workdir}/topol.top {ndx_flag} -o {workdir}/npt.tpr -maxwarn 5 && "
            f"cd {workdir} && CUDA_VISIBLE_DEVICES=0 gmx mdrun -deffnm {workdir}/npt -ntmpi 1 -ntomp 16 -gpu_id 0 -nb gpu -pme gpu -update gpu"
        )
        retcode, stdout, stderr = await run_in_container(
            settings.CONTAINER_GROMACS, ["bash", "-c", npt_cmd], timeout=settings.TIMEOUT_GROMACS)
        if retcode != 0:
            raise RuntimeError(f"NPT failed:\n{stderr[-2000:]}")
        if not os.path.exists(f"{job_dir}/npt.gro"):
            raise RuntimeError(f"NPT failed: npt.gro not generated.\n{_read_log_tail(f'{workdir}/npt.log')}")

        # 8. Production MD
        task.progress = 80
        task.progress_msg = f"Production MD ({req.sim_time_ns} ns)..."
        md_cmd = (
            f"cd {workdir} && gmx grompp -f {mdp_base}/md.mdp -c {workdir}/npt.gro "
            f"-t {workdir}/npt.cpt -p {workdir}/topol.top {ndx_flag} -o {workdir}/md.tpr -maxwarn 5 && "
            f"cd {workdir} && CUDA_VISIBLE_DEVICES=0 gmx mdrun -deffnm {workdir}/md -ntmpi 1 -ntomp 16 -gpu_id 0 -nb gpu -pme gpu -update gpu -bonded gpu"
        )
        retcode, stdout, stderr = await run_in_container(
            settings.CONTAINER_GROMACS, ["bash", "-c", md_cmd], timeout=settings.TIMEOUT_GROMACS)
        if retcode != 0:
            raise RuntimeError(f"MD production failed:\n{stderr[-2000:]}")
        if not os.path.exists(f"{job_dir}/md.xtc"):
            raise RuntimeError(f"MD production failed: md.xtc not generated.\n{_read_log_tail(f'{workdir}/md.log')}")

        # Basic analysis: RMSD
        task.progress = 95
        task.progress_msg = "Computing RMSD analysis..."
        await run_in_container(
            settings.CONTAINER_GROMACS,
            ["bash", "-c",
             f"echo '4 4' | gmx rms -s {workdir}/md.tpr -f {workdir}/md.xtc "
             f"-o {workdir}/rmsd.xvg -tu ns 2>&1"],
            timeout=300,
        )

        output_files = {
            "trajectory": f"{workdir}/md.xtc",
            "topology": f"{workdir}/md.tpr",
            "energy": f"{workdir}/md.edr",
            "rmsd": f"{workdir}/rmsd.xvg",
        }
        task.output_files = list(output_files.values())
        task.progress_msg = f"MD simulation complete. {req.sim_time_ns} ns trajectory ready."

        return {
            "output_files": output_files,
            "sim_time_ns": req.sim_time_ns,
            "forcefield": req.forcefield,
            "output_dir": workdir,
        }

    task = await task_manager.submit("gromacs", req.model_dump(), _run)
    return TaskRef(
        task_id=task.task_id,
        status=task.status,
        tool="gromacs",
        poll_url=f"/api/v1/tasks/{task.task_id}",
    )
