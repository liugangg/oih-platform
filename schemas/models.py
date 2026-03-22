"""
Unified Pydantic Schemas for all OIH tools
"""
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional
from enum import Enum


# ─── Shared ───────────────────────────────────────────────────────────────────

class TaskRef(BaseModel):
    task_id: str
    status: str
    tool: str
    poll_url: str


# ─── AlphaFold3 ───────────────────────────────────────────────────────────────

class AF3InputType(str, Enum):
    PROTEIN = "protein"
    RNA = "rna"
    DNA = "dna"
    LIGAND = "ligand"
    ION = "ion"

class AF3Modification(BaseModel):
    ptmType: str = Field(..., description="CCD code, e.g. SEP=phosphoSer, TPO=phosphoThr, PTR=phosphoTyr")
    ptmPosition: int = Field(..., description="1-based residue position in sequence")

class AF3Chain(BaseModel):
    type: AF3InputType
    sequence: Optional[str] = None       # protein / RNA / DNA
    smiles: Optional[str] = None         # ligand
    count: int = 1
    modifications: Optional[List["AF3Modification"]] = None  # PTMs

class AlphaFold3Request(BaseModel):
    job_name: str = Field(..., description="Job identifier")
    chains: List[AF3Chain] = Field(..., description="All chains/molecules to fold")
    num_seeds: int = Field(5, ge=1, le=20, description="Number of model seeds")
    run_relaxation: bool = True

class AlphaFold3Result(BaseModel):
    task_id: str
    pdb_files: List[str]
    confidence_scores: Dict[str, float]
    ranking_scores: List[float]


# ─── RFdiffusion ──────────────────────────────────────────────────────────────

class RFdiffusionMode(str, Enum):
    UNCONDITIONAL = "unconditional"
    MOTIF_SCAFFOLDING = "motif_scaffolding"
    PARTIAL_DIFFUSION = "partial_diffusion"
    BINDER_DESIGN = "binder_design"

class RFdiffusionRequest(BaseModel):
    job_name: str
    mode: RFdiffusionMode = RFdiffusionMode.BINDER_DESIGN
    target_pdb: Optional[str] = Field(None, description="Target PDB path (for binder design)")
    hotspot_residues: Optional[str] = Field(None, example="A30,A33,A34", description="Hotspot residues e.g. 'A30,A33'")
    motif_pdb: Optional[str] = Field(None, description="Motif PDB for scaffolding")
    contigs: Optional[str] = Field(None, example="A1-150/0 70-100", description="RFdiffusion contigs string")
    num_designs: int = Field(10, ge=1, le=100)
    num_diffusion_steps: int = Field(50, ge=10, le=200)

class RFdiffusionResult(BaseModel):
    task_id: str
    pdb_files: List[str]
    num_designs: int


# ─── ProteinMPNN ──────────────────────────────────────────────────────────────

class ProteinMPNNRequest(BaseModel):
    job_name: str
    input_pdb: str = Field(..., description="Path to input PDB file (backbone from RFdiffusion)")
    chains_to_design: str = Field("A", description="Chain IDs to redesign e.g. 'A' or 'A,B'")
    fixed_residues: Optional[str] = Field(None, description="Residue positions to keep fixed")
    num_sequences: int = Field(8, ge=1, le=100)
    sampling_temp: float = Field(0.1, ge=0.0, le=1.0, description="Sampling temperature")
    use_soluble_model: bool = False

class ProteinMPNNResult(BaseModel):
    task_id: str
    fasta_file: str
    sequences: List[Dict[str, Any]]


# ─── BindCraft ────────────────────────────────────────────────────────────────

class BindCraftRequest(BaseModel):
    job_name: str
    target_pdb: str = Field(..., description="Target protein PDB")
    target_hotspots: Optional[str] = Field(None, description="Hotspot residues e.g. 'A30,A33'")
    num_designs: int = Field(100, ge=10, le=1000)
    filters: Optional[Dict[str, Any]] = Field(None, description="BindCraft filter settings")
    advanced_settings: Optional[Dict[str, Any]] = None

class BindCraftResult(BaseModel):
    task_id: str
    final_designs_csv: str
    top_binders: List[Dict[str, Any]]
    pdb_files: List[str]


# ─── Pocket Analysis ──────────────────────────────────────────────────────────

class FpocketRequest(BaseModel):
    job_name: str
    input_pdb: str
    min_sphere_size: float = Field(3.0, description="Min alpha sphere radius")
    min_druggability_score: float = Field(0.0, ge=0.0, le=1.0)

class P2RankRequest(BaseModel):
    job_name: str
    input_pdb: str
    model: Literal["default", "alphafold", "conservation"] = "default"

class PocketResult(BaseModel):
    task_id: str
    tool: str
    pockets: List[Dict[str, Any]]
    output_dir: str


# ─── Molecular Docking ────────────────────────────────────────────────────────

class DockingEngine(str, Enum):
    VINA_GPU = "vina-gpu"
    AUTODOCK_GPU = "autodock-gpu"
    GNINA = "gnina"
    DIFFDOCK = "diffdock"

class DockingRequest(BaseModel):
    job_name: str
    engine: DockingEngine = DockingEngine.GNINA
    receptor_pdb: str = Field(..., description="Prepared receptor PDB/PDBQT")
    ligand: str = Field(..., description="Ligand SMILES string or file path")
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    center_z: Optional[float] = None
    box_size_x: float = 25.0
    box_size_y: float = 25.0
    box_size_z: float = 25.0
    num_poses: int = Field(9, ge=1, le=20)
    pocket_id: Optional[int] = Field(None, description="Auto-set box from pocket analysis result")
    exhaustiveness: int = Field(8, ge=1, le=64)

class DockingResult(BaseModel):
    task_id: str
    engine: str
    poses: List[Dict[str, Any]]   # [{pose_id, affinity_kcal_mol, rmsd, file}]
    best_affinity: float
    output_dir: str


# ─── DiffDock ────────────────────────────────────────────────────────────────

class DiffDockRequest(BaseModel):
    job_name: str
    receptor_pdb: str
    ligand_smiles: str = Field(..., description="Ligand SMILES string")
    num_poses: int = Field(10, ge=1, le=40)
    inference_steps: int = Field(20, ge=10, le=50)
    samples_per_complex: int = 10

class DiffDockResult(BaseModel):
    task_id: str
    ranked_poses: List[Dict[str, Any]]
    confidence_scores: List[float]


# ─── GROMACS MD ──────────────────────────────────────────────────────────────

class GromacsPreset(str, Enum):
    PROTEIN_WATER = "protein_water"
    PROTEIN_LIGAND = "protein_ligand"
    MEMBRANE_PROTEIN = "membrane_protein"

class GROMACSRequest(BaseModel):
    job_name: str
    input_pdb: str
    preset: GromacsPreset = GromacsPreset.PROTEIN_WATER
    forcefield: str = Field("amber99sb-ildn", description="Force field")
    water_model: str = Field("tip3p", description="Water model")
    box_padding_nm: float = Field(1.0, description="Box padding in nm")
    sim_time_ns: float = Field(10.0, ge=0.1, le=1000.0, description="Simulation time in ns")
    temperature_k: int = Field(300, description="Temperature in K")
    ligand_itp: Optional[str] = Field(None, description="Ligand topology file path")
    ligand_sdf: Optional[str] = Field(None, description="Ligand SDF file path (from docking output), triggers GAFF2 parameterization")
    gpu_id: str = Field("0", description="GPU ID for compute")

class GROMACSResult(BaseModel):
    task_id: str
    trajectory_file: str      # .xtc
    topology_file: str        # .tpr
    energy_file: str          # .edr
    analysis: Dict[str, Any]  # RMSD, Rg, energies


# ─── Full Pipeline ────────────────────────────────────────────────────────────

class DrugDiscoveryPipelineRequest(BaseModel):
    """One-shot: fetch target → fpocket → dock → AF3 validate → MD"""
    job_name: str
    # Target: pdb_id (RCSB fetch) OR target_pdb (local path)
    pdb_id: Optional[str] = Field(None, description="RCSB PDB ID to fetch (e.g. '5XWR')")
    target_pdb: Optional[str] = Field(None, description="Local PDB path (skip fetch_pdb step)")
    # Protein sequence for AF3 complex validation
    protein_sequence: Optional[str] = Field(None, description="Target protein sequence for AF3 complex prediction")
    # Ligand: ligand_name (PubChem lookup) OR ligand_smiles (direct)
    ligand_name: Optional[str] = Field(None, description="Molecule name for PubChem lookup (e.g. 'erlotinib')")
    ligand_smiles: Optional[str] = Field(None, description="Ligand SMILES (skip fetch_molecule step)")
    # Pipeline options
    run_af3_validation: bool = Field(False, description="Run AF3 complex prediction + RMSD comparison vs GNINA")
    run_md: bool = Field(False, description="Run GROMACS 10ns MD on selected best pose")
    rmsd_threshold_angstrom: float = Field(2.0, description="RMSD threshold: use AF3 complex for MD if RMSD<threshold, else GNINA pose")
    docking_engine: DockingEngine = DockingEngine.GNINA

class BinderDesignPipelineRequest(BaseModel):
    """One-shot: target → designed binders → sequences → ranked"""
    job_name: str = "binder_design"
    pdb_id: Optional[str] = Field(None, description="PDB ID to fetch from RCSB (e.g. '1N8Z')")
    target_pdb: Optional[str] = Field(None, description="Local PDB file path (alternative to pdb_id)")
    hotspot_residues: Optional[list[str]] = Field(None, description="Hotspot residues e.g. ['A30','A33']")
    num_designs: int = Field(10, ge=1, le=100, description="Alias for num_rfdiffusion_designs")
    num_rfdiffusion_designs: int = 50
    num_mpnn_sequences: int = 8
    run_af3_validation: bool = True
    dry_run: bool = Field(False, description="If True, return mock results without running tools")

# ── ESM ──────────────────────────────────────────────────────
class ESMTask(str, Enum):
    embedding   = "embedding"
    similarity  = "similarity"
    batch_embed = "batch_embed"

class ESMRequest(BaseModel):
    job_name: str = Field(..., description="任务名称")
    sequences: list[str] = Field(..., description="蛋白质序列列表")
    task: ESMTask = Field(ESMTask.embedding, description="任务类型")
    model_name: str = Field("esm2_t33_650M_UR50D", description="ESM2模型")
    repr_layer: int = Field(33, description="提取嵌入的层")

class ESMResult(BaseModel):
    job_name: str
    task: str
    mean_embeddings: list | None = None
    similarity_matrix: list | None = None
    output_dir: str


class ESMScoreRequest(BaseModel):
    job_name: str
    sequences: List[str] = Field(..., description="Protein sequences to score")
    model_name: str = Field("esm2_t33_650M_UR50D", description="ESM2 model for pseudo-perplexity")


class ESMMutantScanRequest(BaseModel):
    job_name: str
    sequence: str = Field(..., description="Wild-type protein sequence")
    scan_positions: Optional[List[int]] = Field(None, description="1-based positions to scan (default: all)")
    model_name: str = Field("esm1v_t33_650M_UR90S_1", description="ESM-1v model for mutation effect prediction")

# ── Chemprop ─────────────────────────────────────────────────
class ChempropTask(str, Enum):
    predict = "predict"
    train   = "train"

class ChempropRequest(BaseModel):
    job_name: str = Field(..., description="任务名称")
    task: ChempropTask = Field(ChempropTask.predict)
    smiles: list[str] | None = Field(None, description="SMILES列表")
    model_path: str | None = Field(None, description="模型路径")
    train_csv: str | None = Field(None, description="训练CSV路径")
    target_columns: list[str] | None = Field(None, description="目标列名")
    epochs: int = Field(30, description="训练轮数")
    property_name: str = Field("activity", description="属性名称")

class ChempropResult(BaseModel):
    job_name: str
    task: str
    predictions: list | None = None
    model_output_dir: str | None = None
    output_dir: str


# ─── FreeSASA ────────────────────────────────────────────────────────────────

class FreeSASARequest(BaseModel):
    job_name: str
    input_pdb: str = Field(..., description="Antibody PDB path (from RFdiffusion/ProteinMPNN output)")

class ConjugationSite(BaseModel):
    residue: str = Field(..., description="Residue ID e.g. K127")
    chain: str
    sasa: float = Field(..., description="SASA in Å²")

class FreeSASAResult(BaseModel):
    conjugation_sites: List[ConjugationSite]


# ─── Linker Select ───────────────────────────────────────────────────────────

class LinkerSelectRequest(BaseModel):
    payload_smiles: Optional[str] = Field(None, description="Payload molecule SMILES")
    linker_type: Optional[Literal["cleavable", "non_cleavable"]] = Field(None, description="Linker type (legacy, maps to cleavable filter)")
    cleavable: Optional[bool] = Field(None, description="Filter by cleavability")
    reaction_type: Optional[str] = Field(None, description="Filter by reaction type e.g. maleimide_thiol")
    compatible_payload: Optional[str] = Field(None, description="Fuzzy match against compatible_payloads e.g. MMAE, DM1")
    clinical_status: Optional[str] = Field(None, description="Filter: approved / clinical / research")
    max_results: int = Field(5, description="Max number of results")

class LinkerInfo(BaseModel):
    id: str
    name: str
    smiles: str
    reaction_type: str
    dar_range: str
    approved_adcs: List[str]
    stability_plasma: str
    cleavage_mechanism: Optional[str]
    notes: str

class LinkerSelectResult(BaseModel):
    recommended_linkers: List[LinkerInfo]
    total_matched: int
    recommendation: str = ""


# ─── RDKit Conjugate ─────────────────────────────────────────────────────────

class RDKitConjugateRequest(BaseModel):
    job_name: str
    antibody_pdb: str = Field(..., description="Antibody PDB path")
    conjugation_site: str = Field(..., description="e.g. K127")
    linker_smiles: str = Field(..., description="Linker SMILES from linker_select")
    payload_smiles: str = Field(..., description="Payload SMILES from fetch_molecule")
    linker_name: Optional[str] = Field(None, description="Linker name for reaction type lookup (e.g. VC-PABC, SMCC)")
    reaction_type: str = Field("auto", description="Reaction type or 'auto' to detect from functional groups")

class PocketGuidedBinderPipelineRequest(BaseModel):
    job_name: str = Field(..., description="Unique job identifier")
    pdb_id: str = Field(..., description="PDB ID to fetch from RCSB (e.g. '2A91')")
    target_name: Optional[str] = Field(None, description="Target protein name (e.g. 'HER2') for tier classification")
    antigen_chain: Optional[str] = Field(None, description="Antigen chain ID in the PDB (for tier 1 extraction)")
    binder_type: str = Field("de_novo", description="Binder type: 'de_novo' (RFdiffusion, skip IgFold), 'nanobody' or 'antibody' (enable IgFold filter)")
    chains: Optional[str] = Field(None, description="Chains to keep (e.g. 'A')")
    probe_smiles: str = Field("CC(=O)Oc1ccccc1C(=O)O", description="Small probe molecule SMILES for DiffDock cross-validation (default: aspirin)")
    num_rfdiffusion_designs: int = Field(10, description="Number of RFdiffusion backbone designs")
    num_mpnn_sequences: int = Field(8, description="MPNN sequences per backbone")
    binder_length: str = Field("70-120", description="Binder length range for RFdiffusion contig")
    run_diffdock_validation: bool = Field(True, description="Run DiffDock blind docking to cross-validate pockets")
    run_af3_validation: bool = Field(True, description="Run AF3 on top MPNN designs")
    dry_run: bool = Field(False, description="Return mock results without running tools")

# ─── DiscoTope3 ──────────────────────────────────────────────────────────────

# ─── IgFold ───────────────────────────────────────────────────────────────────

class IgFoldRequest(BaseModel):
    job_name: str
    sequences: Dict[str, str] = Field(..., description="Chain sequences dict, e.g. {'H': 'EVQLVE...'}. Nanobody = H only.")
    do_refine: bool = Field(False, description="OpenMM refinement (slower but better geometry)")


# ─── DiscoTope3 ──────────────────────────────────────────────────────────────

class DiscoTope3Request(BaseModel):
    job_name: str
    input_pdb: str = Field(..., description="Path to protein PDB file (antibody or antigen)")
    struc_type: str = Field("solved", description="Structure type: 'solved' (experimental) or 'alphafold' (predicted)")
    calibrated_score_epi_threshold: float = Field(0.90, description="Epitope threshold: low=0.40, moderate=0.90, higher=1.50")
    multichain_mode: bool = Field(False, description="Predict on entire complex (all chains)")
    cpu_only: bool = Field(False, description="Force CPU inference (default uses GPU)")


class RDKitConjugateResult(BaseModel):
    adc_smiles: str
    covalent: bool
    reaction_type_used: str
    reaction_chemistry: str
    detection_method: str
    dar_range: str
    stability_note: str
    linker_info: Dict[str, Any]
    output_sdf: str
    atom_count: int
    warnings: List[str]
