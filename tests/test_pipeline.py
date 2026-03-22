"""
Unit tests for drug_discovery_pipeline helper functions.

Tests cover pure functions only (no containers, no network).
Run: pytest tests/test_pipeline.py -v
"""
import os
import tempfile
import pytest
import numpy as np
from routers.pipeline import (
    _pocket_to_box,
    _select_gnina_poses,
    _compute_rmsd_af3_gnina,
    _extract_ligand_heavy_atoms_from_cif,
    _extract_ligand_heavy_atoms_from_sdf,
    _compute_optimal_atom_matching,
    convert_af3_cif_to_pdb,
    _write_af3_ligand_as_sdf,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"

# Minimal AF3-style mmCIF with one protein residue + one ligand (3 heavy atoms)
MINIMAL_LIGAND_CIF = """\
data_test
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM   1  C  CA  .  ALA  A  1  1  ?   1.000  2.000  3.000  1.0  10.0  1  ALA  A  CA  1
HETATM 2  C  C1  .  LIG  B  2  .  ?  10.000 20.000 30.000  1.0  10.0  1  LIG  B  C1  1
HETATM 3  O  O1  .  LIG  B  2  .  ?  11.000 21.000 31.000  1.0  10.0  1  LIG  B  O1  1
HETATM 4  N  N1  .  LIG  B  2  .  ?  12.000 22.000 32.000  1.0  10.0  1  LIG  B  N1  1
#
loop_
_entity.id
_entity.type
1 polymer
2 non-polymer
#
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
B 2
#
"""


@pytest.fixture
def tmp_cif_with_ligand(tmp_path):
    """Write a minimal CIF that contains a non-polymer ligand chain."""
    cif_path = tmp_path / "test_ligand.cif"
    cif_path.write_text(MINIMAL_LIGAND_CIF)
    return str(cif_path)


@pytest.fixture
def tmp_cif_protein_only(tmp_path):
    """CIF with only a polymer chain — no ligand."""
    content = MINIMAL_LIGAND_CIF.replace(
        "HETATM 2  C  C1  .  LIG  B  2  .  ?  10.000 20.000 30.000  1.0  10.0  1  LIG  B  C1  1\n"
        "HETATM 3  O  O1  .  LIG  B  2  .  ?  11.000 21.000 31.000  1.0  10.0  1  LIG  B  O1  1\n"
        "HETATM 4  N  N1  .  LIG  B  2  .  ?  12.000 22.000 32.000  1.0  10.0  1  LIG  B  N1  1\n",
        "",
    ).replace(
        "2 non-polymer\n",
        "",
    ).replace(
        "B 2\n",
        "",
    )
    p = tmp_path / "protein_only.cif"
    p.write_text(content)
    return str(p)


@pytest.fixture
def tmp_sdf_3atoms(tmp_path):
    """SDF with a simple 3-heavy-atom molecule: C, O, N."""
    # Minimal V2000 SDF with atoms at the same position as MINIMAL_LIGAND_CIF
    sdf_content = """\

     RDKit          3D

  3  2  0  0  0  0  0  0  0  0999 V2000
   10.0000   20.0000   30.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   11.0000   21.0000   31.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   12.0000   22.0000   32.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
M  END
$$$$
"""
    p = tmp_path / "test.sdf"
    p.write_text(sdf_content)
    return str(p)


@pytest.fixture
def tmp_sdf_3atoms_shifted(tmp_path):
    """Same topology as tmp_sdf_3atoms but atoms shifted by 1 Å in x."""
    sdf_content = """\

     RDKit          3D

  3  2  0  0  0  0  0  0  0  0999 V2000
   11.0000   20.0000   30.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   12.0000   21.0000   31.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   13.0000   22.0000   32.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
M  END
$$$$
"""
    p = tmp_path / "shifted.sdf"
    p.write_text(sdf_content)
    return str(p)


# ─── _pocket_to_box ───────────────────────────────────────────────────────────

class TestPocketToBox:
    def test_basic_coordinates(self):
        pocket = {
            "x_bary_(sph)": "10.5",
            "y_bary_(sph)": "20.3",
            "z_bary_(sph)": "30.1",
            "real_volume": "8000.0",
        }
        cx, cy, cz, box = _pocket_to_box(pocket)
        assert cx == 10.5
        assert cy == 20.3
        assert cz == 30.1
        # cbrt(8000)=20, *1.2=24
        assert abs(box - 24.0) < 0.1

    def test_missing_keys_use_defaults(self):
        cx, cy, cz, box = _pocket_to_box({})
        assert cx == 0.0
        assert cy == 0.0
        assert cz == 0.0
        # default volume=15625 → cbrt=25, *1.2=30
        assert abs(box - 30.0) < 0.1

    def test_box_clamped_to_minimum(self):
        _, _, _, box = _pocket_to_box({"real_volume": "1.0"})
        assert box == 15.0

    def test_box_clamped_to_maximum(self):
        _, _, _, box = _pocket_to_box({"real_volume": "1e12"})
        assert box == 40.0

    def test_float_inputs(self):
        pocket = {"x_bary_(sph)": 5.0, "y_bary_(sph)": 10.0,
                  "z_bary_(sph)": 15.0, "real_volume": 1000.0}
        cx, cy, cz, box = _pocket_to_box(pocket)
        assert cx == 5.0
        assert cy == 10.0
        assert cz == 15.0
        assert box == 15.0   # cbrt(1000)*1.2=12 → clamped

    def test_returns_four_values(self):
        assert len(_pocket_to_box({})) == 4


# ─── _select_gnina_poses ──────────────────────────────────────────────────────

class TestSelectGninaPoses:
    def test_empty_returns_none_none(self):
        assert _select_gnina_poses([]) == (None, None)

    def test_single_pose_no_cnn_data(self):
        poses = [{"pose_id": 1, "affinity_kcal_mol": -5.5}]
        best_cnn, best_aff = _select_gnina_poses(poses)
        assert best_cnn["pose_id"] == 1
        assert best_aff["pose_id"] == 1

    def test_selects_most_negative_affinity(self):
        poses = [
            {"pose_id": 1, "minimizedAffinity": -5.0, "affinity_kcal_mol": -5.0},
            {"pose_id": 2, "minimizedAffinity": -7.5, "affinity_kcal_mol": -7.5},
            {"pose_id": 3, "minimizedAffinity": -6.0, "affinity_kcal_mol": -6.0},
        ]
        _, best_aff = _select_gnina_poses(poses)
        assert best_aff["pose_id"] == 2

    def test_selects_highest_cnnscore(self):
        poses = [
            {"pose_id": 1, "affinity_kcal_mol": -7.0, "CNNscore": 0.3},
            {"pose_id": 2, "affinity_kcal_mol": -5.0, "CNNscore": 0.9},
            {"pose_id": 3, "affinity_kcal_mol": -6.0, "CNNscore": 0.5},
        ]
        best_cnn, _ = _select_gnina_poses(poses)
        assert best_cnn["pose_id"] == 2

    def test_best_cnn_and_best_affinity_can_differ(self):
        poses = [
            {"pose_id": 1, "minimizedAffinity": -8.0, "affinity_kcal_mol": -8.0, "CNNscore": 0.3},
            {"pose_id": 2, "minimizedAffinity": -5.0, "affinity_kcal_mol": -5.0, "CNNscore": 0.95},
        ]
        best_cnn, best_aff = _select_gnina_poses(poses)
        assert best_cnn["pose_id"] == 2
        assert best_aff["pose_id"] == 1

    def test_falls_back_to_pose1_when_no_cnnscore(self):
        poses = [{"pose_id": 1, "affinity_kcal_mol": -5.0},
                 {"pose_id": 2, "affinity_kcal_mol": -7.0}]
        best_cnn, _ = _select_gnina_poses(poses)
        assert best_cnn["pose_id"] == 1

    def test_uses_minimizedaffinity_when_present(self):
        poses = [
            {"pose_id": 1, "minimizedAffinity": -3.0, "affinity_kcal_mol": -8.0},
            {"pose_id": 2, "minimizedAffinity": -9.0, "affinity_kcal_mol": -2.0},
        ]
        _, best_aff = _select_gnina_poses(poses)
        assert best_aff["pose_id"] == 2


# ─── _extract_ligand_heavy_atoms_from_cif ────────────────────────────────────

class TestExtractLigandFromCif:
    def test_extracts_three_atoms(self, tmp_cif_with_ligand):
        atoms = _extract_ligand_heavy_atoms_from_cif(tmp_cif_with_ligand)
        assert len(atoms) == 3

    def test_correct_elements(self, tmp_cif_with_ligand):
        atoms = _extract_ligand_heavy_atoms_from_cif(tmp_cif_with_ligand)
        elems = {a[0] for a in atoms}
        assert elems == {"C", "O", "N"}

    def test_correct_coordinates(self, tmp_cif_with_ligand):
        atoms = _extract_ligand_heavy_atoms_from_cif(tmp_cif_with_ligand)
        # C atom should be at (10, 20, 30)
        c_atom = next(a for a in atoms if a[0] == "C")
        assert abs(c_atom[1] - 10.0) < 0.01
        assert abs(c_atom[2] - 20.0) < 0.01
        assert abs(c_atom[3] - 30.0) < 0.01

    def test_protein_only_returns_empty(self, tmp_cif_protein_only):
        atoms = _extract_ligand_heavy_atoms_from_cif(tmp_cif_protein_only)
        assert atoms == []

    def test_missing_file_raises(self):
        with pytest.raises(Exception):
            _extract_ligand_heavy_atoms_from_cif("/nonexistent/path.cif")


# ─── _extract_ligand_heavy_atoms_from_sdf ────────────────────────────────────

class TestExtractLigandFromSdf:
    def test_extracts_three_atoms(self, tmp_sdf_3atoms):
        atoms = _extract_ligand_heavy_atoms_from_sdf(tmp_sdf_3atoms)
        assert len(atoms) == 3

    def test_correct_elements(self, tmp_sdf_3atoms):
        atoms = _extract_ligand_heavy_atoms_from_sdf(tmp_sdf_3atoms)
        elems = {a[0] for a in atoms}
        assert elems == {"C", "O", "N"}

    def test_correct_coordinates(self, tmp_sdf_3atoms):
        atoms = _extract_ligand_heavy_atoms_from_sdf(tmp_sdf_3atoms)
        c_atom = next(a for a in atoms if a[0] == "C")
        assert abs(c_atom[1] - 10.0) < 0.01
        assert abs(c_atom[2] - 20.0) < 0.01
        assert abs(c_atom[3] - 30.0) < 0.01

    def test_missing_file_returns_empty(self):
        atoms = _extract_ligand_heavy_atoms_from_sdf("/nonexistent.sdf")
        assert atoms == []


# ─── _compute_optimal_atom_matching ──────────────────────────────────────────

class TestComputeOptimalAtomMatching:
    def _atoms(self, elems_xyzs):
        """Helper: [(elem, x, y, z), ...]"""
        return [(e, x, y, z) for e, x, y, z in elems_xyzs]

    def test_identical_atoms_zero_rmsd(self):
        atoms = [("C", 1.0, 2.0, 3.0), ("O", 4.0, 5.0, 6.0)]
        rmsd, row, col = _compute_optimal_atom_matching(atoms, atoms)
        assert rmsd == pytest.approx(0.0, abs=1e-9)
        assert row is not None
        assert col is not None

    def test_uniform_shift_gives_correct_rmsd(self):
        atoms_a = [("C", 0.0, 0.0, 0.0), ("O", 3.0, 0.0, 0.0)]
        atoms_b = [("C", 1.0, 0.0, 0.0), ("O", 4.0, 0.0, 0.0)]
        rmsd, _, _ = _compute_optimal_atom_matching(atoms_a, atoms_b)
        assert rmsd == pytest.approx(1.0, abs=1e-6)

    def test_different_counts_return_inf(self):
        a = [("C", 0.0, 0.0, 0.0)]
        b = [("C", 0.0, 0.0, 0.0), ("O", 1.0, 0.0, 0.0)]
        rmsd, row, col = _compute_optimal_atom_matching(a, b)
        assert rmsd == float("inf")
        assert row is None

    def test_element_mismatch_returns_inf(self):
        a = [("C", 0.0, 0.0, 0.0), ("N", 1.0, 0.0, 0.0)]
        b = [("C", 0.0, 0.0, 0.0), ("O", 1.0, 0.0, 0.0)]
        rmsd, _, _ = _compute_optimal_atom_matching(a, b)
        assert rmsd == float("inf")

    def test_reordered_atoms_matched_optimally(self):
        """If atoms are in different order, matching should still give low RMSD."""
        atoms_a = [("C", 0.0, 0.0, 0.0), ("O", 5.0, 0.0, 0.0), ("N", 10.0, 0.0, 0.0)]
        # B has same atoms but in reverse order
        atoms_b = [("N", 10.0, 0.0, 0.0), ("O", 5.0, 0.0, 0.0), ("C", 0.0, 0.0, 0.0)]
        rmsd, row, col = _compute_optimal_atom_matching(atoms_a, atoms_b)
        assert rmsd == pytest.approx(0.0, abs=1e-6)
        assert row is not None

    def test_returns_correct_mapping_length(self):
        atoms = [("C", i, 0.0, 0.0) for i in range(4)]
        rmsd, row, col = _compute_optimal_atom_matching(atoms, atoms)
        assert len(row) == 4
        assert len(col) == 4

    def test_mapping_is_bijection(self):
        atoms = [("C", 0.0, 0.0, 0.0), ("O", 1.0, 0.0, 0.0), ("N", 2.0, 0.0, 0.0)]
        _, row, col = _compute_optimal_atom_matching(atoms, atoms)
        assert sorted(row) == [0, 1, 2]
        assert sorted(col) == [0, 1, 2]


# ─── convert_af3_cif_to_pdb ──────────────────────────────────────────────────

class TestConvertAf3CifToPdb:
    def test_creates_pdb_file(self, tmp_cif_with_ligand, tmp_path):
        pdb_out = str(tmp_path / "out.pdb")
        result = convert_af3_cif_to_pdb(tmp_cif_with_ligand, pdb_out, protein_only=False)
        assert result == pdb_out
        assert os.path.exists(pdb_out)

    def test_pdb_not_empty(self, tmp_cif_with_ligand, tmp_path):
        pdb_out = str(tmp_path / "out.pdb")
        convert_af3_cif_to_pdb(tmp_cif_with_ligand, pdb_out, protein_only=False)
        assert os.path.getsize(pdb_out) > 0

    def test_protein_only_excludes_hetatm_chain(self, tmp_cif_with_ligand, tmp_path):
        pdb_out = str(tmp_path / "protein_only.pdb")
        convert_af3_cif_to_pdb(tmp_cif_with_ligand, pdb_out, protein_only=True)
        content = open(pdb_out).read()
        # Ligand chain B coords (10, 20, 30) should not appear
        assert "LIG" not in content

    def test_full_complex_includes_hetatm(self, tmp_cif_with_ligand, tmp_path):
        pdb_out = str(tmp_path / "full.pdb")
        convert_af3_cif_to_pdb(tmp_cif_with_ligand, pdb_out, protein_only=False)
        content = open(pdb_out).read()
        assert "ATOM" in content or "HETATM" in content

    def test_missing_cif_raises_runtime_error(self, tmp_path):
        with pytest.raises(Exception):
            convert_af3_cif_to_pdb("/nonexistent.cif", str(tmp_path / "out.pdb"))


# ─── _write_af3_ligand_as_sdf ─────────────────────────────────────────────────

class TestWriteAf3LigandAsSdf:
    def test_writes_sdf_with_af3_coords(self, tmp_cif_with_ligand, tmp_sdf_3atoms, tmp_path):
        from rdkit import Chem
        af3_atoms = _extract_ligand_heavy_atoms_from_cif(tmp_cif_with_ligand)
        # Identity mapping: af3_atoms and sdf atoms have the same order by element (sorted)
        # Use _compute_optimal_atom_matching to get the right mapping
        gnina_atoms = _extract_ligand_heavy_atoms_from_sdf(tmp_sdf_3atoms)
        _, row_ind, col_ind = _compute_optimal_atom_matching(af3_atoms, gnina_atoms)

        out_sdf = str(tmp_path / "af3_ligand.sdf")
        ok = _write_af3_ligand_as_sdf(af3_atoms, row_ind, col_ind, tmp_sdf_3atoms, out_sdf)
        assert ok
        assert os.path.exists(out_sdf)

        mol = next(Chem.SDMolSupplier(out_sdf, removeHs=True))
        assert mol is not None
        assert mol.GetNumAtoms() == 3

    def test_output_coords_match_af3(self, tmp_cif_with_ligand, tmp_sdf_3atoms, tmp_path):
        """AF3 positions should appear verbatim in the output SDF."""
        from rdkit import Chem
        af3_atoms = _extract_ligand_heavy_atoms_from_cif(tmp_cif_with_ligand)
        gnina_atoms = _extract_ligand_heavy_atoms_from_sdf(tmp_sdf_3atoms)
        _, row_ind, col_ind = _compute_optimal_atom_matching(af3_atoms, gnina_atoms)

        out_sdf = str(tmp_path / "af3_ligand.sdf")
        _write_af3_ligand_as_sdf(af3_atoms, row_ind, col_ind, tmp_sdf_3atoms, out_sdf)

        out_atoms = _extract_ligand_heavy_atoms_from_sdf(out_sdf)
        af3_xyz = sorted((a[1], a[2], a[3]) for a in af3_atoms)
        out_xyz = sorted((a[1], a[2], a[3]) for a in out_atoms)
        for (ax, ay, az), (ox, oy, oz) in zip(af3_xyz, out_xyz):
            assert abs(ax - ox) < 0.01
            assert abs(ay - oy) < 0.01
            assert abs(az - oz) < 0.01


# ─── _compute_rmsd_af3_gnina (integration of all steps) ─────────────────────

@pytest.mark.asyncio
async def test_rmsd_zero_when_coords_identical(tmp_cif_with_ligand, tmp_sdf_3atoms):
    """Identical CIF and SDF coords → RMSD ≈ 0."""
    rmsd, row, col = await _compute_rmsd_af3_gnina(
        tmp_cif_with_ligand, tmp_sdf_3atoms, "fake_receptor.pdb"
    )
    assert rmsd == pytest.approx(0.0, abs=1e-6)
    assert row is not None
    assert col is not None


@pytest.mark.asyncio
async def test_rmsd_nonzero_when_shifted(tmp_cif_with_ligand, tmp_sdf_3atoms_shifted):
    """SDF shifted 1 Å in x → RMSD should equal 1.0 Å."""
    rmsd, row, col = await _compute_rmsd_af3_gnina(
        tmp_cif_with_ligand, tmp_sdf_3atoms_shifted, "fake.pdb"
    )
    assert rmsd == pytest.approx(1.0, abs=1e-5)
    assert row is not None


@pytest.mark.asyncio
async def test_rmsd_inf_on_protein_only_cif(tmp_cif_protein_only, tmp_sdf_3atoms):
    """CIF with no ligand → inf."""
    rmsd, row, col = await _compute_rmsd_af3_gnina(
        tmp_cif_protein_only, tmp_sdf_3atoms, "fake.pdb"
    )
    assert rmsd == float("inf")
    assert row is None


@pytest.mark.asyncio
async def test_rmsd_inf_on_missing_sdf(tmp_cif_with_ligand):
    """Missing SDF → inf."""
    rmsd, _, _ = await _compute_rmsd_af3_gnina(
        tmp_cif_with_ligand, "/nonexistent.sdf", "fake.pdb"
    )
    assert rmsd == float("inf")


# ─── Schema validation ────────────────────────────────────────────────────────

class TestDrugDiscoveryPipelineSchema:
    def test_pdb_id_field(self):
        from schemas.models import DrugDiscoveryPipelineRequest
        req = DrugDiscoveryPipelineRequest(
            job_name="test", pdb_id="5XWR",
            ligand_smiles="CC(=O)Oc1ccccc1C(=O)O",
        )
        assert req.pdb_id == "5XWR"
        assert req.target_pdb is None
        assert req.run_af3_validation is False
        assert req.run_md is False

    def test_ligand_name_field(self):
        from schemas.models import DrugDiscoveryPipelineRequest
        req = DrugDiscoveryPipelineRequest(
            job_name="test",
            target_pdb="/data/oih/outputs/fetch_pdb/5XWR.pdb",
            ligand_name="aspirin",
        )
        assert req.ligand_name == "aspirin"
        assert req.ligand_smiles is None

    def test_rmsd_threshold_default(self):
        from schemas.models import DrugDiscoveryPipelineRequest
        req = DrugDiscoveryPipelineRequest(
            job_name="test", target_pdb="/tmp/test.pdb", ligand_smiles="CC(=O)O",
        )
        assert req.rmsd_threshold_angstrom == 2.0

    def test_full_pipeline_flags(self):
        from schemas.models import DrugDiscoveryPipelineRequest
        req = DrugDiscoveryPipelineRequest(
            job_name="egfr_erlotinib",
            pdb_id="1IVO",
            ligand_name="erlotinib",
            protein_sequence="MKTIIALSYIFCLVFA",
            run_af3_validation=True,
            run_md=True,
            rmsd_threshold_angstrom=1.5,
        )
        assert req.pdb_id == "1IVO"
        assert req.ligand_name == "erlotinib"
        assert req.run_md is True
        assert req.rmsd_threshold_angstrom == 1.5
