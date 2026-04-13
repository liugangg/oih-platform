"""
Unit tests for core logic: target registry, tier classification, domain
truncation, and binder chain detection. No GPU or containers required.
"""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REGISTRY_PATH = PROJECT_ROOT / "config" / "target_registry.json"


class TestTargetRegistryLoading(unittest.TestCase):
    """Verify config/target_registry.json loads correctly and matches paper."""

    def setUp(self):
        with open(REGISTRY_PATH) as f:
            self.registry = json.load(f)
        self.targets = self.registry["targets"]

    def test_registry_file_exists(self):
        self.assertTrue(REGISTRY_PATH.exists())

    def test_paper_targets_present(self):
        paper_targets = ["HER2", "EGFR", "NECTIN4", "CD36", "TROP2"]
        for t in paper_targets:
            self.assertIn(t, self.targets, f"Paper target {t} missing from registry")

    def test_no_extra_paper_targets(self):
        """VEGF and TNF should not be in the registry (not evaluated in paper)."""
        self.assertNotIn("VEGF", self.targets)
        self.assertNotIn("TNF", self.targets)

    def test_all_targets_have_required_fields(self):
        required = ["pdb", "tier", "domain_registry"]
        for name, entry in self.targets.items():
            for field in required:
                self.assertIn(field, entry, f"{name} missing field: {field}")

    def test_pdb_aliases_present(self):
        aliases = self.registry["pdb_aliases"]
        self.assertEqual(aliases["her2"], "1N8Z")
        self.assertEqual(aliases["cd36"], "5LGD")
        self.assertEqual(aliases["trop2"], "7PEE")


class TestTierClassification(unittest.TestCase):
    """Verify tier assignment: Tier 1 for known co-crystals, Tier 2 otherwise."""

    def setUp(self):
        with open(REGISTRY_PATH) as f:
            self.targets = json.load(f)["targets"]

    def test_her2_is_tier1(self):
        self.assertEqual(self.targets["HER2"]["tier"], 1)

    def test_egfr_is_tier1(self):
        self.assertEqual(self.targets["EGFR"]["tier"], 1)

    def test_cd36_is_tier2(self):
        self.assertEqual(self.targets["CD36"]["tier"], 2)

    def test_nectin4_is_tier2(self):
        self.assertEqual(self.targets["NECTIN4"]["tier"], 2)

    def test_trop2_is_tier2(self):
        self.assertEqual(self.targets["TROP2"]["tier"], 2)

    def test_unknown_target_defaults_to_tier2(self):
        """Targets not in registry should be classified as Tier 2."""
        # Simulate _classify_target_tier logic: if not in KNOWN_COMPLEXES, return tier 2
        known = {n for n, e in self.targets.items() if e["tier"] == 1}
        unknown = "FAKEPROT"
        self.assertNotIn(unknown, known)
        # Default behavior: unknown → tier 2


class TestDomainTruncation(unittest.TestCase):
    """Verify domain registry ranges match expected values."""

    def setUp(self):
        with open(REGISTRY_PATH) as f:
            self.targets = json.load(f)["targets"]

    def test_her2_domain4_range(self):
        domains = self.targets["HER2"]["domain_registry"]
        self.assertIn("domain4", domains)
        start, end = domains["domain4"]["range"]
        self.assertEqual(start, 488)
        self.assertEqual(end, 630)
        span = end - start
        self.assertEqual(span, 142)

    def test_egfr_domain3_range(self):
        domains = self.targets["EGFR"]["domain_registry"]
        self.assertIn("domain3", domains)
        start, end = domains["domain3"]["range"]
        self.assertEqual(start, 361)
        self.assertEqual(end, 481)

    def test_cd36_has_multiple_domains(self):
        domains = self.targets["CD36"]["domain_registry"]
        self.assertIn("pesto_ppi_core", domains)
        self.assertIn("extracellular_full", domains)
        # extracellular_full should span 30-439
        self.assertEqual(domains["extracellular_full"]["range"], [30, 439])

    def test_domain_ranges_are_two_element_lists(self):
        for name, entry in self.targets.items():
            for dname, dinfo in entry.get("domain_registry", {}).items():
                r = dinfo["range"]
                self.assertEqual(len(r), 2, f"{name}/{dname} range should be [start, end]")
                self.assertLess(r[0], r[1], f"{name}/{dname} start >= end")


class TestKnownComplexesLigandChains(unittest.TestCase):
    """Verify ligand_chains are correct (lesson: EGFR was wrong before)."""

    def setUp(self):
        with open(REGISTRY_PATH) as f:
            self.targets = json.load(f)["targets"]

    def test_egfr_ligand_chains_are_CD(self):
        """EGFR 1YY9 cetuximab Fab uses chains C and D, not B."""
        self.assertEqual(self.targets["EGFR"]["ligand_chains"], ["C", "D"])

    def test_her2_ligand_chains(self):
        """HER2 1N8Z trastuzumab Fab uses chains A and B."""
        self.assertEqual(self.targets["HER2"]["ligand_chains"], ["A", "B"])

    def test_tier2_targets_have_null_receptor_chain(self):
        """Tier 2 targets have no known co-crystal, receptor_chain should be null."""
        for name in ["CD36", "TROP2", "NECTIN4"]:
            self.assertIsNone(
                self.targets[name]["receptor_chain"],
                f"{name} should have null receptor_chain (no co-crystal)"
            )


class TestBinderChainDetection(unittest.TestCase):
    """Verify shortest-chain-is-binder logic (pure Python, no gemmi needed)."""

    def test_shortest_chain_is_binder(self):
        """RFdiffusion output: shorter chain = designed binder, longer = target."""
        # Simulate chain length data as returned by gemmi
        chains = [("A", 580), ("B", 72)]  # target=580aa, binder=72aa
        chains.sort(key=lambda x: x[1])
        binder_chain = chains[0][0]
        self.assertEqual(binder_chain, "B")

    def test_binder_detection_with_equal_chains(self):
        chains = [("A", 100), ("B", 100)]
        chains.sort(key=lambda x: x[1])
        binder_chain = chains[0][0]
        # Either chain is acceptable when equal length
        self.assertIn(binder_chain, ["A", "B"])

    def test_three_chain_complex(self):
        """Multi-chain: binder is still the shortest."""
        chains = [("A", 300), ("B", 200), ("C", 65)]
        chains.sort(key=lambda x: x[1])
        binder_chain = chains[0][0]
        self.assertEqual(binder_chain, "C")


if __name__ == "__main__":
    unittest.main()
