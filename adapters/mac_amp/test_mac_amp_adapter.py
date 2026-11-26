"""
Unit tests for the MAC-AMP Input Adapter (`adapters/mac_amp`).
Tests:
  1. Valid 6-field JSON produces the correct MAC-AMP configuration and Hydra YAML.
  2. Mapping `parameters.target_organism` -> `target_bacterium` and optional MAC-AMP parameters.
  3. Mapping `inputs.peptide_dataset_ref`, `inputs.model_ref`, and `inputs.workflow_config_ref`.
  4. Handling unsupported bacteria properly (rejected with `clarification_required`).
  5. Handling missing target organism (`examples/mac_amp_missing_target.json`) and unresolved requests.
  6. Handling missing dataset files, missing model checkpoints, and missing workflow configs.
  7. Rejecting raw unstructured natural language (no OpenAI API calls in the adapter).
"""

import json
import tempfile
import unittest
from pathlib import Path

try:
    from .schema import MacAmpConfig, SUPPORTED_BACTERIA, TranslationResult
    from .translator import MacAmpAdapter, SpecKitTranslator
except ImportError:
    from schema import MacAmpConfig, SUPPORTED_BACTERIA, TranslationResult
    from translator import MacAmpAdapter, SpecKitTranslator

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"


class TestMacAmpAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = SpecKitTranslator(verify_resources=False)

    def test_valid_mac_amp_request_example(self) -> None:
        """Valid JSON from examples/mac_amp_request.json produces expected MacAmpConfig and YAML."""
        example_path = EXAMPLES_DIR / "mac_amp_request.json"
        result = self.adapter.translate(example_path)

        self.assertEqual(result.status, "success")
        self.assertIsNotNone(result.config)
        assert result.config is not None
        self.assertEqual(result.config.target_bacterium, "Escherichia_coli")
        self.assertEqual(result.config.bacterium_code, "EC")
        self.assertEqual(
            result.config.warmup_dir,
            "../Generation_Module/data/warmup/warmup_AMP_Escherichia_coli.csv",
        )
        self.assertEqual(
            result.config.ref_pdb_path,
            "/data/AMP_Escherichia_coli_811_new/ref_pdbs",
        )
        self.assertEqual(result.config.model_path, "../Generation_Module/model_ckpt")
        self.assertEqual(result.config.real_epoch, 15)
        self.assertEqual(result.config.batch_size, 10)

        yaml_text = self.adapter.translate_for_downstream(example_path, save_to_specs=False)
        self.assertIn('target_bacterium: "Escherichia_coli"', yaml_text)
        self.assertIn('bacterium_code: "EC"', yaml_text)
        self.assertIn('model_path: "../Generation_Module/model_ckpt"', yaml_text)

    def test_unregistered_dataset_is_rejected(self) -> None:
        """An example upload ID must never be converted into a guessed filename."""
        result = self.adapter.translate(EXAMPLES_DIR / "mac_amp_with_dataset.json")
        self.assertEqual(result.status, "clarification_required")
        self.assertIsNone(result.config)
        self.assertIn("peptide_dataset_ref", result.missing_or_ambiguous_fields)

    def test_organism_abbreviations_and_all_supported_bacteria(self) -> None:
        """All 5 supported MAC-AMP bacteria and common abbreviations map to canonical names and codes."""
        cases = [
            ("E. coli", "Escherichia_coli", "EC"),
            ("Escherichia coli", "Escherichia_coli", "EC"),
            ("Staphylococcus aureus", "Staphylococcus_aureus", "SA"),
            ("S. aureus", "Staphylococcus_aureus", "SA"),
            ("MRSA", "Staphylococcus_aureus", "SA"),
            ("Pseudomonas aeruginosa", "Pseudomonas_aeruginosa", "PA"),
            ("P. aeruginosa", "Pseudomonas_aeruginosa", "PA"),
            ("Klebsiella pneumoniae", "Klebsiella_pneumoniae", "KP"),
            ("Enterococcus faecium", "Enterococcus_faecium", "EF"),
        ]
        for raw_org, expected_canonical, expected_code in cases:
            req = {
                "schema_version": "1.0",
                "algorithm_id": "mac_amp",
                "operation": "peptide_generation",
                "parameters": {"target_organism": raw_org},
                "inputs": {},
                "unresolved": [],
            }
            res = self.adapter.translate(req)
            self.assertEqual(res.status, "success", f"Failed for {raw_org}")
            assert res.config is not None
            self.assertEqual(res.config.target_bacterium, expected_canonical)
            self.assertEqual(res.config.bacterium_code, expected_code)

    def test_optional_mac_amp_hyperparameter_overrides(self) -> None:
        """Supported MAC-AMP parameters in JSON override Spec-Kit defaults."""
        req = {
            "schema_version": "1.0",
            "algorithm_id": "mac_amp",
            "operation": "peptide_generation",
            "parameters": {
                "target_organism": "Staphylococcus aureus",
                "real_epoch": 25,
                "sand_box_epoch": 8,
                "batch_size": 16,
                "num_samples": 64,
                "max_length": 40,
                "temperature": 0.7,
                "ablation_mode": 2,
                "experiment": "all_llama_local",
            },
            "inputs": {},
            "unresolved": [],
        }
        res = self.adapter.translate(req)
        self.assertEqual(res.status, "success")
        assert res.config is not None
        self.assertEqual(res.config.target_bacterium, "Staphylococcus_aureus")
        self.assertEqual(res.config.bacterium_code, "SA")
        self.assertEqual(res.config.real_epoch, 25)
        self.assertEqual(res.config.sand_box_epoch, 8)
        self.assertEqual(res.config.batch_size, 16)
        self.assertEqual(res.config.num_samples, 64)
        self.assertEqual(res.config.max_length, 40)
        self.assertEqual(res.config.temperature, 0.7)
        self.assertEqual(res.config.ablation_mode, 2)
        self.assertEqual(res.config.experiment.value, "all_llama_local")

    def test_unsupported_bacterium_rejected(self) -> None:
        """Unsupported bacteria are caught and rejected instead of silently guessed."""
        for bad_organism in ["Bacillus subtilis", "Salmonella enterica", "Mycobacterium tuberculosis"]:
            req = {
                "schema_version": "1.0",
                "algorithm_id": "mac_amp",
                "operation": "peptide_generation",
                "parameters": {"target_organism": bad_organism},
                "inputs": {},
                "unresolved": [],
            }
            res = self.adapter.translate(req)
            self.assertEqual(res.status, "clarification_required")
            self.assertIsNone(res.config)
            self.assertIn("target_organism", res.missing_or_ambiguous_fields)
            self.assertIn("Unsupported target bacterium", res.clarification_message or "")

            with self.assertRaises(ValueError):
                self.adapter.translate_for_downstream(req, save_to_specs=False)

    def test_missing_target_organism_rejected(self) -> None:
        """Missing target_organism (examples/mac_amp_missing_target.json) returns clarification_required."""
        missing_path = EXAMPLES_DIR / "mac_amp_missing_target.json"
        res = self.adapter.translate(missing_path)
        self.assertEqual(res.status, "clarification_required")
        self.assertIsNone(res.config)
        self.assertIn("target_organism", res.missing_or_ambiguous_fields)

    def test_resource_verification_existing_and_missing_files(self) -> None:
        """Adapter verifies that dataset paths, model files, and PDB directories actually exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Create valid dataset CSV, model directory, and ref_pdb directory
            dataset_file = root / "upload_peptides_001.csv"
            dataset_file.write_text("sequence,mic\nRRIRRPRLPRPRVPRPRI,1.2\n", encoding="utf-8")

            model_dir = root / "Generation_Module" / "model_ckpt"
            model_dir.mkdir(parents=True)
            (model_dir / "pytorch_model.bin").write_bytes(b"dummy")

            pdb_dir = root / "data" / "AMP_Escherichia_coli_811_new" / "ref_pdbs"
            pdb_dir.mkdir(parents=True)

            verified_adapter = SpecKitTranslator(
                resource_root=root,
                resource_registry={"upload_peptides_001": str(dataset_file)},
                verify_resources=True,
            )

            # 1. Valid request where all resources exist on disk
            example_with_ds = EXAMPLES_DIR / "mac_amp_with_dataset.json"
            ok_res = verified_adapter.translate(example_with_ds)
            self.assertEqual(ok_res.status, "success")
            assert ok_res.config is not None
            self.assertTrue(ok_res.config.warmup_dir.endswith("upload_peptides_001.csv"))

            # 2. Request referencing a non-existent dataset
            bad_ds_req = {
                "schema_version": "1.0",
                "algorithm_id": "mac_amp",
                "operation": "peptide_generation",
                "parameters": {"target_organism": "Escherichia coli"},
                "inputs": {"peptide_dataset_ref": "nonexistent_dataset_999"},
                "unresolved": [],
            }
            bad_ds_res = verified_adapter.translate(bad_ds_req)
            self.assertEqual(bad_ds_res.status, "clarification_required")
            self.assertTrue(
                any("peptide_dataset_ref" in field for field in bad_ds_res.missing_or_ambiguous_fields)
            )

            # 3. Request referencing a non-existent model_ref
            bad_model_req = {
                "schema_version": "1.0",
                "algorithm_id": "mac_amp",
                "operation": "peptide_generation",
                "parameters": {"target_organism": "Escherichia coli"},
                "inputs": {"model_ref": "missing_checkpoint_dir"},
                "unresolved": [],
            }
            bad_model_res = verified_adapter.translate(bad_model_req)
            self.assertEqual(bad_model_res.status, "clarification_required")
            self.assertTrue(
                any("model_ref" in field for field in bad_model_res.missing_or_ambiguous_fields)
            )

    def test_wrong_algorithm_or_unresolved_items_rejected(self) -> None:
        """Adapter rejects rl_diffusion requests or requests with unresolved items."""
        rl_path = EXAMPLES_DIR / "rl_diffusion_request.json"
        rl_res = self.adapter.translate(rl_path)
        self.assertEqual(rl_res.status, "clarification_required")
        self.assertIn("algorithm_id", rl_res.missing_or_ambiguous_fields)

        unresolved_req = {
            "schema_version": "1.0",
            "algorithm_id": "mac_amp",
            "operation": "peptide_generation",
            "parameters": {"target_organism": "Escherichia coli"},
            "inputs": {},
            "unresolved": ["conflicting_organism_request"],
        }
        unres = self.adapter.translate(unresolved_req)
        self.assertEqual(unres.status, "clarification_required")
        self.assertIn("conflicting_organism_request", unres.missing_or_ambiguous_fields)

    def test_raw_natural_language_rejected(self) -> None:
        """Adapter rejects unstructured natural language strings instead of calling OpenAI."""
        with self.assertRaises(ValueError):
            self.adapter.translate("Configure a run targeting Escherichia coli")


if __name__ == "__main__":
    unittest.main()
