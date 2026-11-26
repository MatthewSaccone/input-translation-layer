"""End-to-end tests with deterministic providers; no network calls."""
from core.natural_language_pipeline import prepare_from_prompt
from llm.interpreter import ExtractedIntent, MacAmpSettings


class FakeProvider:
    def __init__(self, **fields):
        self.fields = fields

    def extract(self, prompt, specs):
        return ExtractedIntent(
            algorithm_id=self.fields.get("algorithm_id", "mac_amp"),
            operation=self.fields.get("operation", "peptide_generation"),
            target_organism=self.fields.get("target_organism", "E. coli"),
            n_samples=self.fields.get("n_samples"),
            save_to_xyz=None, conditioning=None, qed_direction=None,
            sas_direction=None, affinity_direction=None,
            ambiguities=self.fields.get("ambiguities", []),
            mac_amp_settings=self.fields.get("mac_amp_settings"),
        )


def test_full_mac_amp_preview_with_tuning():
    result = prepare_from_prompt(
        "Generate peptides targeting E. coli, batch size 16",
        provider=FakeProvider(mac_amp_settings=MacAmpSettings(batch_size=16)),
    )
    assert result.pipeline.status == "config_preview_unverified"
    assert result.request["parameters"]["batch_size"] == 16
    assert "batch_size: 16" in result.pipeline.yaml


def test_missing_organism_is_invalid():
    result = prepare_from_prompt("Generate peptides", provider=FakeProvider(target_organism=None))
    assert result.pipeline.status == "invalid_request"
    assert result.pipeline.yaml is None


def test_ambiguous_request_stops_before_adapter():
    result = prepare_from_prompt(
        "Generate some candidates",
        provider=FakeProvider(algorithm_id=None, operation=None, target_organism=None,
                              ambiguities=["unknown candidate type"]),
    )
    assert result.pipeline.status == "invalid_request"
    assert result.pipeline.yaml is None


def test_rl_diffusion_review_preview_is_not_execution_ready():
    result = prepare_from_prompt(
        "Generate 20 3D molecules",
        provider=FakeProvider(algorithm_id="rl_diffusion", operation="sample", target_organism=None, n_samples=20),
        trusted_inputs={"model_ref": "registered_model_001"},
    )
    assert result.pipeline.status == "review_preview_only"
    assert result.pipeline.argument_preview["execution_ready"] is False
    assert result.pipeline.argument_preview["resource_refs"] == {"model_ref": "registered_model_001"}
    assert result.pipeline.yaml is None


def test_invalid_mac_amp_tuning_is_blocked():
    result = prepare_from_prompt("Peptides", provider=FakeProvider(mac_amp_settings=MacAmpSettings(batch_size=-1)))
    assert result.pipeline.status == "invalid_request"


def test_no_trusted_model_reference_is_invented():
    result = prepare_from_prompt("Peptides E. coli", provider=FakeProvider())
    assert result.request["inputs"] == {}
