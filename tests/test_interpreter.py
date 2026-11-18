from __future__ import annotations

import pytest

from llm.interpreter import ExtractedIntent, interpret_user_prompt, to_translation_request


def intent(**updates):
    defaults = dict(
        algorithm_id="rl_diffusion", operation="sample", target_organism=None,
        n_samples=None, save_to_xyz=None, conditioning=None,
        qed_direction=None, sas_direction=None, affinity_direction=None,
        ambiguities=[],
    )
    return ExtractedIntent(**(defaults | updates))


class FakeProvider:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def extract(self, prompt, specs):
        self.calls += 1
        return self.value


def test_rl_diffusion_sample():
    result = to_translation_request(intent(n_samples=50, save_to_xyz=True))
    assert result == {
        "schema_version": "1.0", "algorithm_id": "rl_diffusion", "operation": "sample",
        "parameters": {"n_samples": 50, "save_to_xyz": True},
        "inputs": {}, "unresolved": [],
    }


def test_mac_amp_basic():
    value = intent(algorithm_id="mac_amp", operation="peptide_generation", target_organism="Escherichia coli")
    assert to_translation_request(value)["parameters"] == {"target_organism": "Escherichia coli"}


def test_mac_amp_uploaded_data_only_from_trusted_context():
    value = intent(algorithm_id="mac_amp", operation="peptide_generation", target_organism="Escherichia coli")
    result = to_translation_request(value, trusted_inputs={"peptide_dataset_ref": "upload_peptides_001", "bad": "bad"})
    assert result["inputs"] == {"peptide_dataset_ref": "upload_peptides_001"}


def test_mac_amp_missing_target_not_invented():
    value = intent(algorithm_id="mac_amp", operation="peptide_generation")
    assert to_translation_request(value)["parameters"] == {}


def test_sample_registered_model():
    value = intent(n_samples=100, save_to_xyz=True)
    assert to_translation_request(value, trusted_inputs={"model_ref": "model_rl_001"})["inputs"] == {"model_ref": "model_rl_001"}


def test_sample_qualitative_goals_do_not_become_cutoffs():
    value = intent(n_samples=50, qed_direction="maximize", sas_direction="minimize")
    result = to_translation_request(value)
    assert result["parameters"] == {"n_samples": 50, "requested_objectives": {"qed": "maximize", "sas": "minimize"}}
    assert "qed_cutoff" not in result["parameters"]


def test_rl_diffusion_training():
    value = intent(operation="train", conditioning=["qed", "sas", "affinity"])
    result = to_translation_request(value, trusted_inputs={"training_dataset_ref": "upload_molecules_001"})
    assert result["parameters"] == {"conditioning": ["qed", "sas", "affinity"]}
    assert result["inputs"] == {"training_dataset_ref": "upload_molecules_001"}


def test_ambiguous_algorithm():
    value = intent(algorithm_id=None, operation=None)
    result = to_translation_request(value)
    assert set(result["unresolved"]) == {"algorithm_id", "operation"}


def test_incompatible_operation_not_forwarded():
    value = intent(algorithm_id="mac_amp", operation="sample", n_samples=3)
    result = to_translation_request(value)
    assert result["operation"] is None
    assert result["parameters"] == {}
    assert "algorithm_operation_mismatch" in result["unresolved"]


def test_invalid_types_rejected_by_pydantic():
    with pytest.raises(Exception):
        intent(n_samples="lots")


def test_interpret_calls_injected_provider():
    fake = FakeProvider(intent(n_samples=25))
    result = interpret_user_prompt("Generate 25 molecules", provider=fake)
    assert fake.calls == 1
    assert result["parameters"]["n_samples"] == 25


@pytest.mark.parametrize("value", ["", "  ", None])
def test_empty_prompt_rejected(value):
    fake = FakeProvider(intent())
    with pytest.raises(ValueError):
        interpret_user_prompt(value, provider=fake)
    assert fake.calls == 0


def test_long_prompt_rejected():
    with pytest.raises(ValueError):
        interpret_user_prompt("x" * 12001, provider=FakeProvider(intent()))


# Auto-routing is an LLM behavior. These offline tests verify that the provider's
# selected capability passes through unchanged, and the routing instructions
# contain examples for unnamed requests. They do NOT measure actual LLM accuracy.
def test_unnamed_peptide_request_routes_from_provider():
    fake = FakeProvider(intent(algorithm_id="mac_amp", operation="peptide_generation", target_organism="Escherichia coli"))
    result = interpret_user_prompt("Design antimicrobial peptides for E. coli", provider=fake)
    assert result["algorithm_id"] == "mac_amp"
    assert result["operation"] == "peptide_generation"


def test_unnamed_molecule_request_routes_from_provider():
    fake = FakeProvider(intent(algorithm_id="rl_diffusion", operation="sample", n_samples=50))
    result = interpret_user_prompt("Generate 50 new 3D molecules", provider=fake)
    assert result["algorithm_id"] == "rl_diffusion"
    assert result["parameters"] == {"n_samples": 50}


def test_unnamed_molecular_training_routes_from_provider():
    fake = FakeProvider(intent(algorithm_id="rl_diffusion", operation="train", conditioning=["qed"]))
    result = interpret_user_prompt("Train a generative model using my molecular dataset", provider=fake)
    assert result["operation"] == "train"


def test_ambiguous_goals_remain_unrouted():
    fake = FakeProvider(intent(algorithm_id=None, operation=None, ambiguities=["Scientific result is unclear"]))
    result = interpret_user_prompt("Generate some promising candidates", provider=fake)
    assert result["algorithm_id"] is None
    assert "algorithm_id" in result["unresolved"]


def test_default_catalog_and_instructions_include_unnamed_routing():
    from llm.prompts import build_user_input, SYSTEM_INSTRUCTIONS
    formatted = build_user_input("Generate antimicrobial peptides against E. coli")
    assert '"mac_amp"' in formatted and '"rl_diffusion"' in formatted
    assert "DOES NOT need to mention an algorithm" in SYSTEM_INSTRUCTIONS
    assert "ambiguous" in SYSTEM_INSTRUCTIONS.lower()

# Revision tests: offline conversion only. Real LLM routing must be evaluated separately.
from llm.interpreter import NumericalGoal


def test_exact_threshold_reported_without_schema_change():
    extracted = intent(n_samples=50, numerical_goals=[
        NumericalGoal(metric="qed", operator=">=", value=0.8, upper_value=None, unit=None)
    ])
    result = to_translation_request(extracted)
    assert result["parameters"] == {"n_samples": 50}
    assert "numerical_objective_requires_schema_approval:qed:>=:0.8" in result["unresolved"]


def test_exact_threshold_opt_in_is_typed_not_invented():
    extracted = intent(n_samples=50, numerical_goals=[
        NumericalGoal(metric="qed", operator=">=", value=0.8, upper_value=None, unit=None),
        NumericalGoal(metric="sas", operator="<=", value=3, upper_value=None, unit=None),
    ])
    result = to_translation_request(extracted, enable_numeric_objectives=True)
    goals = result["parameters"]["requested_objectives"]
    assert goals["qed"] == {"operator": ">=", "value": 0.8, "upper_value": None, "unit": None}
    assert goals["sas"]["value"] == 3
    assert result["unresolved"] == []


def test_invalid_qed_cutoff_is_not_forwarded():
    extracted = intent(numerical_goals=[NumericalGoal(metric="qed", operator=">", value=1.5, upper_value=None, unit=None)])
    result = to_translation_request(extracted, enable_numeric_objectives=True)
    assert "requested_objectives" not in result["parameters"]
    assert "qed: value must be between 0 and 1" in result["unresolved"]


def test_affinity_needs_units_before_opt_in():
    extracted = intent(numerical_goals=[NumericalGoal(metric="affinity", operator="<=", value=-8, upper_value=None, unit=None)])
    result = to_translation_request(extracted, enable_numeric_objectives=True)
    assert "affinity: missing measurement units / scoring convention" in result["unresolved"]
    assert "requested_objectives" not in result["parameters"]


def test_nonfinite_numeric_goals_rejected():
    with pytest.raises(ValueError):
        NumericalGoal(metric="qed", operator=">", value=float("nan"), upper_value=None, unit=None)


def test_unhandled_requirements_and_contradictions_preserved():
    extracted = intent(n_samples=50, ambiguities=["Conflicting molecule counts"],
                       unhandled_requirements=["Requested a custom docking target"])
    result = to_translation_request(extracted)
    assert result["parameters"]["n_samples"] == 50
    assert "Conflicting molecule counts" in result["unresolved"]
    assert "Requested a custom docking target" in result["unresolved"]


def test_duplicate_reasons_deduplicated():
    result = to_translation_request(intent(ambiguities=[" unclear ", "unclear", " "]))
    assert result["unresolved"] == ["unclear"]


def test_cross_operation_values_cannot_disappear_silently():
    extracted = intent(operation="train", n_samples=20, save_to_xyz=True, conditioning=["qed"])
    result = to_translation_request(extracted)
    assert "n_samples_not_supported_for_selected_operation" in result["unresolved"]
    assert "save_to_xyz_not_supported_for_selected_operation" in result["unresolved"]
    assert result["parameters"] == {"conditioning": ["qed"]}


def test_custom_registry_restricts_operation():
    restricted = {"mac_amp": {"operations": {"peptide_generation": {}}}, "rl_diffusion": {"operations": {"sample": {}}}}
    result = to_translation_request(intent(operation="train", conditioning=["qed"]), algorithm_specs=restricted)
    assert result["operation"] is None
    assert "unsupported_algorithm_operation" in result["unresolved"]
    assert result["parameters"] == {}


def test_custom_registry_cannot_enable_new_unsupported_operation():
    restricted = {"rl_diffusion": {"operations": {"train": {}}}}
    result = to_translation_request(intent(operation="sample", n_samples=5), algorithm_specs=restricted)
    assert result["operation"] is None
    assert "unsupported_algorithm_operation" in result["unresolved"]


def test_untrusted_file_id_not_extracted_from_text():
    result = interpret_user_prompt(
        "My model is located at C:/secret/model.pt", provider=FakeProvider(intent(n_samples=10))
    )
    assert result["inputs"] == {}


def test_followup_context_passed_to_provider_without_implicit_merge():
    prior = {"schema_version": "1.0", "algorithm_id": "rl_diffusion", "operation": "sample", "parameters": {"n_samples": 50}, "inputs": {}, "unresolved": []}
    class FollowupProvider:
        def __init__(self):
            self.prompt = ""
        def extract(self, prompt, specs):
            self.prompt = prompt
            return intent(n_samples=100, save_to_xyz=True)
    fake = FollowupProvider()
    result = interpret_user_prompt("Actually, make it 100 and save XYZ", previous_request=prior, provider=fake)
    assert "n_samples" in fake.prompt and "100" in fake.prompt
    assert result["parameters"] == {"n_samples": 100, "save_to_xyz": True}


def test_malformed_previous_context_is_rejected():
    with pytest.raises(ValueError):
        interpret_user_prompt("Make it 100", previous_request={"schema_version": "wrong"}, provider=FakeProvider(intent()))


def test_multistep_request_is_flagged_by_extractor():
    extracted = intent(algorithm_id=None, operation=None, unhandled_requirements=["multi_operation_request"])
    result = interpret_user_prompt("First peptides then 3D molecules", provider=FakeProvider(extracted))
    assert "multi_operation_request" in result["unresolved"]
    assert result["algorithm_id"] is None


def test_negative_count_is_not_forwarded():
    result = to_translation_request(intent(n_samples=-3))
    assert "n_samples" not in result["parameters"]
    assert "n_samples must be a positive integer" in result["unresolved"]


def test_boolean_false_is_preserved():
    result = to_translation_request(intent(save_to_xyz=False))
    assert result["parameters"]["save_to_xyz"] is False


def test_system_instructions_cover_numeric_and_unsupported_inputs():
    from llm.prompts import SYSTEM_INSTRUCTIONS
    assert "unhandled_requirements" in SYSTEM_INSTRUCTIONS
    assert "numerical_goals" in SYSTEM_INSTRUCTIONS
    assert "COMPLETE" not in SYSTEM_INSTRUCTIONS or "complete" in SYSTEM_INSTRUCTIONS.lower()


# Fixes identified by the first live evaluation (6/10 initial score).
def test_route_recovery_preserves_molecular_parameters():
    fake = FakeProvider(intent(algorithm_id=None, operation="sample", n_samples=30, qed_direction="maximize", ambiguities=["algorithm_id"]))
    result = interpret_user_prompt("Make 30 molecules with high drug-likeness", provider=fake)
    assert result["algorithm_id"] == "rl_diffusion"
    assert result["parameters"] == {"n_samples": 30, "requested_objectives": {"qed": "maximize"}}
    assert "algorithm_id" not in result["unresolved"]


def test_route_recovery_preserves_qed_threshold_as_unresolved():
    fake = FakeProvider(intent(algorithm_id=None, operation="sample", n_samples=20, numerical_goals=[
        NumericalGoal(metric="qed", operator=">=", value=0.8, upper_value=None, unit=None)
    ]))
    result = interpret_user_prompt("Generate 20 molecules with QED at least 0.8", provider=fake)
    assert result["algorithm_id"] == "rl_diffusion"
    assert result["parameters"]["n_samples"] == 20
    assert "numerical_objective_requires_schema_approval:qed:>=:0.8" in result["unresolved"]


def test_route_recovery_does_not_change_unclear_request():
    fake = FakeProvider(intent(algorithm_id=None, operation="sample", n_samples=30, ambiguities=["not sure which algorithm"] ))
    result = interpret_user_prompt("Make 30 molecules with high drug-likeness", provider=fake)
    assert result["algorithm_id"] is None


def test_route_recovery_does_not_redirect_explicit_unsupported_algorithm():
    fake = FakeProvider(intent(algorithm_id=None, operation="sample", n_samples=20))
    result = interpret_user_prompt("Use Imaginary algorithm to generate 20 molecules", provider=fake)
    assert result["algorithm_id"] is None


def test_route_recovery_does_not_redirect_multitask():
    fake = FakeProvider(intent(algorithm_id=None, operation="sample", n_samples=20, unhandled_requirements=["multi_operation_request"]))
    result = interpret_user_prompt("Make 20 peptides and molecules", provider=fake)
    assert result["algorithm_id"] is None


def test_nonspecific_bacterium_does_not_become_target():
    fake = FakeProvider(intent(algorithm_id="mac_amp", operation="peptide_generation", target_organism="bacterium"))
    result = interpret_user_prompt("Generate peptides for a bacterium", provider=fake)
    assert result["parameters"] == {}


def test_e_coli_abbreviation_preserved_as_user_wrote_it():
    fake = FakeProvider(intent(algorithm_id="mac_amp", operation="peptide_generation", target_organism="E. coli"))
    result = interpret_user_prompt("Generate antimicrobial peptides targeting E. coli", provider=fake)
    assert result["parameters"]["target_organism"] == "E. coli"
