"""Regression tests for the non-executing adapter. No scientific code is imported."""
from copy import deepcopy

import pytest

from adapters.rl_diffusion.RLDiffAdapter import AdapterError, RLDiffAdapter


def request(operation="sample", parameters=None, inputs=None):
    # Logical resource IDs are deliberately synthetic, not actual checkpoints.
    return {
        "schema_version": "1.0", "algorithm_id": "rl_diffusion",
        "operation": operation, "parameters": parameters or {},
        "inputs": inputs if inputs is not None else {"model_ref": "fixture_model"},
        "unresolved": [],
    }


def test_reference_remains_id_and_preview_is_not_native():
    original = request(parameters={"n_samples": 5})
    snapshot = deepcopy(original)
    result = RLDiffAdapter().prepare_preview(original)
    assert result["resource_refs"] == original["inputs"]
    assert "model_path" not in result
    assert result["execution_ready"] is False
    assert original == snapshot


def test_missing_model_is_not_replaced_by_default():
    with pytest.raises(AdapterError, match="model_ref"):
        RLDiffAdapter().prepare_preview(request(inputs={}))


def test_objectives_are_preserved_and_block_native_handoff():
    original = request(parameters={"requested_objectives": {"qed": "maximize"}})
    result = RLDiffAdapter().prepare_preview(original)
    assert result["parameters"] == original["parameters"]
    assert any("objectives" in issue for issue in result["blocking_issues"])
    assert not result["execution_ready"]


def test_training_reference_is_not_mistaken_for_native_dataset():
    original = request("train", {"conditioning": ["qed"]}, {"training_dataset_ref": "fixture_data"})
    result = RLDiffAdapter().prepare_preview(original)
    assert result["resource_refs"] == original["inputs"]
    assert result["parameters"]["conditioning"] == ["qed"]
    assert "dataset" not in result
    assert not result["execution_ready"]


def test_invalid_value_rejected_by_shared_validator():
    with pytest.raises(AdapterError, match="positive integer"):
        RLDiffAdapter().prepare_preview(request(parameters={"n_samples": True}))


def test_preview_has_no_working_directory_dependency(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert RLDiffAdapter().prepare_preview(request())["status"] == "review_preview_only"
