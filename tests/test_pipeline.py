from pathlib import Path
import tempfile

from core.pipeline import prepare_mac_amp_request
from llm.interpreter import ExtractedIntent, MacAmpSettings, to_translation_request


def mac_request(**params):
    return {"schema_version": "1.0", "algorithm_id": "mac_amp", "operation": "peptide_generation",
            "parameters": {"target_organism": "E. coli", **params}, "inputs": {}, "unresolved": []}


def test_llm_to_validator_to_adapter_with_mac_tuning():
    intent = ExtractedIntent(
        algorithm_id="mac_amp", operation="peptide_generation", target_organism="E. coli",
        n_samples=None, save_to_xyz=None, conditioning=None,
        qed_direction=None, sas_direction=None, affinity_direction=None,
        mac_amp_settings=MacAmpSettings(batch_size=16, max_length=40),
    )
    request = to_translation_request(intent)
    assert request["parameters"]["batch_size"] == 16
    result = prepare_mac_amp_request(request)
    assert result.status == "config_preview_unverified"
    assert 'batch_size: 16' in result.yaml
    assert 'max_length: 40' in result.yaml


def test_missing_organism_stops_before_adapter():
    request = mac_request()
    request["parameters"] = {}
    assert prepare_mac_amp_request(request).status == "invalid_request"


def test_no_verification_without_root():
    assert prepare_mac_amp_request(mac_request(), verify_resources=True).status == "configuration_error"


def test_verified_resource_existence_preview():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "Generation_Module" / "data" / "warmup").mkdir(parents=True)
        (root / "Generation_Module" / "data" / "warmup" / "warmup_AMP_Escherichia_coli.csv").write_text("sequence\nAAAA\n", encoding="utf-8")
        (root / "Generation_Module" / "model_ckpt").mkdir(parents=True)
        (root / "data" / "AMP_Escherichia_coli_811_new" / "ref_pdbs").mkdir(parents=True)
        result = prepare_mac_amp_request(mac_request(), resource_root=root, verify_resources=True)
        assert result.status == "resources_found_not_execution_verified", result.issues
        assert result.resources_checked


def test_missing_resource_is_adapter_rejection():
    with tempfile.TemporaryDirectory() as tmp:
        result = prepare_mac_amp_request(mac_request(), resource_root=tmp, verify_resources=True)
        assert result.status == "adapter_rejected"
        assert result.issues


def test_rl_is_not_integrated():
    request = {"schema_version": "1.0", "algorithm_id": "rl_diffusion", "operation": "sample",
               "parameters": {"n_samples": 5}, "inputs": {"model_ref": "model_rl_001"}, "unresolved": []}
    assert prepare_mac_amp_request(request).status == "adapter_not_implemented"


def test_registered_path_cannot_escape_verification_root():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
        request = mac_request()
        request["inputs"] = {"peptide_dataset_ref": "uploaded_001"}
        result = prepare_mac_amp_request(
            request, resource_root=tmp, verify_resources=True,
            resource_registry={"uploaded_001": Path(outside) / "other.csv"},
        )
        assert result.status == "adapter_rejected"
        assert any("outside configured root" in issue for issue in result.issues)


def test_workflow_ref_must_be_registered_to_prevent_arbitrary_file_reads():
    request = mac_request()
    request["inputs"] = {"workflow_config_ref": "secrets.json"}
    result = prepare_mac_amp_request(request)
    assert result.status == "adapter_rejected"


def test_invalid_mac_amp_parameter_rejected_before_yaml():
    request = mac_request(batch_size=True)
    result = prepare_mac_amp_request(request)
    assert result.status == "invalid_request"
    assert result.yaml is None


def test_invalid_request_does_not_modify_original_object():
    request = mac_request()
    request["unresolved"] = ["unknown organism variant"]
    original = {**request, "unresolved": list(request["unresolved"])}
    result = prepare_mac_amp_request(request)
    assert result.status == "invalid_request"
    assert request == original


def test_unregistered_model_id_is_not_a_path():
    # A logical identifier must not be copied into a native model-path field.
    request = mac_request()
    request["inputs"] = {"model_ref": "unregistered_fixture"}
    result = prepare_mac_amp_request(request)
    assert result.status == "adapter_rejected"
    assert result.yaml is None


def test_configuration_preview_has_native_compatibility_warning():
    result = prepare_mac_amp_request(mac_request())
    assert result.status == "config_preview_unverified"
    assert any("unverified" in warning for warning in result.warnings)
