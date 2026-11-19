from core.validator import is_valid_translation_request, validate_translation_request


def sample_request(**overrides):
    request = {
        "schema_version": "1.0",
        "algorithm_id": "rl_diffusion",
        "operation": "sample",
        "parameters": {"n_samples": 50, "save_to_xyz": True},
        "inputs": {"model_ref": "model_rl_001"},
        "unresolved": [],
    }
    request.update(overrides)
    return request


def test_valid_sample_request():
    report = validate_translation_request(sample_request())
    assert report.valid
    assert report.errors == ()


def test_missing_model_is_blocking():
    report = validate_translation_request(sample_request(inputs={}))
    assert not report.valid
    assert "inputs.model_ref must be a non-empty string" in report.errors


def test_unresolved_translation_is_not_executable():
    report = validate_translation_request(sample_request(unresolved=["algorithm_id"]))
    assert not report.valid
    assert "request contains unresolved requirements" in report.errors


def test_mac_amp_requires_target_and_resources():
    report = validate_translation_request({
        "schema_version": "1.0",
        "algorithm_id": "mac_amp",
        "operation": "peptide_generation",
        "parameters": {},
        "inputs": {},
        "unresolved": [],
    })
    assert set(report.errors) == {
        "parameters.target_organism must be a non-empty string",
    }


def test_invalid_sample_count_is_rejected():
    request = sample_request(parameters={"n_samples": 0})
    report = validate_translation_request(request)
    assert "parameters.n_samples must be a positive integer" in report.errors


def test_validator_does_not_mutate_request():
    request = sample_request()
    original = request.copy()
    validate_translation_request(request)
    assert request == original


def test_boolean_is_not_an_integer_sample_count():
    request = sample_request(parameters={"n_samples": True})
    assert not is_valid_translation_request(request)


def test_mac_amp_defaults_are_not_forced_into_researcher_request():
    request = {
        "schema_version": "1.0", "algorithm_id": "mac_amp",
        "operation": "peptide_generation", "parameters": {"target_organism": "E. coli", "batch_size": 16, "max_length": 40},
        "inputs": {}, "unresolved": [],
    }
    assert validate_translation_request(request).valid


def test_malformed_unhashable_algorithm_returns_report():
    report = validate_translation_request(sample_request(algorithm_id=["mac_amp"]))
    assert not report.valid
    assert "algorithm_id is missing or unsupported" in report.errors


def test_rejects_extra_input_keys_and_top_level_fields():
    report = validate_translation_request(sample_request(inputs={"model_ref": "ref", "unexpected": "path"}, extra="bad"))
    assert not report.valid
    assert "unsupported input: unexpected" in report.errors
    assert "unsupported top-level field: extra" in report.errors


def test_mac_amp_parameter_types():
    for invalid in (True, 0, -2, "16"):
        report = validate_translation_request({
            "schema_version": "1.0", "algorithm_id": "mac_amp", "operation": "peptide_generation",
            "parameters": {"target_organism": "E. coli", "batch_size": invalid},
            "inputs": {}, "unresolved": [],
        })
        assert not report.valid


def test_invalid_objective_structure_rejected():
    for objectives in ({"qed": "bad"}, {"wrong": "maximize"}, {"qed": {"operator": ">=", "value": 2, "upper_value": None, "unit": None}}):
        report = validate_translation_request(sample_request(parameters={"requested_objectives": objectives}))
        assert not report.valid
