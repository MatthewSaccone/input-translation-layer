"""Deterministic validation of the shared six-field interpretation contract.

This checks request shape and supported parameters, NOT resource existence or
scientific execution readiness. MAC-AMP's adapter resolves optional registered
resources and approved defaults; RL-Diffusion retains its current requirements
until its adapter is completed. No scientific algorithms are run here.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

SUPPORTED_OPERATIONS = {
    "mac_amp": {"peptide_generation"},
    "rl_diffusion": {"sample", "train"},
}

# MAC-AMP has adapter defaults, so these are optional in the *request*.
# Their existence/compatibility must be checked by the adapter before running.
REQUIRED_INPUTS = {
    ("mac_amp", "peptide_generation"): set(),
    ("rl_diffusion", "sample"): {"model_ref"},
    ("rl_diffusion", "train"): {"training_dataset_ref"},
}

MAC_AMP_INTEGER_PARAMETERS = {
    "num_samples", "sample", "max_seq_len", "max_length", "batch_size",
    "real_epoch", "sand_box_epoch", "real_train_stage_iteration",
    "sandbox_iteration", "reward_sample_num",
}
MAC_AMP_ALLOWED_PARAMETERS = {
    "target_organism", *MAC_AMP_INTEGER_PARAMETERS,
    "temperature", "ablation_mode", "experiment", "skip_independent_reviewer",
}
ALLOWED_PARAMETERS = {
    ("mac_amp", "peptide_generation"): MAC_AMP_ALLOWED_PARAMETERS,
    ("rl_diffusion", "sample"): {"n_samples", "save_to_xyz", "requested_objectives"},
    ("rl_diffusion", "train"): {"conditioning"},
}
ALLOWED_INPUTS = {
    ("mac_amp", "peptide_generation"): {"peptide_dataset_ref", "model_ref", "workflow_config_ref"},
    ("rl_diffusion", "sample"): {"model_ref"},
    ("rl_diffusion", "train"): {"training_dataset_ref"},
}
EXPERIMENT_MODES = {
    "baseline_api", "all_llama_local", "all_qwen_local",
    "qwen_reviewers_gpt5_ac", "api_reviewers_qwen_ac",
}
OBJECTIVE_DIRECTIONS = {"qed": "maximize", "sas": "minimize", "affinity": "improve"}
OBJECTIVE_OPERATORS = {">", ">=", "<", "<=", "=", "range"}


@dataclass(frozen=True)
class ValidationReport:
    request: dict[str, Any]
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


def _add(errors: list[str], message: str) -> None:
    if message not in errors:
        errors.append(message)


def _require_nonempty_string(value: Any, field: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        _add(errors, f"{field} must be a non-empty string")


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _validate_objectives(objectives: Any, errors: list[str]) -> None:
    if not isinstance(objectives, Mapping):
        _add(errors, "parameters.requested_objectives must be an object")
        return
    if not objectives:
        _add(errors, "parameters.requested_objectives cannot be empty")
    for key, value in objectives.items():
        name = f"parameters.requested_objectives.{key}"
        if key not in OBJECTIVE_DIRECTIONS:
            _add(errors, f"unsupported objective: {key}")
            continue
        if isinstance(value, str):
            if value != OBJECTIVE_DIRECTIONS[key]:
                _add(errors, f"{name} has an unsupported direction")
            continue
        if not isinstance(value, Mapping):
            _add(errors, f"{name} must be a supported direction or threshold object")
            continue
        if set(value) != {"operator", "value", "upper_value", "unit"}:
            _add(errors, f"{name} must contain operator, value, upper_value, and unit")
            continue
        op, number, upper, unit = (value[x] for x in ("operator", "value", "upper_value", "unit"))
        if not isinstance(op, str) or op not in OBJECTIVE_OPERATORS:
            _add(errors, f"{name}.operator is unsupported")
        if type(number) not in (int, float) or not isfinite(number):
            _add(errors, f"{name}.value must be finite numeric")
            continue
        if upper is not None and (type(upper) not in (int, float) or not isfinite(upper)):
            _add(errors, f"{name}.upper_value must be finite numeric or null")
            continue
        if op == "range" and (upper is None or number > upper):
            _add(errors, f"{name} needs a valid upper bound")
        if op != "range" and upper is not None:
            _add(errors, f"{name} cannot have upper_value for a single-value comparison")
        if unit is not None and (not isinstance(unit, str) or not unit.strip()):
            _add(errors, f"{name}.unit must be a non-empty string or null")
        if key == "qed" and (not 0 <= number <= 1 or (upper is not None and not 0 <= upper <= 1)):
            _add(errors, f"{name} QED must be between 0 and 1")
        if key in ("qed", "sas") and unit is not None:
            _add(errors, f"{name} must not have units")
        if key == "affinity" and unit is None:
            _add(errors, f"{name} requires an explicit unit or scoring convention")


def validate_translation_request(request: Mapping[str, Any]) -> ValidationReport:
    """Validate input without mutation or file/network access."""
    errors: list[str] = []
    if not isinstance(request, Mapping):
        return ValidationReport({}, ("request must be an object",))
    normalized = dict(request)
    six_fields = {"schema_version", "algorithm_id", "operation", "parameters", "inputs", "unresolved"}
    for extra in normalized.keys() - six_fields:
        _add(errors, f"unsupported top-level field: {extra}")
    if normalized.get("schema_version") != "1.0":
        _add(errors, "schema_version must be '1.0'")
    algorithm, operation = normalized.get("algorithm_id"), normalized.get("operation")
    parameters, inputs, unresolved = (normalized.get(x) for x in ("parameters", "inputs", "unresolved"))
    # First check types to avoid TypeError on unhashable JSON values.
    if not isinstance(algorithm, str) or algorithm not in SUPPORTED_OPERATIONS:
        _add(errors, "algorithm_id is missing or unsupported")
    if not isinstance(operation, str) or operation not in (SUPPORTED_OPERATIONS.get(algorithm, set()) if isinstance(algorithm, str) else set()):
        _add(errors, "operation is missing or unsupported for algorithm_id")
    if not isinstance(parameters, Mapping):
        _add(errors, "parameters must be an object")
        parameters = {}
    if not isinstance(inputs, Mapping):
        _add(errors, "inputs must be an object")
        inputs = {}
    if not isinstance(unresolved, list):
        _add(errors, "unresolved must be a list")
        unresolved = []
    for issue in unresolved:
        if not isinstance(issue, str) or not issue.strip():
            _add(errors, "unresolved entries must be non-empty strings")
    if unresolved:
        _add(errors, "request contains unresolved requirements")
    capability = (algorithm, operation) if isinstance(algorithm, str) and isinstance(operation, str) else None
    for key in parameters:
        if key not in ALLOWED_PARAMETERS.get(capability, set()):
            _add(errors, f"unsupported parameter: {key}")
    for key, value in inputs.items():
        if key not in ALLOWED_INPUTS.get(capability, set()):
            _add(errors, f"unsupported input: {key}")
        else:
            _require_nonempty_string(value, f"inputs.{key}", errors)
    if capability == ("mac_amp", "peptide_generation"):
        _require_nonempty_string(parameters.get("target_organism"), "parameters.target_organism", errors)
        for key in MAC_AMP_INTEGER_PARAMETERS:
            if key in parameters and not _positive_int(parameters[key]):
                _add(errors, f"parameters.{key} must be a positive integer")
        if "ablation_mode" in parameters and (type(parameters["ablation_mode"]) is not int or not 0 <= parameters["ablation_mode"] <= 7):
            _add(errors, "parameters.ablation_mode must be an integer from 0 to 7")
        if "temperature" in parameters and (type(parameters["temperature"]) not in (int, float) or not isfinite(parameters["temperature"]) or not 0 <= parameters["temperature"] <= 2):
            _add(errors, "parameters.temperature must be between 0 and 2")
        if "experiment" in parameters and (
            not isinstance(parameters["experiment"], str) or parameters["experiment"] not in EXPERIMENT_MODES
        ):
            _add(errors, "parameters.experiment must be a supported experiment mode")
        if "skip_independent_reviewer" in parameters and not isinstance(parameters["skip_independent_reviewer"], str):
            _add(errors, "parameters.skip_independent_reviewer must be a string")
    elif capability == ("rl_diffusion", "sample"):
        if "n_samples" in parameters and not _positive_int(parameters["n_samples"]):
            _add(errors, "parameters.n_samples must be a positive integer")
        if "save_to_xyz" in parameters and type(parameters["save_to_xyz"]) is not bool:
            _add(errors, "parameters.save_to_xyz must be a boolean")
        if "requested_objectives" in parameters:
            _validate_objectives(parameters["requested_objectives"], errors)
    elif capability == ("rl_diffusion", "train"):
        if "conditioning" in parameters:
            cond = parameters["conditioning"]
            if not isinstance(cond, list) or not cond or any(not isinstance(x, str) or x not in {"qed", "sas", "affinity"} for x in cond):
                _add(errors, "parameters.conditioning must contain only qed, sas, or affinity")
    for required in REQUIRED_INPUTS.get(capability, set()):
        _require_nonempty_string(inputs.get(required), f"inputs.{required}", errors)
    return ValidationReport(normalized, tuple(errors))


def is_valid_translation_request(request: Mapping[str, Any]) -> bool:
    return validate_translation_request(request).valid
