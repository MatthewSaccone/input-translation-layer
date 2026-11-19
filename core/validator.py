"""Fail-closed validation for translated algorithm requests.

The interpreter extracts intent. This module decides whether that intent is
complete enough to hand to an algorithm adapter. It does not resolve files,
download models, or execute scientific code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


SUPPORTED_OPERATIONS = {
    "mac_amp": {"peptide_generation"},
    "rl_diffusion": {"sample", "train"},
}

REQUIRED_INPUTS = {
    ("mac_amp", "peptide_generation"): {
        "peptide_dataset_ref",
        "model_ref",
        "workflow_config_ref",
    },
    ("rl_diffusion", "sample"): {"model_ref"},
    ("rl_diffusion", "train"): {"training_dataset_ref"},
}

ALLOWED_PARAMETERS = {
    ("mac_amp", "peptide_generation"): {"target_organism"},
    ("rl_diffusion", "sample"): {"n_samples", "save_to_xyz", "requested_objectives"},
    ("rl_diffusion", "train"): {"conditioning"},
}


@dataclass(frozen=True)
class ValidationReport:
    """The original request plus all reasons it cannot execute."""

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


def validate_translation_request(request: Mapping[str, Any]) -> ValidationReport:
    """Validate a version 1.0 translation request without mutating it.

    Resource references are intentionally checked by key and non-empty value;
    resolving whether a reference exists belongs to the adapter or registry.
    """
    errors: list[str] = []
    if not isinstance(request, Mapping):
        return ValidationReport({}, ("request must be an object",))

    normalized = dict(request)
    if normalized.get("schema_version") != "1.0":
        _add(errors, "schema_version must be '1.0'")

    algorithm = normalized.get("algorithm_id")
    operation = normalized.get("operation")
    parameters = normalized.get("parameters")
    inputs = normalized.get("inputs")
    unresolved = normalized.get("unresolved")

    if algorithm not in SUPPORTED_OPERATIONS:
        _add(errors, "algorithm_id is missing or unsupported")
    if not isinstance(operation, str) or operation not in SUPPORTED_OPERATIONS.get(algorithm, set()):
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

    capability = (algorithm, operation)
    allowed_parameters = ALLOWED_PARAMETERS.get(capability, set())
    for name in parameters:
        if name not in allowed_parameters:
            _add(errors, f"unsupported parameter: {name}")

    if capability == ("mac_amp", "peptide_generation"):
        _require_nonempty_string(parameters.get("target_organism"), "parameters.target_organism", errors)
    elif capability == ("rl_diffusion", "sample"):
        if "n_samples" in parameters:
            count = parameters["n_samples"]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                _add(errors, "parameters.n_samples must be a positive integer")
        if "save_to_xyz" in parameters and not isinstance(parameters["save_to_xyz"], bool):
            _add(errors, "parameters.save_to_xyz must be a boolean")
        objectives = parameters.get("requested_objectives")
        if objectives is not None and not isinstance(objectives, Mapping):
            _add(errors, "parameters.requested_objectives must be an object")
    elif capability == ("rl_diffusion", "train"):
        conditioning = parameters.get("conditioning")
        if conditioning is not None and (
            not isinstance(conditioning, list)
            or not conditioning
            or any(item not in {"qed", "sas", "affinity"} for item in conditioning)
        ):
            _add(errors, "parameters.conditioning must contain only qed, sas, or affinity")

    for required_input in REQUIRED_INPUTS.get(capability, set()):
        _require_nonempty_string(inputs.get(required_input), f"inputs.{required_input}", errors)

    return ValidationReport(normalized, tuple(errors))


def is_valid_translation_request(request: Mapping[str, Any]) -> bool:
    """Return whether a translated request is ready for an adapter."""
    return validate_translation_request(request).valid
