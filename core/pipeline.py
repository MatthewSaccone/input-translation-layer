"""MAC-AMP input-translation pipeline (no scientific model execution).

Interpretation is performed by `llm.interpreter` in the caller; this module
validates its six-field output and prepares a MAC-AMP Hydra configuration.
RL-Diffusion is intentionally excluded until its adapter is complete.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from adapters.mac_amp.translator import MacAmpAdapter, _dict_to_hydra_yaml
from core.validator import ValidationReport, validate_translation_request


@dataclass(frozen=True)
class PipelineResult:
    status: str
    request: dict[str, Any]
    issues: tuple[str, ...]
    yaml: str | None = None
    # Physical path checks are NOT proof of scientific checkpoint compatibility.
    resources_checked: bool = False


def prepare_mac_amp_request(
    request: Mapping[str, Any],
    *,
    resource_root: str | Path | None = None,
    resource_registry: Mapping[str, str | Path] | None = None,
    verify_resources: bool = False,
) -> PipelineResult:
    """Validate and prepare; return status instead of launching scientific code.

    Default mode is offline *preview*. For filesystem checks set both
    `verify_resources=True` and `resource_root=...`. No files are written.
    """
    report: ValidationReport = validate_translation_request(request)
    if not report.valid:
        return PipelineResult("invalid_request", report.request, report.errors)
    if report.request["algorithm_id"] != "mac_amp":
        return PipelineResult("adapter_not_implemented", report.request, ("RL-Diffusion adapter not yet integrated",))
    if verify_resources and resource_root is None:
        return PipelineResult("configuration_error", report.request, ("resource_root is required to verify resources",))
    adapter = MacAmpAdapter(
        resource_root=resource_root,
        resource_registry=resource_registry,
        verify_resources=verify_resources,
    )
    try:
        result = adapter.translate(report.request)
    except (ValueError, FileNotFoundError) as exc:
        return PipelineResult("adapter_rejected", report.request, (str(exc),))
    if result.status != "success" or result.config is None:
        return PipelineResult("adapter_rejected", report.request, tuple(result.missing_or_ambiguous_fields or [result.clarification_message or "Unknown adapter issue"]))
    yaml_text = _dict_to_hydra_yaml(result.config.model_dump(mode="json"))
    return PipelineResult(
        "resources_found_not_execution_verified" if verify_resources else "config_preview_unverified",
        report.request, (), yaml_text, verify_resources,
    )
