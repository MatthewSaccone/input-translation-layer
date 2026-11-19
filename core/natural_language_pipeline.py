"""Researcher prompt -> interpretation -> validation -> MAC-AMP YAML preview.

No scientific algorithms are executed, and this module writes no files.
The unfinished RL-Diffusion adapter is explicitly not dispatched.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.pipeline import PipelineResult, prepare_mac_amp_request
from core.validator import validate_translation_request
from llm.interpreter import IntentProvider, interpret_user_prompt


@dataclass(frozen=True)
class NaturalLanguagePipelineResult:
    request: dict[str, Any]
    pipeline: PipelineResult


def prepare_from_prompt(
    prompt: str,
    *,
    provider: IntentProvider | None = None,
    trusted_inputs: dict[str, str] | None = None,
    resource_root: str | Path | None = None,
    resource_registry: Mapping[str, str | Path] | None = None,
    verify_resources: bool = False,
) -> NaturalLanguagePipelineResult:
    """Interpret once, validate, then dispatch only MAC-AMP requests.

    This is a non-executing preview, not an assertion of scientific readiness.
    """
    request = interpret_user_prompt(prompt, provider=provider, trusted_inputs=trusted_inputs)
    report = validate_translation_request(request)
    if not report.valid:
        return NaturalLanguagePipelineResult(
            request,
            PipelineResult("invalid_request", report.request, report.errors),
        )
    if request["algorithm_id"] != "mac_amp":
        return NaturalLanguagePipelineResult(
            request,
            PipelineResult(
                "adapter_not_implemented", request,
                ("RL-Diffusion adapter is not yet integrated",),
            ),
        )
    return NaturalLanguagePipelineResult(
        request,
        prepare_mac_amp_request(
            request,
            resource_root=resource_root,
            resource_registry=resource_registry,
            verify_resources=verify_resources,
        ),
    )
