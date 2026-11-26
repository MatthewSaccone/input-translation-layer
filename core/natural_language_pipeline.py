"""Researcher prompt -> interpretation -> validation -> non-executing preview.

MAC-AMP returns a configuration preview; RL-Diffusion returns a review record.
Neither path verifies scientific execution readiness or writes files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from adapters.rl_diffusion.RLDiffAdapter import AdapterError, RLDiffAdapter
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
    """Interpret once, validate, then dispatch to a non-executing adapter.

    This is a non-executing preview, not an assertion of scientific readiness.
    """
    request = interpret_user_prompt(prompt, provider=provider, trusted_inputs=trusted_inputs)
    report = validate_translation_request(request)
    if not report.valid:
        return NaturalLanguagePipelineResult(
            request,
            PipelineResult("invalid_request", report.request, report.errors),
        )
    if request["algorithm_id"] == "rl_diffusion":
        # Dispatch only to the review adapter. Do not resolve native paths or launch
        # scientific jobs until the algorithm owner has verified the handoff.
        try:
            preview = RLDiffAdapter().prepare_preview(report.request)
        except AdapterError as exc:
            return NaturalLanguagePipelineResult(
                request, PipelineResult("adapter_rejected", report.request, (str(exc),))
            )
        return NaturalLanguagePipelineResult(
            request,
            PipelineResult(
                "review_preview_only", report.request,
                tuple(preview["blocking_issues"]), argument_preview=preview,
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
