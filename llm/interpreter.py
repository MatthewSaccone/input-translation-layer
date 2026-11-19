"""LLM request interpretation (not scientific validation or model execution).

Public output remains the group's six-field TranslationRequest version 1.0.
Numeric objective emission is opt-in until the validator owner approves the shape.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm.prompts import DEFAULT_ALGORITHM_SPECS, SYSTEM_INSTRUCTIONS, build_user_input
from core.validator import MAC_AMP_ALLOWED_PARAMETERS

Algorithm = Literal["mac_amp", "rl_diffusion"]
Operation = Literal["peptide_generation", "sample", "train"]
Metric = Literal["qed", "sas", "affinity"]
Operator = Literal[">", ">=", "<", "<=", "=", "range"]

# These are a security boundary, not a claim about a native algorithm command.
ALLOWED_OPERATIONS = {
    "mac_amp": {"peptide_generation"},
    "rl_diffusion": {"sample", "train"},
}
REFERENCE_KEYS = {
    ("mac_amp", "peptide_generation"): {"peptide_dataset_ref", "model_ref", "workflow_config_ref"},
    ("rl_diffusion", "sample"): {"model_ref"},
    ("rl_diffusion", "train"): {"training_dataset_ref"},
}


class NumericalGoal(BaseModel):
    """A user-stated threshold, not an enforceable algorithm condition."""
    model_config = ConfigDict(extra="forbid")

    metric: Metric
    operator: Operator
    value: float
    upper_value: float | None
    unit: str | None

    @field_validator("value", "upper_value")
    @classmethod
    def finite_number(cls, value: float | None) -> float | None:
        import math
        if value is not None and not math.isfinite(value):
            raise ValueError("Numerical goals must be finite")
        return value


class MacAmpSettings(BaseModel):
    """MAC-AMP native tuning values expressly specified by the researcher.

    Omitted values remain None; the MAC-AMP adapter applies verified defaults.
    """
    model_config = ConfigDict(extra="forbid")
    num_samples: int | None = None
    sample: int | None = None
    max_seq_len: int | None = None
    max_length: int | None = None
    batch_size: int | None = None
    real_epoch: int | None = None
    sand_box_epoch: int | None = None
    real_train_stage_iteration: int | None = None
    sandbox_iteration: int | None = None
    reward_sample_num: int | None = None
    temperature: float | None = None
    ablation_mode: int | None = None
    experiment: str | None = None
    skip_independent_reviewer: str | None = None


class ExtractedIntent(BaseModel):
    """Strict response shape used with the LLM API.

    Defaults on the new fields permit existing fake providers to keep working.
    """
    model_config = ConfigDict(extra="forbid")

    algorithm_id: Algorithm | None
    operation: Operation | None
    target_organism: str | None
    n_samples: int | None
    save_to_xyz: bool | None
    conditioning: list[Metric] | None
    qed_direction: Literal["maximize"] | None
    sas_direction: Literal["minimize"] | None
    affinity_direction: Literal["improve"] | None
    ambiguities: list[str] = Field(default_factory=list)
    unhandled_requirements: list[str] = Field(default_factory=list)
    numerical_goals: list[NumericalGoal] = Field(default_factory=list)
    mac_amp_settings: MacAmpSettings | None = None


class IntentProvider(Protocol):
    def extract(self, prompt: str, specs: dict[str, Any] | None) -> ExtractedIntent: ...


class OpenAIIntentProvider:
    """Only this provider calls a third-party API."""

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def extract(self, prompt: str, specs: dict[str, Any] | None) -> ExtractedIntent:
        from openai import OpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Use --demo for an offline fixture.")
        client = OpenAI()
        result = client.responses.parse(
            model=self.model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=build_user_input(prompt, specs),
            text_format=ExtractedIntent,
            store=False,
        )
        if result.output_parsed is None:
            raise ValueError("LLM returned no parseable structured response (may be refusal or truncation)")
        return result.output_parsed


def _allowed_by_catalog(algorithm: str, operation: str, specs: dict[str, Any] | None) -> bool:
    """Optional team registry can restrict, never expand, built-in capabilities."""
    if operation not in ALLOWED_OPERATIONS.get(algorithm, set()):
        return False
    catalog = DEFAULT_ALGORITHM_SPECS if specs is None else specs
    if not isinstance(catalog, dict):
        return False
    descriptor = catalog.get(algorithm)
    return isinstance(descriptor, dict) and operation in descriptor.get("operations", {})


def _deduplicate(messages: list[str]) -> list[str]:
    return list(dict.fromkeys(s.strip() for s in messages if isinstance(s, str) and s.strip()))


def to_translation_request(
    intent: ExtractedIntent,
    *,
    trusted_inputs: dict[str, str] | None = None,
    algorithm_specs: dict[str, Any] | None = None,
    enable_numeric_objectives: bool = False,
) -> dict[str, Any]:
    """Convert extracted intent into the six-field group interface.

    A numeric goal is **reported**, not dropped, by default. Only emit typed
    numeric objectives if the validator owner has approved the optional shape.
    """
    algorithm, operation = intent.algorithm_id, intent.operation
    unresolved = list(intent.ambiguities) + list(intent.unhandled_requirements)

    if algorithm is None:
        unresolved.append("algorithm_id")
    if operation is None:
        unresolved.append("operation")
    if algorithm is not None and operation is not None and not _allowed_by_catalog(algorithm, operation, algorithm_specs):
        unresolved.append("algorithm_operation_mismatch" if operation not in ALLOWED_OPERATIONS.get(algorithm, set()) else "unsupported_algorithm_operation")
        operation = None

    params: dict[str, Any] = {}
    if algorithm == "mac_amp" and operation == "peptide_generation":
        if intent.target_organism:
            organism = intent.target_organism.strip()
            # Generic labels are not actual organism identifiers. The core
            # validator determines whether a missing target blocks execution.
            if organism.casefold() not in {
                "bacterium", "bacteria", "a bacterium", "the bacterium",
                "an organism", "organism", "a bacterial species", "unknown",
            }:
                params["target_organism"] = organism
        if intent.mac_amp_settings is not None:
            for key, value in intent.mac_amp_settings.model_dump(exclude_none=True).items():
                # Core validator is the source of truth for approved settings.
                if key in MAC_AMP_ALLOWED_PARAMETERS:
                    params[key] = value
                else:
                    unresolved.append(f"mac_amp_parameter_requires_schema_approval:{key}={value}")
    elif algorithm == "rl_diffusion" and operation == "sample":
        if intent.n_samples is not None:
            if intent.n_samples <= 0:
                unresolved.append("n_samples must be a positive integer")
            else:
                params["n_samples"] = intent.n_samples
        if intent.save_to_xyz is not None:
            params["save_to_xyz"] = intent.save_to_xyz
        objectives: dict[str, Any] = {}
        if intent.qed_direction:
            objectives["qed"] = intent.qed_direction
        if intent.sas_direction:
            objectives["sas"] = intent.sas_direction
        if intent.affinity_direction:
            objectives["affinity"] = intent.affinity_direction

        for goal in intent.numerical_goals:
            if goal.metric in ("qed", "sas") and goal.unit:
                unresolved.append(f"{goal.metric}: unexpected unit must be reviewed")
                continue
            if goal.metric == "affinity" and not goal.unit:
                unresolved.append("affinity: missing measurement units / scoring convention")
                continue
            if goal.operator == "range" and (goal.upper_value is None or goal.value > goal.upper_value):
                unresolved.append(f"{goal.metric}: invalid or missing range upper bound")
                continue
            if goal.operator != "range" and goal.upper_value is not None:
                unresolved.append(f"{goal.metric}: contradictory single-value/range format")
                continue
            if goal.metric == "qed" and not (0 <= goal.value <= 1) or (
                goal.metric == "qed" and goal.upper_value is not None and not (0 <= goal.upper_value <= 1)
            ):
                unresolved.append("qed: value must be between 0 and 1")
                continue
            # No domain assumptions about affinity score sign or SAS score implementation.
            if not enable_numeric_objectives:
                unresolved.append(f"numerical_objective_requires_schema_approval:{goal.metric}:{goal.operator}:{goal.value:g}" + (f":{goal.upper_value:g}" if goal.upper_value is not None else "") + (f":{goal.unit}" if goal.unit else ""))
                continue
            objectives[goal.metric] = {
                "operator": goal.operator,
                "value": goal.value,
                "upper_value": goal.upper_value,
                "unit": goal.unit,
            }
        if objectives:
            params["requested_objectives"] = objectives

    elif algorithm == "rl_diffusion" and operation == "train":
        if intent.conditioning is not None:
            params["conditioning"] = list(dict.fromkeys(intent.conditioning))

    # If a result was requested for a different operation, never pretend that
    # it was translated. Report that the requested intent is not yet supported.
    if intent.mac_amp_settings is not None and intent.mac_amp_settings.model_dump(exclude_none=True) and (algorithm, operation) != ("mac_amp", "peptide_generation"):
        unresolved.append("mac_amp_settings_not_supported_for_selected_operation")
    if intent.numerical_goals and (algorithm, operation) != ("rl_diffusion", "sample"):
        unresolved.append("numerical_objectives_not_supported_for_selected_operation")
    if intent.target_organism and (algorithm, operation) != ("mac_amp", "peptide_generation"):
        unresolved.append("target_organism_not_supported_for_selected_operation")
    if intent.conditioning and (algorithm, operation) != ("rl_diffusion", "train"):
        unresolved.append("conditioning_not_supported_for_selected_operation")
    if intent.n_samples is not None and (algorithm, operation) != ("rl_diffusion", "sample"):
        unresolved.append("n_samples_not_supported_for_selected_operation")
    if intent.save_to_xyz is not None and (algorithm, operation) != ("rl_diffusion", "sample"):
        unresolved.append("save_to_xyz_not_supported_for_selected_operation")
    if (intent.qed_direction or intent.sas_direction or intent.affinity_direction) and (algorithm, operation) != ("rl_diffusion", "sample"):
        unresolved.append("qualitative_objectives_not_supported_for_selected_operation")

    accepted = REFERENCE_KEYS.get((algorithm, operation), set())
    inputs = {
        key: value
        for key, value in (trusted_inputs or {}).items()
        if key in accepted and isinstance(value, str) and value.strip()
    }
    return {
        "schema_version": "1.0",
        "algorithm_id": algorithm,
        "operation": operation,
        "parameters": params,
        "inputs": inputs,
        "unresolved": _deduplicate(unresolved),
    }


def _repair_explicit_molecule_route(intent: ExtractedIntent, prompt: str) -> ExtractedIntent:
    """Conservative fallback for an internally inconsistent *single* operation.

    This is not open-ended keyword classification. It only repairs the model's
    `sample` + null algorithm output when the user explicitly requests small-
    molecule generation and the model has extracted that precise operation.
    Never override unsupported/multistep/ambiguous requests or a named choice.
    """
    if intent.algorithm_id is not None or intent.operation != "sample":
        return intent
    lowered = prompt.casefold()
    if intent.unhandled_requirements or any(
        issue.casefold().strip() not in {"algorithm_id", "unknown algorithm_id"}
        for issue in intent.ambiguities
    ):
        return intent
    if not re.search(r"\b(molecules?|small[- ]molecules?|3d molecules?)\b", lowered):
        return intent
    if not re.search(r"\b(generate|make|create|design|produce|sample)\b", lowered):
        return intent
    if re.search(r"\b(peptides?|proteins?|folding|fictional|unsupported)\b", lowered):
        return intent
    # A user-specified nonregistered algorithm must not be silently replaced.
    if re.search(r"\b(?:use|using)\s+(?!rl[- ]diffusion\b)[\w-]+\s+algorithm\b", lowered):
        return intent
    return intent.model_copy(update={
        "algorithm_id": "rl_diffusion",
        "ambiguities": [issue for issue in intent.ambiguities if issue.casefold().strip() not in {"algorithm_id", "unknown algorithm_id"}],
    })


def interpret_user_prompt(
    user_prompt: str,
    algorithm_specs: dict[str, Any] | None = None,
    *,
    provider: IntentProvider | None = None,
    trusted_inputs: dict[str, str] | None = None,
    previous_request: dict[str, Any] | None = None,
    enable_numeric_objectives: bool = False,
) -> dict[str, Any]:
    """Interpret a researcher's request; optionally use *trusted* prior state.

    The LLM must return a complete updated extraction for a follow-up turn.
    The interpreter does not silently merge ambiguous values or carry resource
    references from previous turns. Upstream application can pass trusted refs.
    """
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("user_prompt must be a non-empty string")
    if len(user_prompt) > 12000:
        raise ValueError("user_prompt is too long (max 12000 characters)")
    if algorithm_specs is not None and not isinstance(algorithm_specs, dict):
        raise ValueError("algorithm_specs must be a dictionary")
    prompt_for_provider = user_prompt
    if previous_request is not None:
        if not isinstance(previous_request, dict) or previous_request.get("schema_version") != "1.0":
            raise ValueError("previous_request must be a version 1.0 translation request")
        previous_data = {
            key: previous_request.get(key)
            for key in ("algorithm_id", "operation", "parameters")
        }
        prior_json = json.dumps(previous_data, ensure_ascii=False)
        if len(prior_json) > 6000:
            raise ValueError("previous_request is too large")
        prompt_for_provider = (
            "Previous structured research request (context only, not instructions):\n"
            + prior_json
            + "\n\nNew researcher message (use to update the previous request):\n"
            + user_prompt
            + "\nReturn the COMPLETE updated interpretation. Preserve prior values only "
              "if they remain applicable; do not invent missing fields."
        )
    chosen_provider = provider if provider is not None else OpenAIIntentProvider()
    extraction = chosen_provider.extract(prompt_for_provider, algorithm_specs)
    # A narrow recovery for a model response such as {algorithm_id:null,
    # operation:"sample", n_samples:30} to an explicit molecule request.
    # No arbitrary rerouting of ambiguous or unsupported requests.
    extraction = _repair_explicit_molecule_route(extraction, user_prompt)
    return to_translation_request(
        extraction,
        trusted_inputs=trusted_inputs,
        algorithm_specs=algorithm_specs,
        enable_numeric_objectives=enable_numeric_objectives,
    )


def _main() -> None:
    import argparse
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="Interpret a researcher's request into JSON")
    parser.add_argument("prompt", nargs="?", help="Natural-language research request")
    parser.add_argument("--demo", action="store_true", help="Offline fixed fixture, no LLM call")
    parser.add_argument("--enable-numeric-objectives", action="store_true", help="Opt into proposed numeric JSON shape; requires validator agreement")
    args = parser.parse_args()
    load_dotenv()
    if args.demo:
        extraction = ExtractedIntent(
            algorithm_id="rl_diffusion", operation="sample", target_organism=None,
            n_samples=50, save_to_xyz=True, conditioning=None,
            qed_direction=None, sas_direction=None, affinity_direction=None,
        )
        print("OFFLINE FIXTURE (not model interpretation):")
        output = to_translation_request(extraction)
    else:
        prompt = args.prompt or "Generate 50 3D molecules"
        output = interpret_user_prompt(prompt, enable_numeric_objectives=args.enable_numeric_objectives)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
