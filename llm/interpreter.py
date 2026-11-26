"""LLM request interpretation (not scientific validation or model execution).

Public output remains the group's six-field TranslationRequest version 1.0.
Numeric objective emission is opt-in until the validator owner approves the shape.
"""
# Postpone evaluation of type annotations; lets us use modern type hints cleanly.
from __future__ import annotations

# JSON is the shared transport format between the interpreter and the validator.
import json
# Read configuration (model name and API key) from environment variables.
import os
# Regular expressions are used only for the narrow route-recovery check.
import re
# Type hints document accepted values and the provider interface.
from typing import Any, Literal, Protocol

# Pydantic defines and checks the shape of information returned by the LLM.
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Separate the task instructions/capability descriptions from API and conversion logic.
from llm.prompts import DEFAULT_ALGORITHM_SPECS, SYSTEM_INSTRUCTIONS, build_user_input
# Import the shared validator's approved MAC-AMP fields instead of creating a competing list.
from core.validator import MAC_AMP_ALLOWED_PARAMETERS

# Constrain the LLM extraction schema to our two registered algorithm IDs.
Algorithm = Literal["mac_amp", "rl_diffusion"]
# Constrain possible operation identifiers; algorithm/operation pairing is checked later.
Operation = Literal["peptide_generation", "sample", "train"]
# Scientific metrics the extraction schema can recognize (not proof a model can enforce them).
Metric = Literal["qed", "sas", "affinity"]
# Supported mathematical comparisons for a requested numerical objective.
Operator = Literal[">", ">=", "<", "<=", "=", "range"]

# These are a security boundary, not a claim about a native algorithm command.
# Fail-closed algorithm/operation whitelist: never authorize an unknown capability.
ALLOWED_OPERATIONS = {
    "mac_amp": {"peptide_generation"},
    "rl_diffusion": {"sample", "train"},
}
# Only these resource-reference keys may be accepted for a given operation.
REFERENCE_KEYS = {
    ("mac_amp", "peptide_generation"): {"peptide_dataset_ref", "model_ref", "workflow_config_ref"},
    ("rl_diffusion", "sample"): {"model_ref"},
    ("rl_diffusion", "train"): {"training_dataset_ref"},
}


# Pydantic model for one user-requested numerical constraint.
class NumericalGoal(BaseModel):
    """A user-stated threshold, not an enforceable algorithm condition."""
    # Reject unexpected Pydantic fields rather than quietly accepting new model-generated keys.
    model_config = ConfigDict(extra="forbid")

    # Which scientific property the user mentioned, e.g., QED.
    metric: Metric
    # Requested comparison, e.g., ">=" in "QED at least 0.8".
    operator: Operator
    # Primary numeric value explicitly stated by the researcher.
    value: float
    # Second endpoint for a range; None for one-sided comparisons.
    upper_value: float | None
    # Units if the user stated them, e.g., for affinity; do not guess.
    unit: str | None

    # Custom Pydantic validation for both numerical-value fields.
    @field_validator("value", "upper_value")
    @classmethod
    # Disallow NaN and infinity, which would make a scientific constraint nonsensical.
    def finite_number(cls, value: float | None) -> float | None:
        # Import here because the finite-value check is only needed in this validator.
        import math
        # None is allowed for optional upper_value; non-finite numbers are not.
        if value is not None and not math.isfinite(value):
            # Raise a validation error that upstream code can report.
            raise ValueError("Numerical goals must be finite")
        # Leave valid numeric values unchanged.
        return value


# MAC-AMP-specific tuning fields extracted from user language, independent of algorithm defaults.
class MacAmpSettings(BaseModel):
    """MAC-AMP native tuning values expressly specified by the researcher.

    Omitted values remain None; the MAC-AMP adapter applies verified defaults.
    """
    # Reject unexpected Pydantic fields rather than quietly accepting new model-generated keys.
    model_config = ConfigDict(extra="forbid")
    # All optional settings default to None: missing text must NOT become fabricated tuning values.
    num_samples: int | None = None
    sample: int | None = None
    max_seq_len: int | None = None
    # Maximum generated peptide length, as requested by the researcher.
    max_length: int | None = None
    # Training batch size; the validator later checks whether it is positive.
    batch_size: int | None = None
    # Real-training epoch count; this is distinct from sandbox epochs.
    real_epoch: int | None = None
    # Sandbox epoch count; spelling matches the existing MAC-AMP adapter key.
    sand_box_epoch: int | None = None
    real_train_stage_iteration: int | None = None
    sandbox_iteration: int | None = None
    reward_sample_num: int | None = None
    temperature: float | None = None
    ablation_mode: int | None = None
    # Optional MAC-AMP experiment backend mode; adapter determines allowed native values.
    experiment: str | None = None
    skip_independent_reviewer: str | None = None


# This is the *intermediate* structured LLM response, not the six-field final contract.
class ExtractedIntent(BaseModel):
    """Strict response shape used with the LLM API.

    Defaults on the new fields permit existing fake providers to keep working.
    """
    # Reject unexpected Pydantic fields rather than quietly accepting new model-generated keys.
    model_config = ConfigDict(extra="forbid")

    # Selected algorithm, or None if the request is ambiguous or unsupported.
    algorithm_id: Algorithm | None
    # Selected operation; None if we cannot reliably determine the task.
    operation: Operation | None
    # Target organism from the prompt; can be an abbreviation such as E. coli.
    target_organism: str | None
    # RL-Diffusion sampling count, separate from MAC-AMP num_samples.
    n_samples: int | None
    # Whether the researcher requested XYZ output; None means not specified.
    save_to_xyz: bool | None
    # Properties used for an RL-Diffusion training request.
    conditioning: list[Metric] | None
    qed_direction: Literal["maximize"] | None
    sas_direction: Literal["minimize"] | None
    affinity_direction: Literal["improve"] | None
    # New list for each request; stores contradictions or unclear meanings.
    ambiguities: list[str] = Field(default_factory=list)
    # Preserve requests we cannot map to a supported field, so they are not lost.
    unhandled_requirements: list[str] = Field(default_factory=list)
    # Keep exact thresholds separately until the validator approves their output shape.
    numerical_goals: list[NumericalGoal] = Field(default_factory=list)
    # Nested MAC-AMP tuning settings; None for unrelated operations or no custom settings.
    mac_amp_settings: MacAmpSettings | None = None


# An interface/contract for extraction providers; enables mock providers in offline tests.
class IntentProvider(Protocol):
    # Any provider must accept prompt + capabilities and return an ExtractedIntent object.
    def extract(self, prompt: str, specs: dict[str, Any] | None) -> ExtractedIntent: ...


# The concrete provider that calls OpenAI; the rest of the interpreter can be tested without it.
class OpenAIIntentProvider:
    """Only this provider calls a third-party API."""

    # Allow tests/callers to choose a model; otherwise use an environment default.
    def __init__(self, model: str | None = None):
        # Prefer the configured model; fall back to gpt-4o-mini when unset.
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Send one researcher request (plus catalog) to the LLM and parse structured intent.
    def extract(self, prompt: str, specs: dict[str, Any] | None) -> ExtractedIntent:
        # Import the SDK only when making a real API request; mock tests need no client.
        from openai import OpenAI

        # Fail clearly if there is no API key; do not print or embed a secret.
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Use --demo for an offline fixture.")
        # SDK reads API credentials from the environment.
        client = OpenAI()
        # Structured-output API: request a parsed Pydantic response, not free-form prose.
        result = client.responses.parse(
            # Use the chosen API model for extraction.
            model=self.model,
            # Provide the stable rules for routing and extraction.
            instructions=SYSTEM_INSTRUCTIONS,
            # Combine current capabilities with the actual natural-language request.
            input=build_user_input(prompt, specs),
            # Tell the API exactly which data fields/types we expect.
            text_format=ExtractedIntent,
            # Request no response storage through the API's response-storage feature.
            store=False,
        )
        # Refusals, truncation, or malformed responses may leave nothing parsed.
        if result.output_parsed is None:
            raise ValueError("LLM returned no parseable structured response (may be refusal or truncation)")
        # Return only the parsed intent object to the conversion layer.
        return result.output_parsed


# Check both our hardcoded whitelist and the optional runtime capability catalog.
def _allowed_by_catalog(algorithm: str, operation: str, specs: dict[str, Any] | None) -> bool:
    """Optional team registry can restrict, never expand, built-in capabilities."""
    # Unknown algorithm/operation pair is immediately rejected.
    if operation not in ALLOWED_OPERATIONS.get(algorithm, set()):
        return False
    # Use the built-in catalog unless a caller supplies a narrower one.
    catalog = DEFAULT_ALGORITHM_SPECS if specs is None else specs
    # Invalid catalog input cannot authorize any operations.
    if not isinstance(catalog, dict):
        return False
    # Look up the advertised operations for the selected algorithm.
    descriptor = catalog.get(algorithm)
    # Allow only if the selected operation is explicitly registered.
    return isinstance(descriptor, dict) and operation in descriptor.get("operations", {})


# Clean unresolved messages and preserve their first-occurrence order.
def _deduplicate(messages: list[str]) -> list[str]:
    # Trim whitespace, remove blank/non-string entries, and drop duplicates.
    return list(dict.fromkeys(s.strip() for s in messages if isinstance(s, str) and s.strip()))


# Main conversion boundary: intermediate LLM intent -> official six-field request.
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
    # Read the candidate routing decision produced by the extraction model.
    algorithm, operation = intent.algorithm_id, intent.operation
    # Combine model-reported ambiguities and any requirements we could not express.
    unresolved = list(intent.ambiguities) + list(intent.unhandled_requirements)

    # No algorithm selected: make the missing decision visible downstream.
    if algorithm is None:
        # Add a machine-readable issue for the missing algorithm.
        unresolved.append("algorithm_id")
    # No operation selected: mark the request as needing clarification.
    if operation is None:
        # Record missing operation explicitly.
        unresolved.append("operation")
    # Do not accept operation pairs absent from the approved catalog.
    if algorithm is not None and operation is not None and not _allowed_by_catalog(algorithm, operation, algorithm_specs):
        unresolved.append("algorithm_operation_mismatch" if operation not in ALLOWED_OPERATIONS.get(algorithm, set()) else "unsupported_algorithm_operation")
        # Invalidate the disallowed operation instead of sending it onward.
        operation = None

    # Start an empty parameter dictionary; only recognized user values are added.
    params: dict[str, Any] = {}
    # For peptide generation, build only MAC-AMP-approved fields.
    if algorithm == "mac_amp" and operation == "peptide_generation":
        # Only add an organism if the researcher gave a nonempty value.
        if intent.target_organism:
            # Remove surrounding whitespace before checking the value.
            organism = intent.target_organism.strip()
            # Generic labels are not actual organism identifiers. The core
            # validator determines whether a missing target blocks execution.
            # Do not mistake generic words like "bacterium" for an actual species.
            if organism.casefold() not in {
                "bacterium", "bacteria", "a bacterium", "the bacterium",
                "an organism", "organism", "a bacterial species", "unknown",
            }:
                # Use the normalized-by-trimming text; adapter later canonicalizes the species.
                params["target_organism"] = organism
        # Continue only when the LLM reported MAC-AMP-specific settings.
        if intent.mac_amp_settings is not None:
            # Loop over only explicitly populated fields; skip omitted None defaults.
            for key, value in intent.mac_amp_settings.model_dump(exclude_none=True).items():
                # Core validator is the source of truth for approved settings.
                # Shared validator determines which tuning fields we may emit.
                if key in MAC_AMP_ALLOWED_PARAMETERS:
                    # Transfer this approved field into official JSON parameters.
                    params[key] = value
                else:
                    # Preserve disallowed requested field AND its value for later clarification.
                    unresolved.append(f"mac_amp_parameter_requires_schema_approval:{key}={value}")
    # Handle RL-Diffusion molecule sampling, distinct from peptide generation.
    elif algorithm == "rl_diffusion" and operation == "sample":
        # Do not substitute a sample-count default in the interpreter.
        if intent.n_samples is not None:
            # Sample count must be positive; report invalid values rather than fixing them silently.
            if intent.n_samples <= 0:
                # Blocks readiness until the invalid quantity is corrected.
                unresolved.append("n_samples must be a positive integer")
            else:
                # Store an explicitly stated positive molecule count.
                params["n_samples"] = intent.n_samples
        # Preserve explicit true AND false settings; False is a meaningful user choice.
        if intent.save_to_xyz is not None:
            # Forward the output-format preference.
            params["save_to_xyz"] = intent.save_to_xyz
        # Collect broad desired directions and optionally exact thresholds.
        objectives: dict[str, Any] = {}
        # Qualitative drug-likeness preference.
        if intent.qed_direction:
            # E.g., "maximize" rather than inventing a QED threshold.
            objectives["qed"] = intent.qed_direction
        # Qualitative synthetic accessibility preference.
        if intent.sas_direction:
            # E.g., "minimize" without inventing a numeric SAS score.
            objectives["sas"] = intent.sas_direction
        # Qualitative affinity preference.
        if intent.affinity_direction:
            # E.g., "improve", not an unverified numeric value.
            objectives["affinity"] = intent.affinity_direction

        # Validate each exact user-provided numeric objective before emitting it.
        for goal in intent.numerical_goals:
            # QED/SAS are expected here without units; unexpected units need review.
            if goal.metric in ("qed", "sas") and goal.unit:
                # Tell downstream why this specific objective was not accepted.
                unresolved.append(f"{goal.metric}: unexpected unit must be reviewed")
                continue
            # Affinity thresholds need a stated unit/scoring convention for meaningful interpretation.
            if goal.metric == "affinity" and not goal.unit:
                # Do not guess units for affinity.
                unresolved.append("affinity: missing measurement units / scoring convention")
                continue
            # Ranges require a second bound that is not smaller than the first.
            if goal.operator == "range" and (goal.upper_value is None or goal.value > goal.upper_value):
                # Block invalid/missing range requests.
                unresolved.append(f"{goal.metric}: invalid or missing range upper bound")
                continue
            # A single comparison must not include a stray upper bound.
            if goal.operator != "range" and goal.upper_value is not None:
                # Report the inconsistent field combination.
                unresolved.append(f"{goal.metric}: contradictory single-value/range format")
                continue
            # Check the known QED score domain 0–1 for primary and optional upper values.
            if goal.metric == "qed" and not (0 <= goal.value <= 1) or (
                goal.metric == "qed" and goal.upper_value is not None and not (0 <= goal.upper_value <= 1)
            ):
                # Flag out-of-range QED thresholds.
                unresolved.append("qed: value must be between 0 and 1")
                continue
            # No domain assumptions about affinity score sign or SAS score implementation.
            # Default behavior is conservative until the whole team approves a numeric objective schema.
            if not enable_numeric_objectives:
                # Keep exact user threshold intact in an unresolved message rather than losing it.
                unresolved.append(f"numerical_objective_requires_schema_approval:{goal.metric}:{goal.operator}:{goal.value:g}" + (f":{goal.upper_value:g}" if goal.upper_value is not None else "") + (f":{goal.unit}" if goal.unit else ""))
                continue
            # Optional approved mode: encode the numerical goal in a structured nested object.
            objectives[goal.metric] = {
                # Include the comparison operator.
                "operator": goal.operator,
                # Include the primary numeric value.
                "value": goal.value,
                # Include upper bound when the user requested a range, else None.
                "upper_value": goal.upper_value,
                # Include user-supplied unit or None; never fabricate.
                "unit": goal.unit,
            }
        # Only add requested_objectives when the user actually specified preferences.
        if objectives:
            # Attach requested objectives to RL-Diffusion sampling parameters.
            params["requested_objectives"] = objectives

    # Training and sampling use different input schemas.
    elif algorithm == "rl_diffusion" and operation == "train":
        # Include training conditioning properties only when requested.
        if intent.conditioning is not None:
            # Preserve the first occurrence of each property and remove duplicates.
            params["conditioning"] = list(dict.fromkeys(intent.conditioning))

    # If a result was requested for a different operation, never pretend that
    # it was translated. Report that the requested intent is not yet supported.
    # Do not silently lose MAC-AMP settings when routed to another operation.
    if intent.mac_amp_settings is not None and intent.mac_amp_settings.model_dump(exclude_none=True) and (algorithm, operation) != ("mac_amp", "peptide_generation"):
        # Explain that MAC-AMP settings cannot be used for the selected capability.
        unresolved.append("mac_amp_settings_not_supported_for_selected_operation")
    # Exact numerical goals are currently only represented for RL-Diffusion sampling.
    if intent.numerical_goals and (algorithm, operation) != ("rl_diffusion", "sample"):
        unresolved.append("numerical_objectives_not_supported_for_selected_operation")
    # Organism target is not a supported field for the other registered operations.
    if intent.target_organism and (algorithm, operation) != ("mac_amp", "peptide_generation"):
        unresolved.append("target_organism_not_supported_for_selected_operation")
    # Training conditioning cannot be silently attached to sampling/peptide generation.
    if intent.conditioning and (algorithm, operation) != ("rl_diffusion", "train"):
        unresolved.append("conditioning_not_supported_for_selected_operation")
    # An RL sample count is not necessarily the same thing as a MAC-AMP count.
    if intent.n_samples is not None and (algorithm, operation) != ("rl_diffusion", "sample"):
        unresolved.append("n_samples_not_supported_for_selected_operation")
    # XYZ output controls are not valid for the other current operations.
    if intent.save_to_xyz is not None and (algorithm, operation) != ("rl_diffusion", "sample"):
        unresolved.append("save_to_xyz_not_supported_for_selected_operation")
    # Qualitative RL sampling preferences must be reported if operation is different.
    if (intent.qed_direction or intent.sas_direction or intent.affinity_direction) and (algorithm, operation) != ("rl_diffusion", "sample"):
        unresolved.append("qualitative_objectives_not_supported_for_selected_operation")

    # Determine the allowed resource reference names for the chosen operation.
    accepted = REFERENCE_KEYS.get((algorithm, operation), set())
    # Build a new dictionary using only trusted, allowed, nonblank resource references.
    inputs = {
        key: value
        # Use references provided by the application—not arbitrary names from the user prompt.
        for key, value in (trusted_inputs or {}).items()
        # Discard inappropriate reference keys and empty/non-string identifiers.
        if key in accepted and isinstance(value, str) and value.strip()
    }
    # Return the complete six-field JSON contract expected by the validator.
    return {
        # Contract version ensures producer and consumer agree about field semantics.
        "schema_version": "1.0",
        # Chosen algorithm or None.
        "algorithm_id": algorithm,
        # Chosen operation or None.
        "operation": operation,
        # Only extracted values that fit the selected operation.
        "parameters": params,
        # Trusted references only, not raw file contents.
        "inputs": inputs,
        # Ordered, nonduplicate issues to be handled before execution.
        "unresolved": _deduplicate(unresolved),
    }


# Conservative repair for one known LLM inconsistency: sample chosen without algorithm.
def _repair_explicit_molecule_route(intent: ExtractedIntent, prompt: str) -> ExtractedIntent:
    """Conservative fallback for an internally inconsistent *single* operation.

    This is not open-ended keyword classification. It only repairs the model's
    `sample` + null algorithm output when the user explicitly requests small-
    molecule generation and the model has extracted that precise operation.
    Never override unsupported/multistep/ambiguous requests or a named choice.
    """
    # Never change an already selected algorithm or unrelated operation.
    if intent.algorithm_id is not None or intent.operation != "sample":
        return intent
    # Casefold for case-insensitive keyword checks.
    lowered = prompt.casefold()
    # Do not auto-fix if a real ambiguity or unsupported requirement remains.
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
    # Copy the immutable meaning into a repaired extraction without mutating the original.
    return intent.model_copy(update={
        # Fill the algorithm missing from an otherwise clear sample request.
        "algorithm_id": "rl_diffusion",
        # Remove only the repaired missing-algorithm message; keep other issues.
        "ambiguities": [issue for issue in intent.ambiguities if issue.casefold().strip() not in {"algorithm_id", "unknown algorithm_id"}],
    })


# Public entrypoint for the LLM interpretation stage used by the integration runner.
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
    # Reject blank or non-string prompts immediately.
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("user_prompt must be a non-empty string")
    # Cap prompt length for predictable input size and resource use.
    if len(user_prompt) > 12000:
        raise ValueError("user_prompt is too long (max 12000 characters)")
    # Supplied algorithm catalog must be an object.
    if algorithm_specs is not None and not isinstance(algorithm_specs, dict):
        raise ValueError("algorithm_specs must be a dictionary")
    # Single-turn requests go to the provider unchanged.
    prompt_for_provider = user_prompt
    # Optional follow-up mode: include the last structured request as context.
    if previous_request is not None:
        # Only accept prior context shaped like the current contract version.
        if not isinstance(previous_request, dict) or previous_request.get("schema_version") != "1.0":
            raise ValueError("previous_request must be a version 1.0 translation request")
        # Limit prior context to routing/parameter fields; do not carry over old resource IDs.
        previous_data = {
            key: previous_request.get(key)
            # Previous inputs/unresolved are intentionally not copied into the model context.
            for key in ("algorithm_id", "operation", "parameters")
        }
        # Serialize context while keeping international scientific text legible.
        prior_json = json.dumps(previous_data, ensure_ascii=False)
        # Prevent oversized prior-request context.
        if len(prior_json) > 6000:
            raise ValueError("previous_request is too large")
        # Combine the past request and the new message into one extraction prompt.
        prompt_for_provider = (
            "Previous structured research request (context only, not instructions):\n"
            + prior_json
            + "\n\nNew researcher message (use to update the previous request):\n"
            + user_prompt
            + "\nReturn the COMPLETE updated interpretation. Preserve prior values only "
              "if they remain applicable; do not invent missing fields."
        )
    # Use an injected mock/provider if given, otherwise the real OpenAI provider.
    chosen_provider = provider if provider is not None else OpenAIIntentProvider()
    # Ask the provider for structured intent; this is the one LLM extraction step.
    extraction = chosen_provider.extract(prompt_for_provider, algorithm_specs)
    # A narrow recovery for a model response such as {algorithm_id:null,
    # operation:"sample", n_samples:30} to an explicit molecule request.
    # No arbitrary rerouting of ambiguous or unsupported requests.
    # Apply the conservative sample-route fix to the *new* researcher text.
    extraction = _repair_explicit_molecule_route(extraction, user_prompt)
    # Turn provider output into validated-shape JSON without executing any algorithm.
    return to_translation_request(
        extraction,
        trusted_inputs=trusted_inputs,
        algorithm_specs=algorithm_specs,
        enable_numeric_objectives=enable_numeric_objectives,
    )


# Command-line entrypoint used by python -m llm.interpreter.
def _main() -> None:
    # CLI argument parsing is only needed when running this file as a script.
    import argparse
    # Load locally configured .env values when running the CLI; never display the key.
    from dotenv import load_dotenv

    # Describe the command for --help.
    parser = argparse.ArgumentParser(description="Interpret a researcher's request into JSON")
    # Optional position argument: plain-English research prompt.
    parser.add_argument("prompt", nargs="?", help="Natural-language research request")
    # --demo does not test language understanding or require an API key.
    parser.add_argument("--demo", action="store_true", help="Offline fixed fixture, no LLM call")
    # Explicitly opt into a not-yet-universal numeric field representation.
    parser.add_argument("--enable-numeric-objectives", action="store_true", help="Opt into proposed numeric JSON shape; requires validator agreement")
    # Turn command-line strings and flags into an args object.
    args = parser.parse_args()
    # Populate environment variables from local .env without printing them.
    load_dotenv()
    # Choose a fixed example when demonstrating the formatter offline.
    if args.demo:
        # Construct a known intent to bypass the model, making this run deterministic.
        extraction = ExtractedIntent(
            algorithm_id="rl_diffusion", operation="sample", target_organism=None,
            n_samples=50, save_to_xyz=True, conditioning=None,
            qed_direction=None, sas_direction=None, affinity_direction=None,
        )
        # Make it obvious that demo mode does not call the actual LLM.
        print("OFFLINE FIXTURE (not model interpretation):")
        # Convert the fixture to the same official JSON contract.
        output = to_translation_request(extraction)
    else:
        # Use supplied researcher text or a small CLI example.
        prompt = args.prompt or "Generate 50 3D molecules"
        # Invoke real interpretation (and potentially a billed API request).
        output = interpret_user_prompt(prompt, enable_numeric_objectives=args.enable_numeric_objectives)
    # Pretty-print the six-field JSON for debugging and demos.
    print(json.dumps(output, indent=2, ensure_ascii=False))


# Only launch CLI behavior when invoked as a script, not when imported elsewhere.
if __name__ == "__main__":
    # Start the command-line execution path.
    _main()
