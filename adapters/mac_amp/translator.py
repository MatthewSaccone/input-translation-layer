"""
Acts as the translation component between:
  1. Upstream Layer: Provides a plain-English user prompt.
  2. Spec-Kit (.specify/): Supplies the project constitution and default configuration template.
  3. Downstream Layer (MAC-AMP_v1): Receives a ready-to-run Hydra `config.yaml` payload.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

from dotenv import load_dotenv
from openai import OpenAI

try:
    from .schema import BASE_DIR, MacAmpConfig, TranslationResult
except ImportError:
    from schema import BASE_DIR, MacAmpConfig, TranslationResult

# Load OPENAI_API_KEY, OPENAI_MODEL, and MAX_COMPLETION_TOKENS from .env
load_dotenv()

CONSTITUTION_PATH = BASE_DIR / ".specify" / "memory" / "constitution.md"
CONFIG_TEMPLATE_PATH = BASE_DIR / ".specify" / "templates" / "config-template.json"



# Hydra YAML Serializer (`_dict_to_hydra_yaml`)
# MAC-AMP_v1's main_workflow.py uses `@hydra.main(config_path="cfg", config_name="config")`.
# This helper converts our validated dictionary into the exact YAML structure expected
# by `MAC_AMP_Multi_Agent_Framework/cfg/config.yaml` (including Hydra's `defaults:` header).
def _dict_to_hydra_yaml(payload: Dict[str, Any]) -> str:
    """Formats the validated payload as a Hydra config.yaml file for MAC-AMP_v1."""
    lines = [
        "defaults:",
        "  - _self_",
        "  - env: test_generator",
        "  - override hydra/launcher: local",
        "  - override hydra/output: local",
        "",
        "hydra:",
        "  job:",
        "    chdir: True",
        "",
        "# Target Bacterium & Dataset Paths",
        f'target_bacterium: "{payload["target_bacterium"]}"',
        f'bacterium_code: "{payload["bacterium_code"]}"',
        f'warmup_dir: "{payload["warmup_dir"]}"',
        f'ref_pdb_path: "{payload["ref_pdb_path"]}"',
        "",
        "# env",
        f'task: {payload["task"]}',
        f'env_name: {payload["env_name"]}',
        "",
        "# LLM parameters",
        f'model: {payload["model"]}',
        f'temperature: {payload["temperature"]}',
        f'suffix: {payload["suffix"]}',
        "",
        "# PPO Training Parameters",
        f'num_samples: {payload["num_samples"]}',
        f'sample: {payload["sample"]}',
        f'max_seq_len: {payload["max_seq_len"]}',
        "",
        "# Control Parameters",
        f'skip_independent_reviewer: "{payload["skip_independent_reviewer"]}"',
        f'real_epoch: {payload["real_epoch"]}',
        f'sand_box_epoch: {payload["sand_box_epoch"]}',
        f'max_length: {payload["max_length"]}',
        f'drift_wordlist: "{payload["drift_wordlist"]}"',
        f'ablation_mode: {payload["ablation_mode"]}',
        f'experiment: "{payload["experiment"]}"',
        f'default_sign: {payload["default_sign"]}',
        f'log_path: {payload["log_path"]}',
        f'real_train_stage_iteration: {payload["real_train_stage_iteration"]}',
        f'sandbox_iteration: {payload["sandbox_iteration"]}',
        f'reward_sample_num: {payload["reward_sample_num"]}',
        f'batch_size: {payload["batch_size"]}',
        "",
    ]
    return "\n".join(lines)


# Core Translation Component (`SpecKitTranslator`)
class SpecKitTranslator:
    """
    Translates plain-English user requests into validated MAC-AMP_v1 Hydra YAML configs.
    """

    # Client & Cost-Control Setup 
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
        dry_run: bool = False,
    ):
        self.dry_run = dry_run
        # Default to gpt-4o-mini (~$0.0002/run) and cap output at 300 tokens
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_completion_tokens = max_completion_tokens or int(
            os.getenv("MAX_COMPLETION_TOKENS", "300")
        )

        # Only initialize the OpenAI client when not running in zero-token dry-run mode
        if not self.dry_run:
            resolved_key = api_key or os.getenv("OPENAI_API_KEY")
            if not resolved_key or resolved_key == "sk-your-openai-api-key-here":
                raise EnvironmentError(
                    "Missing OpenAI API key. Set OPENAI_API_KEY in .env, pass api_key=..., "
                    "or run with --dry-run (0 tokens)."
                )
            self.client: Optional[OpenAI] = OpenAI(api_key=resolved_key)
        else:
            self.client = None

    # Dynamic Spec-Kit System Prompt Builder 
    def _build_system_prompt(self) -> str:
        """
        Reads `.specify/memory/constitution.md` and `.specify/templates/config-template.json`
        from disk so the LLM's instructions always reflect your latest Spec-Kit rules.
        """
        constitution_text = (
            CONSTITUTION_PATH.read_text(encoding="utf-8")
            if CONSTITUTION_PATH.exists()
            else ""
        )
        template_json = (
            CONFIG_TEMPLATE_PATH.read_text(encoding="utf-8")
            if CONFIG_TEMPLATE_PATH.exists()
            else "{}"
        )
        return (
            "You are the Spec-Kit NLP-to-Configuration translation component for CLMFAP/MAC-AMP_v1.\n"
            "Follow the project Constitution and Default Config Template below strictly.\n\n"
            f"=== CONSTITUTION (.specify/memory/constitution.md) ===\n{constitution_text}\n\n"
            f"=== DEFAULT TEMPLATE (.specify/templates/config-template.json) ===\n{template_json}"
        )

    # Zero-Token Local Dry-Run Parser ---
    def _dry_run_translate(self, upstream_input: str) -> TranslationResult:
        """
        Offline regex fallback used when `--dry-run` is passed.
        Allows testing upstream -> downstream layer integration for $0.00 (0 tokens).
        """
        bacterium_match = re.search(
            r"(?:targeting|target|against|category|for)\s+([A-Za-z0-9_.\-]+(?:\s+[A-Za-z0-9_.\-]+)?)",
            upstream_input,
            re.IGNORECASE,
        )
        if not bacterium_match:
            return TranslationResult(
                status="clarification_required",
                config=None,
                missing_or_ambiguous_fields=["target_bacterium"],
                clarification_message="Could not identify target_bacterium (bacterium/strain) in prompt.",
            )

        raw_target = bacterium_match.group(1).strip()
        raw_target = re.sub(
            r"\s+(?:using|with|for|at|in|from|dataset|iterations|steps|epochs)$",
            "",
            raw_target,
            flags=re.IGNORECASE,
        ).strip()

        csv_match = re.search(r"([A-Za-z0-9_./\\\-]+\.csv)", upstream_input, re.IGNORECASE)
        epoch_match = re.search(r"(\d+)\s+(?:epochs|iterations|steps)", upstream_input, re.IGNORECASE)
        batch_match = re.search(r"batch\s+size\s+(?:of\s+)?(\d+)", upstream_input, re.IGNORECASE)
        temp_match = re.search(r"temperature\s+(?:of\s+)?([0-9]*\.?[0-9]+)", upstream_input, re.IGNORECASE)

        kwargs: Dict[str, Any] = {"target_bacterium": raw_target}
        if csv_match:
            kwargs["warmup_dir"] = csv_match.group(1)
        if epoch_match:
            kwargs["real_epoch"] = int(epoch_match.group(1))
        if batch_match:
            kwargs["batch_size"] = int(batch_match.group(1))
        if temp_match:
            kwargs["temperature"] = float(temp_match.group(1))

        config = MacAmpConfig(**kwargs)
        return TranslationResult(status="success", config=config)

    # OpenAI Structured Outputs Call (`translate`) 
    def translate(self, upstream_input: str) -> TranslationResult:
        """
        Sends the plain-English request to OpenAI with `response_format=TranslationResult`.
        OpenAI guarantees the response adheres to the `TranslationResult` Pydantic schema.
        """
        if self.dry_run:
            return self._dry_run_translate(upstream_input)

        assert self.client is not None
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            temperature=0.0,  # Deterministic extraction
            max_completion_tokens=self.max_completion_tokens,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": upstream_input},
            ],
            response_format=TranslationResult,
        )

        result = completion.choices[0].message.parsed
        if result is None:
            raise RuntimeError(
                f"Translation refused by model: {completion.choices[0].message.refusal}"
            )
        return result

    # Downstream Layer Handoff (`translate_for_downstream`)
    def translate_for_downstream(
        self,
        upstream_input: str,
        save_to_specs: bool = True,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        End-to-end entrypoint called by your application or CLI:
          1. Calls `translate(upstream_input)` to parse and validate the prompt.
          2. Raises a clear error if `target_bacterium` was missing (`clarification_required`).
          3. Converts the validated model into a Hydra `config.yaml` string.
          4. Saves `config.yaml` and `spec.md` into `specs/<NNN-bacterium>/`.
          5. Returns the Hydra YAML string ready for MAC-AMP_v1.
        """
        result = self.translate(upstream_input)

        if result.status != "success" or result.config is None:
            raise ValueError(
                f"Clarification required for {result.missing_or_ambiguous_fields}: "
                f"{result.clarification_message}"
            )

        payload = result.config.model_dump(mode="json")
        yaml_content = _dict_to_hydra_yaml(payload)

        if save_to_specs:
            self._persist_spec_kit_artifacts(upstream_input, payload, yaml_content)

        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(yaml_content, encoding="utf-8")

        return yaml_content

    # Spec-Kit Artifact Persistence (`specs/<NNN-bacterium>/`)
    def _persist_spec_kit_artifacts(
        self, upstream_input: str, payload: Dict[str, Any], yaml_content: str
    ) -> Path:
        """
        Creates a sequentially numbered folder under `specs/` (e.g., `specs/004-escherichia-coli/`)
        and writes two files:
          - `config.yaml`: The exact Hydra config file for MAC-AMP_v1.
          - `spec.md`: The Spec-Kit human-readable audit record of the run.
        """
        specs_root = BASE_DIR / "specs"
        specs_root.mkdir(parents=True, exist_ok=True)

        existing = sorted([d for d in specs_root.iterdir() if d.is_dir()])
        next_num = len(existing) + 1
        slug = re.sub(r"[^a-z0-9]+", "-", payload["target_bacterium"].lower()).strip("-")
        run_dir = specs_root / f"{next_num:03d}-{slug}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. Write Hydra config.yaml (drop-in replacement for MAC_AMP_Multi_Agent_Framework/cfg/config.yaml)
        (run_dir / "config.yaml").write_text(yaml_content, encoding="utf-8")

        # 2. Write spec.md audit log
        mode_label = "DRY-RUN (0 Tokens)" if self.dry_run else f"OpenAI ({self.model})"
        spec_md = (
            f"# Run Specification: {run_dir.name}\n\n"
            f"- **Source User Prompt**: `{upstream_input}`\n"
            f"- **Translation Mode**: `{mode_label}`\n"
            f"- **Validation Status**: `VALID (MAC-AMP_v1 Hydra YAML)`\n\n"
            f"## Target Bacterium & Auto-Resolved MAC-AMP_v1 Paths\n"
            f"- **`target_bacterium`**: `{payload['target_bacterium']}`\n"
            f"- **`bacterium_code`**: `{payload['bacterium_code']}`\n"
            f"- **`warmup_dir`**: `{payload['warmup_dir']}`\n"
            f"- **`ref_pdb_path`**: `{payload['ref_pdb_path']}`\n"
        )
        (run_dir / "spec.md").write_text(spec_md, encoding="utf-8")
        return run_dir



# Command-Line & Stdin Pipe Entrypoint (`main`)
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spec-Kit NLP-to-Configuration translation layer for CLMFAP/MAC-AMP_v1."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Plain-English input string from upstream layer (or pass via stdin).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional file path to write a copy of the Hydra config.yaml.",
    )
    parser.add_argument(
        "--no-spec-save",
        action="store_true",
        help="Do not create a new folder under specs/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run locally using 0 OpenAI tokens (no API key required) to test layer integration.",
    )
    args = parser.parse_args()

    # Accept input either as a command-line string argument or piped via stdin
    upstream_text = args.prompt or sys.stdin.read().strip()
    if not upstream_text:
        print("Error: No input provided via argument or stdin.", file=sys.stderr)
        sys.exit(1)

    try:
        translator = SpecKitTranslator(dry_run=args.dry_run)
        downstream_yaml = translator.translate_for_downstream(
            upstream_input=upstream_text,
            save_to_specs=not args.no_spec_save,
            output_path=args.output,
        )
        # Print the Hydra config.yaml content to stdout
        print(downstream_yaml)
    except Exception as exc:
        print(f"status: error\nmessage: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
