"""
MAC-AMP Input Adapter (`SpecKitTranslator` / `MacAmpAdapter`)
=============================================================
Fits into the unified pipeline:
  User Prompt -> LLM Interpreter -> Core Validator -> MAC-AMP Adapter -> MAC-AMP Configuration

Responsibilities:
  1. Accept the validated 6-field JSON request (`schema_version`, `algorithm_id`, `operation`,
     `parameters`, `inputs`, `unresolved`) from the Core Validator without making any OpenAI API calls.
  2. Map JSON `parameters` (such as `target_organism` -> `target_bacterium` and any supported
     MAC-AMP hyperparameters) and `inputs` (`peptide_dataset_ref`, `model_ref`, `workflow_config_ref`)
     to the Pydantic `MacAmpConfig` model.
  3. Apply Spec-Kit defaults from `.specify/templates/config-template.json`, verify supported
     bacteria, and validate that required dataset paths, model files, and resources exist.
  4. Generate the ready-to-run MAC-AMP_v1 Hydra `config.yaml` payload and Spec-Kit artifacts.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from pydantic import ValidationError

try:
    from .schema import BASE_DIR, MacAmpConfig, SUPPORTED_BACTERIA, TranslationResult
except ImportError:
    from schema import BASE_DIR, MacAmpConfig, SUPPORTED_BACTERIA, TranslationResult

CONFIG_TEMPLATE_PATH = BASE_DIR / ".specify" / "templates" / "config-template.json"

# Maps upstream JSON parameter names to MacAmpConfig field names
PARAMETER_ALIASES: Dict[str, str] = {
    "target_organism": "target_bacterium",
    "target_bacterium": "target_bacterium",
    "bacterium_code": "bacterium_code",
    "warmup_dir": "warmup_dir",
    "ref_pdb_path": "ref_pdb_path",
    "model_path": "model_path",
    "task": "task",
    "env_name": "env_name",
    "model": "model",
    "temperature": "temperature",
    "suffix": "suffix",
    "num_samples": "num_samples",
    "n_samples": "num_samples",
    "sample": "sample",
    "max_seq_len": "max_seq_len",
    "max_length": "max_length",
    "batch_size": "batch_size",
    "real_epoch": "real_epoch",
    "epochs": "real_epoch",
    "sand_box_epoch": "sand_box_epoch",
    "real_train_stage_iteration": "real_train_stage_iteration",
    "sandbox_iteration": "sandbox_iteration",
    "reward_sample_num": "reward_sample_num",
    "ablation_mode": "ablation_mode",
    "skip_independent_reviewer": "skip_independent_reviewer",
    "drift_wordlist": "drift_wordlist",
    "experiment": "experiment",
    "default_sign": "default_sign",
    "log_path": "log_path",
}

ALLOWED_INPUT_REFS = {"peptide_dataset_ref", "model_ref", "workflow_config_ref"}


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
        f'model_path: "{payload["model_path"]}"',
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


class SpecKitTranslator:
    """
    Translates validated 6-field MAC-AMP JSON requests into validated MAC-AMP_v1 Hydra YAML configs.
    Does not call any external LLM API.
    """

    def __init__(
        self,
        resource_root: Optional[Union[str, Path]] = None,
        resource_registry: Optional[Mapping[str, Union[str, Path]]] = None,
        verify_resources: bool = False,
        **_: Any,
    ):
        """
        Args:
            resource_root: Optional base directory used to resolve relative dataset, model,
                and workflow config paths when verifying resource existence on disk.
            resource_registry: Optional mapping from logical reference IDs (e.g.
                `"upload_peptides_001"`, `"model_amp_001"`) to file/directory paths.
            verify_resources: When True (or when `resource_root` is provided), checks that
                resolved dataset paths, model files/directories, and workflow configs
                physically exist on disk.
        """
        self.resource_root: Optional[Path] = Path(resource_root) if resource_root is not None else None
        self.resource_registry: Dict[str, str] = {
            k: str(v) for k, v in (resource_registry or {}).items()
        }
        self.verify_resources: bool = verify_resources or (self.resource_root is not None)

    def _parse_upstream_json(
        self, upstream_input: Union[str, Path, Mapping[str, Any]]
    ) -> Dict[str, Any]:
        """Accepts a dictionary, a JSON string, or a path to a .json file."""
        if isinstance(upstream_input, Mapping):
            return dict(upstream_input)

        if isinstance(upstream_input, Path):
            if not upstream_input.exists() or not upstream_input.is_file():
                raise FileNotFoundError(f"Input JSON file not found: {upstream_input}")
            return json.loads(upstream_input.read_text(encoding="utf-8-sig"))

        if isinstance(upstream_input, str):
            text = upstream_input.lstrip("\ufeff").strip()
            if not text:
                raise ValueError("Input JSON cannot be empty.")

            # If it starts with '{', parse directly as a JSON object string
            if text.startswith("{"):
                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    raise ValueError("Input JSON must decode to a JSON object.")
                return parsed

            # If it points to an existing .json file on disk, load it
            candidate_path = Path(text)
            if candidate_path.suffix.lower() == ".json":
                if not candidate_path.exists() or not candidate_path.is_file():
                    raise FileNotFoundError(f"Input JSON file not found: {candidate_path}")
                parsed = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
                if not isinstance(parsed, dict):
                    raise ValueError("Input JSON file must contain a JSON object.")
                return parsed

            raise ValueError(
                "MAC-AMP Adapter expects a structured 6-field JSON request (dict, JSON string, "
                "or path to a .json file), not unstructured natural language."
            )

        raise TypeError(
            f"Unsupported upstream_input type: {type(upstream_input).__name__}. "
            "Expected dict, JSON string, or Path."
        )

    def _resolve_resource_path(self, ref_or_path: str, default_suffix: str = "") -> Path:
        """Maps a logical reference ID or file path to a concrete filesystem Path."""
        mapped = self.resource_registry.get(ref_or_path, ref_or_path)
        p = Path(mapped)
        if not p.is_absolute() and self.resource_root is not None:
            # Handle relative paths starting with ../ or /data/ inside a resource_root sandbox
            cleaned = str(mapped).replace("\\", "/")
            while cleaned.startswith("../"):
                cleaned = cleaned[3:]
            cleaned = cleaned.lstrip("/")
            candidate = self.resource_root / cleaned
            if candidate.exists():
                return candidate
            if default_suffix and not cleaned.lower().endswith(default_suffix):
                with_suffix = self.resource_root / f"{cleaned}{default_suffix}"
                if with_suffix.exists():
                    return with_suffix
            return candidate
        if str(mapped).startswith("/data/") and self.resource_root is not None:
            return self.resource_root / str(mapped).lstrip("/")
        return p

    def _load_workflow_config_overrides(self, workflow_ref: str) -> Dict[str, Any]:
        """Loads optional MAC-AMP parameter overrides from a workflow_config_ref JSON/YAML file."""
        path = self._resolve_resource_path(workflow_ref)
        if not path.exists() or not path.is_file():
            if self.verify_resources or Path(workflow_ref).suffix.lower() in {".json", ".yaml", ".yml"}:
                raise FileNotFoundError(
                    f"Required workflow_config_ref file does not exist: '{workflow_ref}' (resolved to '{path}')"
                )
            return {}

        raw_text = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() == ".json":
            loaded = json.loads(raw_text)
        else:
            try:
                import yaml  # type: ignore
                loaded = yaml.safe_load(raw_text)
            except ImportError:
                loaded = {}
                for line in raw_text.splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or stripped.startswith("-") or ":" not in stripped:
                        continue
                    k, v = stripped.split(":", 1)
                    k = k.strip()
                    v = v.split("#", 1)[0].strip().strip('"').strip("'")
                    if v:
                        try:
                            loaded[k] = json.loads(v)
                        except Exception:
                            loaded[k] = v

        if not isinstance(loaded, dict):
            raise ValueError(f"workflow_config_ref '{workflow_ref}' must contain a key-value mapping.")

        overrides: Dict[str, Any] = {}
        for key, value in loaded.items():
            if key in ("defaults", "hydra"):
                continue
            if key in PARAMETER_ALIASES:
                overrides[PARAMETER_ALIASES[key]] = value
            else:
                raise ValueError(f"Unsupported MAC-AMP parameter '{key}' in workflow_config_ref.")
        return overrides

    def _verify_required_resources(
        self,
        config: MacAmpConfig,
        inputs: Mapping[str, Any],
    ) -> List[str]:
        """
        Verifies that required dataset paths, model files, and input references exist.
        Returns a list of missing resource identifiers/messages.
        """
        missing: List[str] = []

        # 1. Check workflow_config_ref if provided
        workflow_ref = inputs.get("workflow_config_ref")
        if workflow_ref:
            wf_path = self._resolve_resource_path(str(workflow_ref))
            if (self.verify_resources or wf_path.suffix.lower() in {".json", ".yaml", ".yml"}) and not wf_path.exists():
                missing.append(f"workflow_config_ref ({workflow_ref})")

        # 2. Check peptide dataset / warmup_dir
        dataset_ref = inputs.get("peptide_dataset_ref")
        if dataset_ref:
            ds_path = self._resolve_resource_path(str(dataset_ref), default_suffix=".csv")
            if (self.verify_resources or str(dataset_ref).lower().endswith(".csv")) and not ds_path.is_file():
                missing.append(f"peptide_dataset_ref ({dataset_ref})")
        elif self.verify_resources and config.warmup_dir:
            warmup_path = self._resolve_resource_path(config.warmup_dir)
            if not warmup_path.is_file():
                missing.append(f"warmup_dir ({config.warmup_dir})")

        # 3. Check model checkpoint directory / model_ref
        model_ref = inputs.get("model_ref")
        if model_ref:
            m_path = self._resolve_resource_path(str(model_ref))
            if self.verify_resources and not m_path.exists():
                missing.append(f"model_ref ({model_ref})")
        elif self.verify_resources and config.model_path:
            mp = self._resolve_resource_path(config.model_path)
            if not mp.exists():
                missing.append(f"model_path ({config.model_path})")

        # 4. Check reference PDB directory when full resource verification is active
        if self.verify_resources and config.ref_pdb_path:
            pdb_path = self._resolve_resource_path(config.ref_pdb_path)
            if not pdb_path.exists():
                missing.append(f"ref_pdb_path ({config.ref_pdb_path})")

        return missing

    def translate(
        self, upstream_input: Union[str, Path, Mapping[str, Any]]
    ) -> TranslationResult:
        """
        Validates a 6-field TranslationRequest JSON and maps it into a `TranslationResult`
        containing the validated `MacAmpConfig`. Makes zero external API calls.
        """
        data = self._parse_upstream_json(upstream_input)

        # Detect if a raw MacAmpConfig payload was passed directly or a 6-field TranslationRequest
        is_six_field = any(
            k in data
            for k in ("schema_version", "algorithm_id", "operation", "parameters", "inputs", "unresolved")
        )

        if is_six_field:
            schema_version = data.get("schema_version", "1.0")
            if schema_version != "1.0":
                return TranslationResult(
                    status="clarification_required",
                    config=None,
                    missing_or_ambiguous_fields=["schema_version"],
                    clarification_message=f"Unsupported schema_version '{schema_version}'. Expected '1.0'.",
                )

            algorithm_id = data.get("algorithm_id")
            if algorithm_id != "mac_amp":
                return TranslationResult(
                    status="clarification_required",
                    config=None,
                    missing_or_ambiguous_fields=["algorithm_id"],
                    clarification_message=(
                        f"MAC-AMP Adapter only accepts algorithm_id='mac_amp', got '{algorithm_id}'."
                    ),
                )

            operation = data.get("operation")
            if operation != "peptide_generation":
                return TranslationResult(
                    status="clarification_required",
                    config=None,
                    missing_or_ambiguous_fields=["operation"],
                    clarification_message=(
                        f"MAC-AMP Adapter only supports operation='peptide_generation', got '{operation}'."
                    ),
                )

            unresolved = data.get("unresolved") or []
            if not isinstance(unresolved, list):
                raise ValueError("'unresolved' must be a list of strings.")
            if unresolved:
                return TranslationResult(
                    status="clarification_required",
                    config=None,
                    missing_or_ambiguous_fields=[str(item) for item in unresolved],
                    clarification_message=(
                        f"Cannot generate MAC-AMP configuration while request has unresolved items: {unresolved}"
                    ),
                )

            raw_params = data.get("parameters") or {}
            raw_inputs = data.get("inputs") or {}
            if not isinstance(raw_params, Mapping):
                raise ValueError("'parameters' must be a JSON object.")
            if not isinstance(raw_inputs, Mapping):
                raise ValueError("'inputs' must be a JSON object.")
        else:
            raw_params = data
            raw_inputs = {}

        # Validate input reference keys
        unknown_inputs = set(raw_inputs.keys()) - ALLOWED_INPUT_REFS
        if unknown_inputs:
            return TranslationResult(
                status="clarification_required",
                config=None,
                missing_or_ambiguous_fields=sorted(unknown_inputs),
                clarification_message=(
                    f"Unsupported input reference keys for MAC-AMP: {sorted(unknown_inputs)}. "
                    f"Allowed inputs are: {sorted(ALLOWED_INPUT_REFS)}."
                ),
            )

        # Start with optional workflow_config_ref base overrides if supplied
        mapped_kwargs: Dict[str, Any] = {}
        workflow_ref = raw_inputs.get("workflow_config_ref")
        if workflow_ref and isinstance(workflow_ref, str) and workflow_ref.strip():
            try:
                mapped_kwargs.update(self._load_workflow_config_overrides(workflow_ref.strip()))
            except (FileNotFoundError, ValueError) as exc:
                return TranslationResult(
                    status="clarification_required",
                    config=None,
                    missing_or_ambiguous_fields=["workflow_config_ref"],
                    clarification_message=str(exc),
                )

        # Map parameters (including parameters.target_organism -> target_bacterium)
        unsupported_params: List[str] = []
        for key, value in raw_params.items():
            if value is None:
                continue
            target_field = PARAMETER_ALIASES.get(key)
            if target_field is None:
                unsupported_params.append(key)
            else:
                mapped_kwargs[target_field] = value

        if unsupported_params:
            return TranslationResult(
                status="clarification_required",
                config=None,
                missing_or_ambiguous_fields=sorted(unsupported_params),
                clarification_message=(
                    f"Unsupported MAC-AMP parameter(s): {sorted(unsupported_params)}."
                ),
            )

        # Map inputs.peptide_dataset_ref -> warmup_dir if warmup_dir not explicitly overridden
        dataset_ref = raw_inputs.get("peptide_dataset_ref")
        if dataset_ref and isinstance(dataset_ref, str) and dataset_ref.strip():
            ref_str = dataset_ref.strip()
            resolved_ds = self.resource_registry.get(ref_str, ref_str)
            if not str(resolved_ds).lower().endswith(".csv"):
                resolved_ds = f"{resolved_ds}.csv"
            mapped_kwargs.setdefault("warmup_dir", str(resolved_ds))

        # Map inputs.model_ref -> model_path if model_path not explicitly overridden
        model_ref = raw_inputs.get("model_ref")
        if model_ref and isinstance(model_ref, str) and model_ref.strip():
            ref_str = model_ref.strip()
            resolved_model = self.resource_registry.get(ref_str, ref_str)
            mapped_kwargs.setdefault("model_path", str(resolved_model))

        # Check required target_bacterium
        target_val = mapped_kwargs.get("target_bacterium")
        if not target_val or not str(target_val).strip():
            return TranslationResult(
                status="clarification_required",
                config=None,
                missing_or_ambiguous_fields=["target_organism"],
                clarification_message=(
                    "Missing required target organism ('parameters.target_organism'). "
                    f"Supported MAC-AMP organisms are: {', '.join(SUPPORTED_BACTERIA)}."
                ),
            )

        # Validate with MacAmpConfig (auto-resolves bacterium_code, warmup_dir, ref_pdb_path)
        try:
            config = MacAmpConfig(**mapped_kwargs)
        except ValidationError as exc:
            err_fields: List[str] = []
            messages: List[str] = []
            for err in exc.errors():
                loc = ".".join(str(part) for part in err.get("loc", ())) or "target_organism"
                if loc == "target_bacterium":
                    loc = "target_organism"
                err_fields.append(loc)
                messages.append(err.get("msg", str(exc)))
            return TranslationResult(
                status="clarification_required",
                config=None,
                missing_or_ambiguous_fields=list(dict.fromkeys(err_fields)),
                clarification_message="; ".join(messages),
            )

        # Check required resources on disk when configured or when explicit file paths are given
        missing_resources = self._verify_required_resources(config, raw_inputs)
        if missing_resources:
            return TranslationResult(
                status="clarification_required",
                config=None,
                missing_or_ambiguous_fields=missing_resources,
                clarification_message=(
                    f"Required MAC-AMP resource(s) do not exist: {', '.join(missing_resources)}."
                ),
            )

        return TranslationResult(status="success", config=config)

    # Downstream Layer Handoff (`translate_for_downstream`)
    def translate_for_downstream(
        self,
        upstream_input: Union[str, Path, Mapping[str, Any]],
        save_to_specs: bool = True,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        End-to-end entrypoint called by the platform pipeline or CLI:
          1. Calls `translate(upstream_input)` on the validated 6-field JSON.
          2. Raises a clear ValueError if `status != "success"` (e.g., missing target_organism,
             unsupported bacterium, or missing dataset/model resource).
          3. Converts the validated `MacAmpConfig` into a Hydra `config.yaml` string.
          4. Optionally saves `config.yaml` and `spec.md` into `specs/<NNN-bacterium>/`.
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
            source_summary = (
                json.dumps(dict(upstream_input), ensure_ascii=False)
                if isinstance(upstream_input, Mapping)
                else str(upstream_input)
            )
            self._persist_spec_kit_artifacts(source_summary, payload, yaml_content)

        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(yaml_content, encoding="utf-8")

        return yaml_content

    # Spec-Kit Artifact Persistence (`specs/<NNN-bacterium>/`)
    def _persist_spec_kit_artifacts(
        self, source_request: str, payload: Dict[str, Any], yaml_content: str
    ) -> Path:
        """
        Creates a sequentially numbered folder under `specs/` (e.g., `specs/005-escherichia-coli/`)
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
        spec_md = (
            f"# Run Specification: {run_dir.name}\n\n"
            f"- **Source Validated JSON**: `{source_request}`\n"
            f"- **Translation Mode**: `Structured JSON Adapter (0 LLM Calls)`\n"
            f"- **Validation Status**: `VALID (MAC-AMP_v1 Hydra YAML)`\n\n"
            f"## Target Bacterium & Auto-Resolved MAC-AMP_v1 Paths\n"
            f"- **`target_bacterium`**: `{payload['target_bacterium']}`\n"
            f"- **`bacterium_code`**: `{payload['bacterium_code']}`\n"
            f"- **`warmup_dir`**: `{payload['warmup_dir']}`\n"
            f"- **`ref_pdb_path`**: `{payload['ref_pdb_path']}`\n"
            f"- **`model_path`**: `{payload['model_path']}`\n"
        )
        (run_dir / "spec.md").write_text(spec_md, encoding="utf-8")
        return run_dir


# Alias for intuitive naming in the pipeline
MacAmpAdapter = SpecKitTranslator


# Command-Line & Stdin Pipe Entrypoint (`main`)
def main() -> None:
    parser = argparse.ArgumentParser(
        description="MAC-AMP Input Adapter: Translates validated 6-field JSON into MAC-AMP_v1 Hydra config.yaml."
    )
    parser.add_argument(
        "json_input",
        nargs="?",
        help="Validated 6-field JSON string or path to a .json file (or pass via stdin).",
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
        "--resource-root",
        type=str,
        default=None,
        help="Optional root directory to verify that dataset, model, and PDB resources exist on disk.",
    )
    parser.add_argument(
        "--verify-resources",
        action="store_true",
        help="Verify that all referenced dataset paths, model files, and resources exist on disk.",
    )
    args = parser.parse_args()

    upstream_text = args.json_input or sys.stdin.read().strip()
    if not upstream_text:
        print("Error: No JSON input provided via argument or stdin.", file=sys.stderr)
        sys.exit(1)

    try:
        adapter = SpecKitTranslator(
            resource_root=args.resource_root,
            verify_resources=args.verify_resources,
        )
        downstream_yaml = adapter.translate_for_downstream(
            upstream_input=upstream_text,
            save_to_specs=not args.no_spec_save,
            output_path=args.output,
        )
        print(downstream_yaml)
    except Exception as exc:
        print(f"status: error\nmessage: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
