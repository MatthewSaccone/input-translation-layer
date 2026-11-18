"""
schema.py — Spec-Kit Data Contract & Pydantic Validation Schema for CLMFAP/MAC-AMP_v1
=====================================================================================
This module defines the strict Pydantic v2 data structures that validate and map
structured upstream JSON requests into the exact parameter names, types, and file
path conventions required by MAC-AMP_v1 (`MAC_AMP_Multi_Agent_Framework/cfg/config.yaml`).
"""

import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Dynamic Spec-Kit Template Loader
# Locate the package directory and the Spec-Kit default template file on disk.
# By loading defaults from `.specify/templates/config-template.json` at import
# time, `.specify/` remains the single source of truth for default values.
BASE_DIR = Path(__file__).resolve().parent
CONFIG_TEMPLATE_PATH = BASE_DIR / ".specify" / "templates" / "config-template.json"


def _load_template_defaults() -> Dict[str, Any]:
    """Reads default configuration values from Spec-Kit's config-template.json."""
    if CONFIG_TEMPLATE_PATH.exists():
        return json.loads(CONFIG_TEMPLATE_PATH.read_text(encoding="utf-8-sig"))
    return {}


_DEFAULTS = _load_template_defaults()

# Target Bacterium Normalization & Code Lookup Table
# MAC-AMP_v1 uses two representations of the target pathogen:
#   1. Canonical name with underscore (e.g., "Escherichia_coli") for warmup CSV
#      and reference PDB directory names.
#   2. Two-letter bacterium code (e.g., "EC") passed to `predict_mic()` inside
#      `evaluation_service.py`.
# This dictionary maps supported user/LLM organism names and abbreviations to (CanonicalName, Code).
BACTERIUM_CODE_MAP: Dict[str, tuple[str, str]] = {
    "escherichia_coli": ("Escherichia_coli", "EC"),
    "e_coli": ("Escherichia_coli", "EC"),
    "ecoli": ("Escherichia_coli", "EC"),
    "ec": ("Escherichia_coli", "EC"),
    "staphylococcus_aureus": ("Staphylococcus_aureus", "SA"),
    "s_aureus": ("Staphylococcus_aureus", "SA"),
    "saureus": ("Staphylococcus_aureus", "SA"),
    "mrsa": ("Staphylococcus_aureus", "SA"),
    "sa": ("Staphylococcus_aureus", "SA"),
    "pseudomonas_aeruginosa": ("Pseudomonas_aeruginosa", "PA"),
    "p_aeruginosa": ("Pseudomonas_aeruginosa", "PA"),
    "paeruginosa": ("Pseudomonas_aeruginosa", "PA"),
    "pa": ("Pseudomonas_aeruginosa", "PA"),
    "klebsiella_pneumoniae": ("Klebsiella_pneumoniae", "KP"),
    "k_pneumoniae": ("Klebsiella_pneumoniae", "KP"),
    "kpneumoniae": ("Klebsiella_pneumoniae", "KP"),
    "kp": ("Klebsiella_pneumoniae", "KP"),
    "enterococcus_faecium": ("Enterococcus_faecium", "EF"),
    "e_faecium": ("Enterococcus_faecium", "EF"),
    "efaecium": ("Enterococcus_faecium", "EF"),
    "ef": ("Enterococcus_faecium", "EF"),
}

SUPPORTED_BACTERIA: tuple[str, ...] = tuple(
    dict.fromkeys(canonical for canonical, _ in BACTERIUM_CODE_MAP.values())
)


# Allowed Peer-Review Experiment Modes
# Matches the `choices` list in `run_ppo_training.py` (--experiment flag),
# controlling whether MAC-AMP's peer-review module uses OpenAI API or local
# Ollama models (Llama 3.1 / Qwen 2.5).
class ExperimentType(str, Enum):
    BASELINE_API = "baseline_api"
    ALL_LLAMA_LOCAL = "all_llama_local"
    ALL_QWEN_LOCAL = "all_qwen_local"
    QWEN_REVIEWERS_GPT5_AC = "qwen_reviewers_gpt5_ac"
    API_REVIEWERS_QWEN_AC = "api_reviewers_qwen_ac"


# Core MAC-AMP_v1 Configuration Schema (`MacAmpConfig`)
class MacAmpConfig(BaseModel):
    """
    Exact configuration schema matching CLMFAP/MAC-AMP_v1:
      - MAC_AMP_Multi_Agent_Framework/cfg/config.yaml
      - MAC_AMP_Multi_Agent_Framework/main_workflow.py
      - MAC_AMP_Multi_Agent_Framework/code/run_ppo_training.py
      - MAC_AMP_Multi_Agent_Framework/code/evaluation_service.py
    """
    # Forbid any extra/unsupported fields not explicitly defined in this class
    model_config = ConfigDict(extra="forbid")

    # --- Section 4A: Target Bacterium & Auto-Resolved Dataset / Model Paths
    target_bacterium: str = Field(
        ...,
        min_length=1,
        description=(
            "Target bacterium species/strain requested by the user "
            "(e.g., 'Escherichia_coli', 'Staphylococcus_aureus', 'Pseudomonas_aeruginosa')."
        ),
    )
    bacterium_code: str = Field(
        default="EC",
        description=(
            "Two-letter bacterium code passed to predict_mic() in evaluation_service.py "
            "(e.g., 'EC' for Escherichia coli, 'SA' for Staphylococcus aureus, 'PA' for Pseudomonas aeruginosa)."
        ),
    )
    warmup_dir: Optional[str] = Field(
        default=None,
        description=(
            "Path to the reference warmup CSV file containing AMP sequences and MIC values "
            "(passed as --warmup_dir to run_ppo_training.py). Leave None/null if the user did not "
            "specify a custom CSV path so it auto-resolves to "
            "'../Generation_Module/data/warmup/warmup_AMP_<target_bacterium>.csv'."
        ),
    )
    ref_pdb_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to reference PDB structures used by Foldseek in evaluation_service.py. "
            "Auto-resolves to '/data/AMP_<target_bacterium>_811_new/ref_pdbs' when omitted."
        ),
    )
    model_path: str = Field(
        default=_DEFAULTS.get("model_path", "../Generation_Module/model_ckpt"),
        description=(
            "Path to the pretrained AMP generator checkpoint directory "
            "(passed as --model_path to run_ppo_training.py and read via cfg.get('model_path') in main_workflow.py)."
        ),
    )

    # Hydra Environment & LLM Reward Designer Settings
    task: str = Field(
        default=_DEFAULTS.get("task", "test_amp_generator_task"),
        description="Hydra task name in MAC-AMP_v1."
    )
    env_name: str = Field(
        default=_DEFAULTS.get("env_name", "test_amp_generator_env"),
        description="Environment module name in MAC-AMP_v1."
    )
    model: str = Field(
        default=_DEFAULTS.get("model", "gpt-5"),
        description="LLM model used by MAC-AMP_v1 for reward code design and RewardJudgeAgent."
    )
    temperature: float = Field(
        default=_DEFAULTS.get("temperature", 1.0),
        ge=0.0,
        le=2.0,
        description="Sampling temperature for generation."
    )
    suffix: str = Field(
        default=_DEFAULTS.get("suffix", "GPT"),
        description="Task suffix in MAC-AMP_v1."
    )

    # PPO & Sequence Generation Hyperparameters
    num_samples: int = Field(
        default=_DEFAULTS.get("num_samples", 32),
        ge=1,
        description="Number of peptide samples per evaluation step (cfg.num_samples)."
    )
    sample: int = Field(
        default=_DEFAULTS.get("sample", 3),
        ge=1,
        description="Number of approved reward samples generated per design loop (cfg.sample)."
    )
    max_seq_len: int = Field(
        default=_DEFAULTS.get("max_seq_len", 50),
        ge=1,
        description="Maximum sequence token length in Hydra config (cfg.max_seq_len)."
    )
    max_length: int = Field(
        default=_DEFAULTS.get("max_length", 32),
        ge=1,
        description="Maximum generated peptide amino acid length (--max_length in run_ppo_training.py)."
    )
    batch_size: int = Field(
        default=_DEFAULTS.get("batch_size", 10),
        ge=1,
        description="PPO training batch size (--batch_size in run_ppo_training.py)."
    )

    # Multi-Stage PPO & Sandbox Loop Control
    real_epoch: int = Field(
        default=_DEFAULTS.get("real_epoch", 15),
        ge=1,
        description="Number of PPO epochs during real model training (--epochs in real stage)."
    )
    sand_box_epoch: int = Field(
        default=_DEFAULTS.get("sand_box_epoch", 5),
        ge=1,
        description="Number of PPO epochs during sandbox reward evaluation (--epochs in sandbox stage)."
    )
    real_train_stage_iteration: int = Field(
        default=_DEFAULTS.get("real_train_stage_iteration", 3),
        ge=1,
        description="Number of outer real training stage iterations."
    )
    sandbox_iteration: int = Field(
        default=_DEFAULTS.get("sandbox_iteration", 3),
        ge=1,
        description="Number of inner sandbox iterations per real training stage."
    )
    reward_sample_num: int = Field(
        default=_DEFAULTS.get("reward_sample_num", 3),
        ge=1,
        description="Number of candidate reward functions trained in parallel per sandbox iteration."
    )

    # Predictor Ablation, Peer Review & Experiment Controls
    ablation_mode: int = Field(
        default=_DEFAULTS.get("ablation_mode", 0),
        ge=0,
        le=7,
        description=(
            "MAC-AMP_v1 predictor ablation mode (0..7): "
            "0=full predictors; 1=drop OmegaFold (Vb); 2=drop ToxinPred3 (Va); "
            "3=drop Macrel (Sb); 4=drop Va+Vb; 5=drop Vb+Sb; 6=drop Va+Sb; 7=drop Va+Vb+Sb."
        ),
    )
    skip_independent_reviewer: str = Field(
        default=_DEFAULTS.get("skip_independent_reviewer", ""),
        description="Comma-separated peer-review agent perspectives to skip (e.g., 'Perplexity' or '')."
    )
    drift_wordlist: str = Field(
        default=_DEFAULTS.get("drift_wordlist", "Null"),
        description="Drift wordlist setting passed to discussion_agents.py."
    )
    experiment: ExperimentType = Field(
        default=ExperimentType(_DEFAULTS.get("experiment", "baseline_api")),
        description="Peer-review LLM backend experiment mode."
    )
    default_sign: int = Field(
        default=_DEFAULTS.get("default_sign", -1),
        description="Hill score mapping sign passed to predict_mic_score (-1 uses standard Hill score)."
    )
    log_path: str = Field(
        default=_DEFAULTS.get("log_path", "my_test_tag"),
        description="Experiment log directory tag."
    )

    # Post-Parse Auto-Resolution & Supported Bacterium Validator
    # Runs automatically when MacAmpConfig is constructed.
    # It normalizes the bacterium name against BACTERIUM_CODE_MAP, rejects unsupported
    # bacteria, looks up the 2-letter bacterium_code, and auto-fills warmup_dir and
    # ref_pdb_path if the user didn't supply custom paths.
    @model_validator(mode="after")
    def resolve_mac_amp_bacterium_and_paths(self) -> "MacAmpConfig":
        raw_key = re.sub(r"[^a-z0-9]+", "_", self.target_bacterium.strip().lower()).strip("_")

        if raw_key not in BACTERIUM_CODE_MAP:
            supported_str = ", ".join(SUPPORTED_BACTERIA)
            raise ValueError(
                f"Unsupported target bacterium '{self.target_bacterium}'. "
                f"Supported MAC-AMP organisms are: {supported_str}."
            )

        canonical_name, code = BACTERIUM_CODE_MAP[raw_key]
        self.target_bacterium = canonical_name
        self.bacterium_code = code

        # Auto-resolve the warmup MIC CSV path if a custom CSV was not specified
        if not self.warmup_dir or "<" in self.warmup_dir:
            self.warmup_dir = f"../Generation_Module/data/warmup/warmup_AMP_{canonical_name}.csv"
        else:
            clean_csv = self.warmup_dir.strip().replace("\\", "/")
            if not clean_csv.lower().endswith(".csv"):
                raise ValueError(f"warmup_dir must point to a .csv file, got: '{self.warmup_dir}'")
            self.warmup_dir = clean_csv

        # Auto-resolve the Foldseek reference PDB folder for the target bacterium
        if not self.ref_pdb_path or "<" in self.ref_pdb_path:
            self.ref_pdb_path = f"/data/AMP_{canonical_name}_811_new/ref_pdbs"

        return self


# Top-Level Fail-Closed Envelope (`TranslationResult`)
class TranslationResult(BaseModel):
    """Envelope returned by the MAC-AMP Input Adapter."""
    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "clarification_required"]
    config: Optional[MacAmpConfig] = None
    missing_or_ambiguous_fields: List[str] = Field(default_factory=list)
    clarification_message: Optional[str] = None
