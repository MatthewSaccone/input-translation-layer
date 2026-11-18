<!--
  - Defines the standard Markdown layout for the `spec.md` file saved inside
    each `specs/<NNN-bacterium>/` folder alongside `config.yaml`.
  - Records the source validated 6-field JSON request, validation status,
    resolved bacterium paths, and hyperparameter values.
-->

# Run Specification: [RUN_NAME]

- **Source Validated JSON**: `[VALIDATED_SIX_FIELD_JSON]`
- **Translation Mode**: `Structured JSON Adapter (0 LLM Calls)`
- **Validation Status**: `[VALID | CLARIFICATION_REQUIRED]`

## 1. Target Bacterium & Auto-Resolved MAC-AMP_v1 Paths
| Parameter | Value | Resolution Source |
| :--- | :--- | :--- |
| `target_bacterium` | `[Escherichia_coli]` | `parameters.target_organism` (Required) |
| `bacterium_code` | `[EC]` | Auto-Mapped from `target_bacterium` |
| `warmup_dir` | `../Generation_Module/data/warmup/warmup_AMP_[Target_Bacterium].csv` | Auto-Resolved / `inputs.peptide_dataset_ref` |
| `ref_pdb_path` | `/data/AMP_[Target_Bacterium>_811_new/ref_pdbs` | Auto-Resolved |
| `model_path` | `../Generation_Module/model_ckpt` | Default / `inputs.model_ref` |

## 2. MAC-AMP_v1 Hydra & PPO Parameters (`cfg/config.yaml`)
| Parameter | Value | Applied From |
| :--- | :--- | :--- |
| `real_train_stage_iteration` | `3` | Default / JSON `parameters` Override |
| `sandbox_iteration` | `3` | Default / JSON `parameters` Override |
| `real_epoch` | `15` | Default / JSON `parameters` Override |
| `sand_box_epoch` | `5` | Default / JSON `parameters` Override |
| `batch_size` | `10` | Default / JSON `parameters` Override |
| `num_samples` | `32` | Default / JSON `parameters` Override |
| `max_length` | `32` | Default / JSON `parameters` Override |
| `max_seq_len` | `50` | Default / JSON `parameters` Override |
| `temperature` | `1.0` | Default / JSON `parameters` Override |
| `ablation_mode` | `0` | Default / JSON `parameters` Override |
| `experiment` | `"baseline_api"` | Default / JSON `parameters` Override |
