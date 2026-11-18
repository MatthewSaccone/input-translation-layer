<!--
  - Defines the standard Markdown layout for the `spec.md` file saved inside
    each `specs/<NNN-bacterium>/` folder alongside `config.yaml`.
  - Records the original English prompt, validation status, resolved bacterium
    paths, and hyperparameter values.
-->

# Run Specification: [RUN_NAME]

- **Source User Prompt**: `"[ORIGINAL_USER_PROMPT]"`
- **Validation Status**: `[VALID | NEEDS CLARIFICATION]`

## 1. Target Bacterium & Auto-Resolved MAC-AMP_v1 Paths
| Parameter | Value | Resolution Source |
| :--- | :--- | :--- |
| `target_bacterium` | `[Escherichia_coli]` | User Prompt (Required) |
| `bacterium_code` | `[EC]` | Auto-Mapped from `target_bacterium` |
| `warmup_dir` | `../Generation_Module/data/warmup/warmup_AMP_[Target_Bacterium].csv` | Auto-Resolved / User Override |
| `ref_pdb_path` | `/data/AMP_[Target_Bacterium]_811_new/ref_pdbs` | Auto-Resolved |

## 2. MAC-AMP_v1 Hydra & PPO Parameters (`cfg/config.yaml`)
| Parameter | Value | Applied From |
| :--- | :--- | :--- |
| `real_train_stage_iteration` | `3` | Default / User Override |
| `sandbox_iteration` | `3` | Default / User Override |
| `real_epoch` | `15` | Default / User Override |
| `sand_box_epoch` | `5` | Default / User Override |
| `batch_size` | `10` | Default / User Override |
| `num_samples` | `32` | Default / User Override |
| `max_length` | `32` | Default / User Override |
| `max_seq_len` | `50` | Default / User Override |
| `temperature` | `1.0` | Default / User Override |
| `ablation_mode` | `0` | Default / User Override |
| `experiment` | `"baseline_api"` | Default / User Override |
