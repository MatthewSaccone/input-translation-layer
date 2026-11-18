---
name: speckit-translate
description: Translate plain-English MAC-AMP configuration requests into Spec-Kit specification (spec.md) and validated MAC-AMP_v1 Hydra config (config.yaml) using .specify/memory/constitution.md and .specify/templates/.
---

<!--
  - Instructs the IDE's built-in AI agent how to run or follow the Spec-Kit
    translation workflow when you ask for a MAC-AMP run configuration in chat.
-->

# Spec-Kit NLP-to-Configuration Translator (MAC-AMP_v1 Exact Schema)

Whenever the user provides a plain-English request to configure a run (or invokes `/speckit-translate`), follow this exact workflow:

1. **Read the Constitution & Templates:**
   - Check `.specify/memory/constitution.md` for `CLMFAP/MAC-AMP_v1` validation rules and defaults.
   - Reference `.specify/templates/spec-template.md` and `.specify/templates/config-template.json`.

2. **Validate Required Input & Auto-Resolve MAC-AMP_v1 Paths:**
   - Verify that `target_bacterium` is present in the user's request. If missing, halt and ask for clarification.
   - Auto-resolve `bacterium_code` (`EC`, `SA`, `PA`, `KP`, `EF`), `warmup_dir` (`../Generation_Module/data/warmup/warmup_AMP_<Target_Bacterium>.csv`), and `ref_pdb_path` (`/data/AMP_<Target_Bacterium>_811_new/ref_pdbs`).

3. **Generate Output Artifacts (No Code Execution):**
   - Create a new numbered folder under `specs/` (e.g., `specs/001-escherichia-coli/`).
   - Write `specs/<id>/spec.md` and `specs/<id>/config.yaml`.
