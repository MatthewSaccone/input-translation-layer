# MAC-AMP Input Adapter

Translates validated 6-field JSON requests into ready-to-run `CLMFAP/MAC-AMP_v1` Hydra configurations (`config.yaml`) and Spec-Kit run specifications (`spec.md`).

```text
User Prompt -> LLM Interpreter -> Core Validator -> MAC-AMP Adapter -> MAC-AMP Configuration
```

## What It Does
- **Accepts Structured JSON:** Takes the validated 6-field JSON (`algorithm_id: "mac_amp"`, `operation: "peptide_generation"`) with zero LLM calls.
- **Maps Fields & Defaults:**
  - `parameters.target_organism` $\rightarrow$ `target_bacterium` and auto-resolves `bacterium_code`, `warmup_dir`, and `ref_pdb_path`.
  - `inputs.peptide_dataset_ref` $\rightarrow$ `warmup_dir`
  - `inputs.model_ref` $\rightarrow$ `model_path`
  - `inputs.workflow_config_ref` $\rightarrow$ base config overrides
  - Applies official MAC-AMP defaults from `.specify/templates/config-template.json` for any omitted parameters.
- **Validates Requirements:** Rejects missing or unsupported bacteria and checks that required datasets, model checkpoints, and resource paths exist when `verify_resources=True`.

## Supported Bacteria
- `Escherichia coli` (`EC`)
- `Staphylococcus aureus` / `MRSA` (`SA`)
- `Pseudomonas aeruginosa` (`PA`)
- `Klebsiella pneumoniae` (`KP`)
- `Enterococcus faecium` (`EF`)

## Quick Start

From the repository root (`input-translation-layer/`):

```powershell
# Translate an example JSON request into a MAC-AMP Hydra config.yaml
python adapters/mac_amp/translator.py examples/mac_amp_request.json

# Run unit tests
python adapters/mac_amp/test_mac_amp_adapter.py -v
```
