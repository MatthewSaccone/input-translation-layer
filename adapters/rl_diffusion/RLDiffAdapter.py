"""
RL Diffusion Adapter
Reads validated JSON output from LLM request
Extract input data
Passes extracted input data off to RL diffusion model
"""

import json
import os
import re
import sys


class AdapterError(Exception):
    pass

ALLOWED_PARAMS = {
    "sample": {"n_samples", "save_to_xyz", "requested_objectives"},
    "train": {"conditioning"}
}

# PATHS May need to be changes upon integration
GEN_SCRIPT_PATH = "RL-Diffusion/scripts/eval-analyze.py"
TRAIN_SCRIPT_PATH = "RL-Diffusion/scripts/..."

def open_json(json_file_name: str) -> dict:
    with open(json_file_name, encoding="utf-8") as json_file:
        return json.load(json_file)

# JSON Format Example
# ===================
# {
#   "schema_version": "1.0",
#   "algorithm_id": "rl_diffusion",
#   "operation": "sample",
#   "parameters": {
#     "n_samples": 50,
#     "save_to_xyz": true
#   },
#   "inputs": {},
#   "unresolved": []
# }

def validate_json_features(json_data: dict) -> dict:
    """
    Validates JSON features against RL diffusion model
    Prepares features for algorithm argument conversion
    """
    if json_data is None:
        raise AdapterError("Input data cannot be None")
    if json_data["algorithm_id"] != "rl_diffusion":
        raise AdapterError("Not an RL Diffusion request")
    if len(json_data["unresolved"]) != 0:
        raise AdapterError("Request contains unresolved items")

    op = json_data["operation"]
    if op not in ALLOWED_PARAMS:
        raise AdapterError("Unsupported operation")

    params = json_data.get("parameters", {})
    extra_params = set(params) - ALLOWED_PARAMS[op]
    if extra_params:
        raise AdapterError("Unexpected parameters")

    features = {"operation": op, "parameters": params, "inputs": json_data.get("inputs", {}) }
    return features

def generate_exp_name(operation: str) -> str:
    ...

def generate_command(features: dict, exp_name: str) -> list[str]:
    ...

def _main(json_file_name: str):
    data = open_json(json_file_name)
    ...
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise AdapterError("Usage: python RLDiffAdapter.py <json_file_name>")


    _main(sys.argv[1])