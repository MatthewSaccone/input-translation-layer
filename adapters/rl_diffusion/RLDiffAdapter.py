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
import argparse
import subprocess
from json import JSONDecoder


class AdapterError(Exception):
    pass



_TRAIN_DEFAULTS: dict
with open("./templates/training_defaults.json", encoding="utf-8") as f:
    _TRAIN_DEFAULTS = json.load(f)

_ALLOWED_PARAMS = {
    "sample": {"n_samples", "save_to_xyz", "requested_objectives"},
    "train": {"conditioning"}
}

_ACCEPTED_CONDITIONING = ["qed", "sas", "affinity"]

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
    if op not in _ALLOWED_PARAMS:
        raise AdapterError("Unsupported operation")

    params = json_data.get("parameters", {})
    extra_params = set(params) - _ALLOWED_PARAMS[op]
    if extra_params:
        raise AdapterError("Unexpected parameters")

    features = {"operation": op, "parameters": params, "inputs": json_data.get("inputs", {}) }
    return features

def generate_rl_diff_input(features: dict) -> dict:
    params, op = features["parameters"], features["operation"]
    inputs = features["inputs"]

    rl_diff_input = {}

    if op == "train":
        ds = inputs.get("training_dataset_ref")
        if ds is None:
            raise AdapterError("Training dataset cannot be None")
        rl_diff_input["dataset"] = ds

        rl_diff_input["remove_h"] = _TRAIN_DEFAULTS["remove_h"]
        rl_diff_input["batch_size"] = _TRAIN_DEFAULTS["batch_size"]
        rl_diff_input["n_layers"] = _TRAIN_DEFAULTS["n_layers"]
        rl_diff_input["nf"] = _TRAIN_DEFAULTS["nf"]
        rl_diff_input["n_epochs"] = _TRAIN_DEFAULTS["n_epochs"]
        rl_diff_input["CUDA_DEVICES"] = _TRAIN_DEFAULTS["CUDA_DEVICES"]

        conds = params.get("conditioning", [])
        for cond in conds:
            if cond not in _ACCEPTED_CONDITIONING:
                raise AdapterError("Unsupported conditioning: {}".format(cond))
        rl_diff_input["conditioning"] = str.join(" ", conds)
    else:
        ...

    return rl_diff_input

def _main(json_file_name: str):
    data = open_json(json_file_name)
    features = validate_json_features(data)
    translation_output = generate_rl_diff_input(features)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Diffusion Adapter")
    parser.add_argument("json_file", help="JSON input file")
    args = parser.parse_args()

    _main(args.json_file)
