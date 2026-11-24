"""
RL Diffusion Adapter
Reads validated JSON output from LLM request
Extract input data
Passes extracted input data off to RL diffusion model
"""

import json
import argparse

class AdapterError(Exception):
    pass


class RLDiffAdapter:
    def __init__(self):
        self._TRAIN_DEFAULTS: dict
        with open("./templates/training_defaults.json", encoding="utf-8") as f:
            self._TRAIN_DEFAULTS = json.load(f)

        self._SAMPLE_DEFAULTS: dict
        with open("./templates/sample_defaults.json", encoding="utf-8") as f:
            self._SAMPLE_DEFAULTS = json.load(f)

        self._ALLOWED_PARAMS = {
            "sample": {"n_samples", "save_to_xyz", "requested_objectives"},
            "train": {"conditioning"}
        }

        self._ACCEPTED_CONDITIONING = ["qed", "sas", "affinity"]

    def open_json(self,json_file_name: str) -> dict:
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

    def validate_json_features(self, json_data: dict) -> dict:
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

        op = json_data.get("operation")
        if op not in self._ALLOWED_PARAMS:
            raise AdapterError("Unsupported operation")

        params = json_data.get("parameters", {})
        extra_params = set(params) - self._ALLOWED_PARAMS[op]
        if extra_params:
            raise AdapterError("Unexpected parameters")

        features = {"operation": op, "parameters": params, "inputs": json_data.get("inputs", {}) }
        return features

    def generate_rl_diff_input(self, features: dict) -> dict:
        params, op = features.get("parameters"), features.get("operation")
        inputs = features.get("inputs")

        rl_diff_input = {}

        if op == "train":
            ds = inputs.get("training_dataset_ref")
            if ds is None:
                raise AdapterError("Training dataset cannot be None")
            rl_diff_input["dataset"] = ds

            rl_diff_input["remove_h"] = self._TRAIN_DEFAULTS["remove_h"]
            rl_diff_input["batch_size"] = self._TRAIN_DEFAULTS["batch_size"]
            rl_diff_input["n_layers"] = self._TRAIN_DEFAULTS["n_layers"]
            rl_diff_input["nf"] = self._TRAIN_DEFAULTS["nf"]
            rl_diff_input["n_epochs"] = self._TRAIN_DEFAULTS["n_epochs"]
            rl_diff_input["CUDA_DEVICES"] = self._TRAIN_DEFAULTS["CUDA_DEVICES"]

            conds = params.get("conditioning", [])
            for cond in conds:
                if cond not in self._ACCEPTED_CONDITIONING:
                    raise AdapterError("Unsupported conditioning: {}".format(cond))
            rl_diff_input["conditioning"] = str.join(" ", conds)
        else:
            model = self._SAMPLE_DEFAULTS["model_path"] if inputs.get("model_ref") is None else inputs.get("model_ref")
            rl_diff_input["model_path"] = model
            rl_diff_input["mode"] = self._SAMPLE_DEFAULTS["mode"]
            rl_diff_input["baseline"] = self._SAMPLE_DEFAULTS["baseline"]
            rl_diff_input["n_samples"] = params.get("n_samples", self._SAMPLE_DEFAULTS["n_samples"])
            rl_diff_input["batch_size_gen"] = params.get("batch_size_gen", self._SAMPLE_DEFAULTS["batch_size_gen"])
            rl_diff_input["save_to_xyz"] = params.get("save_to_xyz", self._SAMPLE_DEFAULTS["save_to_xyz"])

        return rl_diff_input

    def run(self, json_file_name: str):
        data = self.open_json(json_file_name)
        features = self.validate_json_features(data)
        translation_output = self.generate_rl_diff_input(features)

        return translation_output