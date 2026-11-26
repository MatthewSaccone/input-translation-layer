"""Non-executing review adapter for the shared RL-Diffusion input contract.

The native model and training-data mappings have not been verified. This adapter
therefore preserves the request for review instead of building executable inputs
from unverified references. No model is loaded, trained or sampled here.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from core.validator import validate_translation_request


class AdapterError(ValueError):
    """A request cannot safely be prepared from its current information."""


class RLDiffAdapter:
    """Validate the shared contract and return a clearly labelled review preview."""

    def open_json(self, json_file_name: str | Path) -> dict[str, Any]:
        # utf-8-sig also accepts files written with a Windows UTF-8 BOM.
        with Path(json_file_name).open(encoding="utf-8-sig") as json_file:
            return json.load(json_file)

    def validate_json_features(self, json_data: Mapping[str, Any]) -> dict[str, Any]:
        # Use the same validator as the rest of the platform instead of duplicating
        # partial type checks here. This also checks missing and unexpected fields.
        report = validate_translation_request(json_data)
        if not report.valid:
            raise AdapterError("; ".join(report.errors))
        if report.request["algorithm_id"] != "rl_diffusion":
            raise AdapterError("Not an RL-Diffusion request")
        # Copy the containers so the review result cannot change the caller's data.
        return {
            "operation": report.request["operation"],
            "parameters": deepcopy(dict(report.request["parameters"])),
            "inputs": deepcopy(dict(report.request["inputs"])),
        }

    def generate_rl_diff_input(self, features: Mapping[str, Any]) -> dict[str, Any]:
        """Return a review record, not native CLI arguments or a runnable config."""
        # Recheck the contract even when this method is called outside run().
        validated = self.validate_json_features({
            "schema_version": "1.0",
            "algorithm_id": "rl_diffusion",
            "operation": features.get("operation"),
            "parameters": features.get("parameters"),
            "inputs": features.get("inputs"),
            "unresolved": [],
        })
        operation = validated["operation"]
        parameters = validated["parameters"]
        inputs = validated["inputs"]

        # Do not substitute a resource ID for a physical path, or assume a default
        # checkpoint exists. The algorithm owner must verify the native mapping.
        blockers = ["Native resource mapping and compatibility are unverified."]
        if operation == "sample":
            blockers.append("Model reference requires verified native resource resolution.")
            # The current native handoff cannot enforce requested objectives.
            # Preserve them exactly in the preview rather than silently dropping them.
            if "requested_objectives" in parameters:
                blockers.append("Requested objectives have no verified native sampling mapping.")
        else:
            # A logical training-data reference is not the native dataset identifier.
            # Preserve conditioning as a list; do not guess a dataset or CLI format.
            blockers.append("Training data reference requires verified native dataset mapping.")

        return {
            "status": "review_preview_only",
            "operation": operation,
            "parameters": parameters,
            "resource_refs": inputs,
            "blocking_issues": blockers,
            "execution_ready": False,
        }

    def prepare_preview(self, request: Mapping[str, Any]) -> dict[str, Any]:
        # In-memory entrypoint for the common pipeline; no temporary JSON is needed.
        return self.generate_rl_diff_input(self.validate_json_features(request))

    def run(self, json_file_name: str | Path) -> dict[str, Any]:
        # Keep the historical file-based entrypoint as a compatibility wrapper.
        return self.prepare_preview(self.open_json(json_file_name))
