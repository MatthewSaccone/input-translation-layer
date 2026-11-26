"""Manual end-to-end, non-executing researcher-prompt pipeline demo.

Examples:
  python run_mac_amp_pipeline.py --demo
  python run_mac_amp_pipeline.py "Generate antimicrobial peptides targeting E. coli"
"""
from __future__ import annotations

import argparse
import json
from dotenv import load_dotenv

from core.natural_language_pipeline import prepare_from_prompt
from llm.interpreter import ExtractedIntent


class _OfflineDemoProvider:
    """Fixed fixture ONLY; not a model, not intended for arbitrary inputs."""

    def extract(self, prompt, specs):
        return ExtractedIntent(
            algorithm_id="mac_amp", operation="peptide_generation",
            target_organism="Escherichia coli",
            n_samples=None, save_to_xyz=None, conditioning=None,
            qed_direction=None, sas_direction=None, affinity_direction=None,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Test prompt -> LLM -> validator -> MAC-AMP YAML preview")
    parser.add_argument("prompt", nargs="?", help="Researcher prompt; real LLM API call unless --demo")
    parser.add_argument("--demo", action="store_true", help="Fixed offline example; does NOT interpret the supplied prompt")
    args = parser.parse_args()
    load_dotenv()
    if not args.prompt and not args.demo:
        parser.error("Supply a researcher prompt or --demo")
    prompt = args.prompt if args.prompt else "Generate antimicrobial peptides targeting E. coli"
    provider = _OfflineDemoProvider() if args.demo else None
    try:
        result = prepare_from_prompt(prompt, provider=provider)
    except (ValueError, RuntimeError) as exc:
        print(f"Unable to interpret request: {exc}")
        return 1
    if args.demo:
        print("OFFLINE FIXTURE — no API call; this is not a language-understanding test")
    print("Translated request:")
    print(json.dumps(result.request, indent=2, ensure_ascii=False))
    print(f"Status: {result.pipeline.status}")
    if result.pipeline.issues:
        print("Issues:")
        for issue in result.pipeline.issues:
            print(f"  - {issue}")
    # Print blockers and warnings separately from the preview itself.
    for warning in result.pipeline.warnings:
        print(f"Warning: {warning}")
    if result.pipeline.argument_preview is not None:
        print("\nRL-Diffusion review PREVIEW (not native arguments):")
        print(json.dumps(result.pipeline.argument_preview, indent=2, ensure_ascii=False))
    if result.pipeline.yaml:
        print("\nMAC-AMP configuration PREVIEW (resources not verified):")
        print(result.pipeline.yaml)
    return 0 if result.pipeline.status == "config_preview_unverified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
