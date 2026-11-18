"""Optional manual evaluation against the configured real LLM (requires API key).

Run: python -m tests.evaluate_real_llm
No tests are called automatically or charged by pytest.
"""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from llm.interpreter import interpret_user_prompt

CASES = [
    ("Generate antimicrobial peptides targeting E. coli", "mac_amp", "peptide_generation", {"target_organism": "Escherichia coli"}),
    ("Create 50 new 3D molecules", "rl_diffusion", "sample", {"n_samples": 50}),
    ("Train a molecular diffusion model on my data", "rl_diffusion", "train", {}),
    ("Generate some promising candidates", None, None, {}),
    ("Make 25 molecules but do not save XYZ", "rl_diffusion", "sample", {"n_samples": 25, "save_to_xyz": False}),
    ("Generate peptides for a bacterium", "mac_amp", "peptide_generation", {}),
    ("Make 30 molecules with high drug-likeness", "rl_diffusion", "sample", {"n_samples": 30}),
    ("Generate 20 molecules with QED at least 0.8", "rl_diffusion", "sample", {"n_samples": 20}),
    ("I want 20 molecules and 50 molecules", None, None, {}),
    ("Use a fictional unsupported algorithm to simulate protein folding", None, None, {}),
]


def main() -> None:
    load_dotenv()
    results = []
    for prompt, algorithm, operation, expected in CASES:
        try:
            result = interpret_user_prompt(prompt)
            # Conflicting quantities: both no-route and a routed but unresolved
            # request are acceptable, as long as no sample count is invented.
            conflicting_count = prompt == "I want 20 molecules and 50 molecules"
            routing_ok = (
                (result["algorithm_id"] in (None, "rl_diffusion")
                 and result["operation"] in (None, "sample"))
                if conflicting_count else
                (result["algorithm_id"] == algorithm and result["operation"] == operation)
            )
            extracted_ok = all(result["parameters"].get(k) == v for k, v in expected.items())
            if prompt == "Generate antimicrobial peptides targeting E. coli":
                extracted_ok = result["parameters"].get("target_organism", "").casefold() in {
                    "e. coli", "escherichia coli"
                }
            if prompt == "Generate peptides for a bacterium":
                extracted_ok = "target_organism" not in result["parameters"]
            if conflicting_count:
                extracted_ok = extracted_ok and "n_samples" not in result["parameters"]
            # Numeric threshold test must be preserved (unresolved or typed field).
            numeric_ok = (
                any("numerical_objective" in issue for issue in result["unresolved"])
                or "qed" in result["parameters"].get("requested_objectives", {})
            ) if "at least 0.8" in prompt else True
            issue_ok = bool(result["unresolved"]) if prompt in (
                "Generate some promising candidates",
                "I want 20 molecules and 50 molecules",
                "Use a fictional unsupported algorithm to simulate protein folding",
            ) else True
            qualitative_ok = (
                result["parameters"].get("requested_objectives", {}).get("qed") == "maximize"
            ) if "high drug-likeness" in prompt else True
            passed = routing_ok and extracted_ok and numeric_ok and issue_ok and qualitative_ok
            results.append({"prompt": prompt, "passed": passed, "result": result})
        except Exception as exc:
            results.append({"prompt": prompt, "passed": False, "error": f"{type(exc).__name__}: {exc}"})
    output = Path("llm_evaluation_results.json")
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(row["passed"] for row in results)
    print(f"Real-LLM evaluation: {total}/{len(results)} cases passed (partial automatic scoring)")
    print(f"Inspect individual results in {output} before interpreting the score.")
    if total < len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
