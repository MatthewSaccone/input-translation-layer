"""Legacy smoke-test entrypoint; new regression tests live in tests/."""
from pathlib import Path
import sys

# Support direct invocation from any working directory.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from adapters.rl_diffusion.RLDiffAdapter import AdapterError, RLDiffAdapter

if __name__ == "__main__":
    adapter = RLDiffAdapter()
    fixtures = Path(__file__).resolve().parent
    training = adapter.run(fixtures / "train_test.json")
    assert training["execution_ready"] is False
    # The historical sampling fixture lacks a registered model reference.
    # It must now fail rather than invent a checkpoint path.
    try:
        adapter.run(fixtures / "sample_test.json")
    except AdapterError:
        pass
    else:
        raise AssertionError("Missing model reference should be rejected")
    print("RL-Diffusion review smoke tests passed")
