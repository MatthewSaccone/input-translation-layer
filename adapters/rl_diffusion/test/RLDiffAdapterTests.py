from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))

import RLDiffAdapter



SAMPLE_EXPECTATION: dict = \
{
    "model_path":"outputs/edm_1",
    "mode":"baseline",
    "baseline":"test",
    "n_samples":"50",
    "batch_size_gen":"100",
    "save_to_xyz":"True"
}

TRAIN_EXPECTATION: dict = \
{
    "dataset":"upload_molecules_001",
    "remove_h":"True",
    "batch_size":"128",
    "n_layers":"6",
    "nf":"128",
    "n_epochs":"200",
    "CUDA_DEVICES":"0",
    "conditioning":["qed sas affinity"]
}

adapter = RLDiffAdapter.RLDiffAdapter()

def TrainTest():
    train_input = "./test/train_test.json"
    output = adapter.run(train_input)
    print("Train test output: ", output)
    return output


def  SampleTest():
    sample_input = "./test/sample_test.json"
    output = adapter.run(sample_input)
    print("Sample test output: ", output)
    return output


if __name__ == "__main__":
    assert TrainTest() == TRAIN_EXPECTATION, "Train test failed"
    assert SampleTest() == SAMPLE_EXPECTATION, "Sample test failed"
    print("All tests passed")