from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"run"))
from benchmark_pcs_translation import MODELS, training_configs
from evaluate_pcs_translation import precision_audit, summarize_arrays, translation_arrays
from evaluate_summed_translation import outputs
from qcnn.hybrid import PCSQCNN


@pytest.mark.parametrize("kind", ["general", "local_3x3"])
@pytest.mark.parametrize("summed", [False, True])
def test_reconstruction_matches_direct_torus_and_keeps_head_unaligned(summed, kind):
    torch.manual_seed(7)
    images = torch.rand(3, 8, 8, dtype=torch.float64)
    model = PCSQCNN(8, 10, feature_qubits=2, quantum_layers=3,
                    multiplexer_kind=kind, dtype=torch.float64,
                    reduce_readout_to_feature_distribution=summed).eval()
    with torch.no_grad():
        model.classifier.bias.zero_()
        model.classifier.weight.mul_(10)
    metrics, arrays, validation = translation_arrays(model, images, torch.tensor([0, 1, 2]),
                                                    batch_size=2, validation_images=3)
    assert validation["passed"] and validation["readout_l1_max"] < 1e-12
    for y in range(8):
        for x in range(8):
            _, z = outputs(model, images, (y, x), 2)
            expected = arrays["predictions"][y % (4 if summed else 8), x % (4 if summed else 8)]
            np.testing.assert_array_equal(z.argmax(1).numpy(), expected)
    assert metrics["aligned_feature_tv"][0][0] == 0
    if not summed:
        base, z0 = outputs(model, images, (0, 0), 2)
        moved, z4 = outputs(model, images, (4, 0), 2)
        assert (moved-base).abs().max() > 1e-5
        assert (z4-z0).abs().max() > 1e-5
        assert (moved-torch.roll(base, 1, 1)).abs().max() < 1e-12


def test_feature_distance_is_per_image_and_accuracy_is_not_consistency():
    q = np.tile(np.array([[1., 0.], [0., 1.]])[None, None, :, None, None, :], (4, 4, 1, 1, 1, 1))
    q[1, 2] = q[0, 0, ::-1]
    preds = np.zeros((4, 4, 2), dtype=np.uint8)
    preds[1, 2] = 1
    m = summarize_arrays(preds, np.array([0, 1]), q)
    assert m["accuracy"][0][0] == m["accuracy"][1][2] == .5
    assert m["changed_labels"][1][2] == 1
    assert m["aligned_feature_tv"][1][2] == 1
    assert np.array_equal(q[0, 0].mean(0), q[1, 2].mean(0))


def test_large_trained_generators_are_checked_in_double_precision():
    import copy
    torch.manual_seed(3)
    native = PCSQCNN(8, 10, feature_qubits=2, quantum_layers=3,
                    multiplexer_kind="local_3x3", reduce_readout_to_feature_distribution=True).eval()
    with torch.no_grad():
        for p in native.multiplexers.parameters():
            p.mul_(20)
    accurate = copy.deepcopy(native).double()
    images, labels = torch.rand(3, 8, 8), torch.tensor([0, 1, 2])
    _, arrays, validation = translation_arrays(accurate, images.double(), labels, validation_images=3)
    assert validation["normalization_linf_max"] < 1e-10
    audit = precision_audit(native, accurate, images, labels, arrays, batch_size=3)
    assert audit["grid_predictions"] == 16*3
    assert audit["cell_readout_l1_max"] > validation["normalization_linf_max"]
    assert audit["diagnostic_dtype"] == "torch.float64"


def test_protocol_preserves_2000_and_all_stages(tmp_path):
    for name, (kind, summed, final) in MODELS.items():
        configs = training_configs(name, 2, tmp_path/"data", tmp_path/"out", "cpu")
        assert len(configs) == final//2000
        for stage, c in enumerate(configs):
            assert c.seeds.base_seed == 2 and c.seeds.seed_count == 1
            assert c.training.num_epochs == 2000 and 2000 in c.training.snapshot_epochs
            assert c.dataset.max_offset == 8 and c.dataset.test_max_offset is None
            assert c.optimizer.learning_rate == .03
            assert c.model.constructor_kwargs["multiplexer_kind"] == kind
            assert c.model.constructor_kwargs["reduce_readout_to_feature_distribution"] == summed
            if stage:
                assert f"e{stage*2000}/seed2/checkpoint_final_seed2.pt" in c.training.init_checkpoint
            else:
                assert c.training.init_checkpoint is None
    configs = training_configs("local_full", 0, tmp_path, tmp_path, "cpu", {2000: tmp_path/"existing.pt"})
    assert configs[1].training.init_checkpoint == str(tmp_path/"existing.pt")


def test_local_full_stage_resume_preserves_previous_checkpoint(tmp_path, monkeypatch):
    from dataclasses import replace
    from qcnn.automation import run_mnist_auto_training
    from qcnn.data import PreparedMnistSplits, TensorImageDataset
    from evaluate_summed_translation import file_hash
    dataset = TensorImageDataset(torch.rand(4, 8, 8), torch.tensor([0, 1, 2, 3]))
    monkeypatch.setattr("qcnn.automation.prepare_mnist_splits",
                        lambda **kwargs: PreparedMnistSplits(dataset, dataset))
    counts, prior_hash = [], None
    configs = training_configs("local_full", 0, tmp_path/"data", tmp_path, "cpu")
    first_path = tmp_path/configs[0].output.directory_name/"checkpoint_final_seed0.pt"
    for config in configs[:2]:
        config = replace(config,
            model=replace(config.model, constructor_kwargs={**config.model.constructor_kwargs, "image_size": 8}),
            dataset=replace(config.dataset, train_batch_size=2, test_batch_size=2),
            training=replace(config.training, num_epochs=2, snapshot_epochs=(1, 2), test_evaluation_interval_epochs=1))
        run_mnist_auto_training(config, save_training_state=True, implementation={"test": True},
                               seed_epoch_end_callback=lambda *args: counts.append(args[1]))
        completed = len(counts)
        run_mnist_auto_training(config, resume=True, save_training_state=True, implementation={"test": True},
                               seed_epoch_end_callback=lambda *args: counts.append(args[1]))
        assert len(counts) == completed
        if prior_hash is None:
            prior_hash = file_hash(first_path)
        else:
            assert file_hash(first_path) == prior_hash
    assert counts == [1, 2, 1, 2]


def test_plots_can_be_regenerated_without_checkpoints_or_simulation(tmp_path):
    from evaluate_summed_translation import write_json
    from plot_pcs_translation import plot
    name = "local_full"
    entries = {}
    for total in (2000, 4000, 6000):
        relative = f"records/{total}.json"
        write_json(tmp_path/relative, {"training_history": {
            "train_epoch": [1, 2000], "test_epoch": [10, 2000],
            "train_metrics": {"accuracy": [.2, .6]}, "test_metrics": {"accuracy": [.2, .5]}},
            "final_summaries": {"train": {"metrics": {"accuracy": .6}}, "test": {"metrics": {"accuracy": .5}}}})
        entries[str(total)] = {"record_file": relative}
    write_json(tmp_path/"runs.json", {"runs": {name: {"0": entries}}})
    write_json(tmp_path/"diagnostics"/f"{name}_seed0.json", {
        "metrics": {"accuracy": np.full((32, 32), .5).tolist(),
                    "changed_labels": np.zeros((32, 32)).tolist(),
                    "aligned_feature_tv": np.zeros((4, 4)).tolist()},
        "parameters": 2858,
        "direct_validation": {"readout_l1_max": 0, "logit_linf_max": 0, "changed_labels_max": 0}})
    figures = tmp_path/"manuscript_figures"
    plot(tmp_path, [name], [0], figures_dir=figures)
    assert (figures/"pcs_readout_convergence.pdf").read_bytes() == (tmp_path/"plots/pcs_convergence.pdf").read_bytes()
    assert (tmp_path/"plots/local_full_translation.pdf").is_file()
    assert (tmp_path/"plots/pcs_convergence.pdf").is_file()
    assert len((tmp_path/"plots/translation_maps.csv").read_text().splitlines()) == 1+2*1024+16
