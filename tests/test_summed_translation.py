from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

from qcnn.hybrid import PCSQCNN
from qcnn.automation import run_mnist_auto_training
from qcnn.data import PreparedMnistSplits, TensorImageDataset

# Run modules also import adjacent entrypoints when used from the command line.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run"))
from benchmark_summed_pcs import configurations
from evaluate_summed_translation import accuracy_scan, outputs, periodicity


def test_coarse_periodicity_and_full_readout_negative_control():
    torch.manual_seed(7)
    images = torch.rand(2, 8, 8, dtype=torch.float64)
    full = PCSQCNN(8, 10, feature_qubits=2, quantum_layers=3,
                   multiplexer_kind="local_3x3", dtype=torch.float64).eval()
    summed = PCSQCNN(8, 10, feature_qubits=2, quantum_layers=3,
                     multiplexer_kind="local_3x3", dtype=torch.float64,
                     reduce_readout_to_feature_distribution=True).eval()
    summed.multiplexers.load_state_dict(full.multiplexers.state_dict())
    full_result = periodicity(full, images)
    summed_result = periodicity(summed, images)
    assert full_result["readout_equivariance_l1_max"] < 1e-10
    assert full_result["readout_invariance_l1_max"] > 1e-5
    assert full_result["logit_linf_max"] > 1e-5
    assert summed_result["readout_invariance_l1_max"] < 1e-10
    assert summed_result["logit_linf_max"] < 1e-10
    assert summed_result["prediction_agreement_min"] == 1.0
    base, _ = outputs(summed, images, (0, 0), 2)
    shifted, _ = outputs(summed, images, (1, 0), 2)
    assert (base-shifted).abs().max() > 1e-5


class PositionClassifier(torch.nn.Module):
    def exact_quantum_readout_probabilities(self, images):
        # Deliberately position-sensitive: fixed accuracy does not imply
        # unchanged predictions. Every requested shift must really run.
        return images[:, :1, :1, None]

    def classify_readout_histogram(self, p):
        p = p.flatten(1)
        return torch.cat((p, 1-p), dim=1)


def test_accuracy_and_agreement_match_saved_predictions():
    images = torch.zeros(4, 32, 32)
    images[0, 0, 0] = 1
    images[1, 31, 31] = 1
    images[2, 0, 31] = 1
    labels = torch.tensor([0, 0, 1, 1])
    result, shifts, predictions = accuracy_scan(PositionClassifier(), images, labels)
    assert np.array(result["cell_accuracy"]).shape == (4, 4)
    for row, pred in zip(result["shifts"], predictions):
        assert row["accuracy"] == np.mean(pred == labels.numpy())
    origin = predictions[np.flatnonzero((shifts == (0, 0)).all(axis=1))[0]]
    assert any(row["agreement_with_origin"] < 1 for row in result["shifts"])
    for row, pred in zip(result["shifts"], predictions):
        assert row["agreement_with_origin"] == np.mean(pred == origin)
    assert {(0, k) for k in range(32)} <= set(map(tuple, shifts))
    assert {(k, 0) for k in range(32)} <= set(map(tuple, shifts))
    assert (12, 12) in set(map(tuple, shifts))


def test_protocol_and_resume_without_retraining(tmp_path, monkeypatch):
    configs = configurations(tmp_path / "data", tmp_path / "artifacts", "cpu")
    for stage, config in enumerate(configs):
        assert config.training.num_epochs == 2000
        assert config.training.torch_matmul_precision == "highest"
        assert config.seeds.seed_count == 3 and config.seeds.base_seed == 0
        assert config.dataset.max_offset == 8 and config.dataset.test_max_offset is None
        assert config.optimizer.learning_rate == .03
        assert config.model.constructor_kwargs["reduce_readout_to_feature_distribution"]
        assert config.model.constructor_kwargs["multiplexer_kind"] == "local_3x3"
        if stage:
            assert configs[stage-1].output.directory_name in config.training.init_checkpoint
        else:
            assert config.training.init_checkpoint is None
    # Exercise the actual checkpoint format, JSON conversion and fresh-Adam
    # stage initialization on tiny synthetic data, never on local MNIST.
    images = torch.rand(4, 8, 8)
    dataset = TensorImageDataset(images, torch.tensor([0, 1, 2, 3]))
    monkeypatch.setattr("qcnn.automation.prepare_mnist_splits", lambda **kwargs: PreparedMnistSplits(dataset, dataset))
    counts = []
    for stage, config in enumerate(configs[:2]):
        kwargs = dict(config.model.constructor_kwargs, image_size=8)
        config = replace(config, model=replace(config.model, constructor_kwargs=kwargs),
                         training=replace(config.training, num_epochs=2, snapshot_epochs=(1, 2), test_evaluation_interval_epochs=1),
                         seeds=replace(config.seeds, seed_count=1),
                         dataset=replace(config.dataset, train_batch_size=2, test_batch_size=2))
        result = run_mnist_auto_training(config, save_training_state=True, implementation={"test": True},
                                         seed_epoch_end_callback=lambda *args: counts.append(args[1]))
        json.dumps(result.runs[0].resolved_config)
        completed = len(counts)
        run_mnist_auto_training(config, resume=True, save_training_state=True, implementation={"test": True},
                                seed_epoch_end_callback=lambda *args: counts.append(args[1]))
        assert len(counts) == completed
    assert counts == [1, 2, 1, 2]


def test_diagnostics_do_not_poison_caches_for_subsequent_training():
    from qcnn import quantum
    model = PCSQCNN(8, 10, feature_qubits=2, quantum_layers=3,
                    multiplexer_kind="local_3x3", reduce_readout_to_feature_distribution=True)
    quantum._pauli_basis_matrices.cache_clear()
    quantum._PAULI_BASIS_DEVICE_DTYPE_CACHE.clear()
    quantum._FOURIER_JUNCTION_PHASE_CACHE.clear()
    images = torch.rand(2, 8, 8)
    outputs(model.eval(), images, (4, 0), 2)
    model.train()(images).square().mean().backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
