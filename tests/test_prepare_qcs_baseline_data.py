from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from qcnn import automation
from qcnn.data import PreparedMnistSplits, TensorImageDataset
from qcnn.model_spec import instantiate_model
from qcnn.serialization import load_model_checkpoint
from tests.script_loading import load_run_script


script = load_run_script("prepare_qcs_baselines_for_test", "prepare_qcs_baseline_data.py")


def test_main_protocol_and_only_one_selected_model():
    config = script.make_config()
    assert (config.dataset.image_size, config.dataset.scaled_image_size, config.dataset.max_offset,
            config.dataset.samples_per_class, config.dataset.train_batch_size, config.dataset.test_batch_size) == (32, 16, 8, 1000, 256, 1600)
    assert config.model.class_name == "SpQCNN" and config.model.constructor_kwargs["depth"] == 10
    assert config.training.num_epochs == 2000
    assert (config.seeds.base_seed, config.seeds.seed_count) == (0, 3)
    assert (config.optimizer.kind, config.optimizer.learning_rate, config.optimizer.weight_decay) == ("adam", 0.03, 0)
    assert config.training.test_evaluation_interval_epochs == 10
    for name, count in (("sp_qcnn", 2932), ("pcs_local", 2858), ("hur_x1", 362), ("hur_x8", 2826), ("hur_l4", 452), ("hur_l16", 812)):
        selected_config = script.make_config(model=name)
        assert selected_config.training.torch_matmul_precision == "highest"
        model = instantiate_model(selected_config.model)
        assert sum(p.numel() for p in model.parameters()) == count


@pytest.mark.parametrize("model_name,class_name", [("sp_qcnn", "SpQCNN"), ("pcs_local", "PCSQCNN")])
def test_dry_run_does_not_train(monkeypatch, capsys, model_name, class_name):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not train")
    monkeypatch.setattr(script, "run_mnist_auto_training", forbidden)
    script.main(["--model", model_name, "--dry-run"])
    config = json.loads(capsys.readouterr().out)
    assert config["model"]["class_name"] == class_name
    assert config["training"]["torch_matmul_precision"] == "highest"


@pytest.mark.parametrize("model_name", ["sp_qcnn", "pcs_local"])
def test_full_size_pipeline_resume_and_checkpoint(monkeypatch, tmp_path, model_name):
    # Synthetic fixture, not a benchmark score; actual 12-qubit models.
    generator = torch.Generator().manual_seed(73)
    images = torch.rand(6, 32, 32, generator=generator)
    splits = PreparedMnistSplits(
        TensorImageDataset(images[:4], torch.tensor([0, 3, 5, 9])),
        TensorImageDataset(images[4:], torch.tensor([2, 7])),
    )
    monkeypatch.setattr(automation, "prepare_mnist_splits", lambda **kwargs: splits)
    config = script.make_config(model=model_name, artifacts_root=tmp_path, epochs=2, seed_count=1,
                                directory_name="complete", device="cpu", train_batch_size=2)
    provenance = script.implementation_record(model_name)
    complete_result = automation.run_mnist_auto_training(config, save_training_state=True, implementation=provenance)
    complete = complete_result.runs[0]
    restart_config = replace(config, output=replace(config.output, directory_name="restart"))

    def interrupt(seed, epoch, train_summary, test_summary, history):
        if epoch == 1:
            raise InterruptedError("test interruption after checkpoint")

    with pytest.raises(InterruptedError):
        automation.run_mnist_auto_training(restart_config, save_training_state=True, implementation=provenance,
                                          seed_epoch_end_callback=interrupt)
    record_path = tmp_path / "restart" / "implementation.json"
    original_record = record_path.read_bytes()
    resumed = automation.run_mnist_auto_training(restart_config, resume=True, save_training_state=True,
                                                implementation=provenance).runs[0]
    assert resumed.history.train_loss == complete.history.train_loss
    assert resumed.final_test_summary == complete.final_test_summary
    for name, expected in complete.runner.model.state_dict().items():
        torch.testing.assert_close(resumed.runner.model.state_dict()[name], expected, atol=0, rtol=0)
    restored = load_model_checkpoint(resumed.artifacts.final_checkpoint_path, map_location="cpu")
    torch.testing.assert_close(restored.model(images), resumed.runner.model(images), atol=0, rtol=0)
    assert restored.training_state["data_fingerprints"].keys() == {"train", "test"}
    assert restored.training_state["optimizer_state"]["state"]
    assert restored.training_state["implementation_sha256"] == hashlib.sha256(original_record).hexdigest()
    assert record_path.read_bytes() == original_record
    assert provenance["model"] == model_name
    assert provenance["versions"]["torch"] == str(torch.__version__)
    assert "uv.lock" in provenance["source_sha256"]


def test_cli_captures_source_before_training(monkeypatch, tmp_path):
    # Edit a temporary source tree while the simulated training call runs.
    for name in ("src/qcnn/baselines/sp_qcnn.py", "pyproject.toml", "uv.lock", "run/prepare_qcs_baseline_data.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original source\n")
    monkeypatch.setattr(script, "__file__", str(tmp_path / "run/prepare_qcs_baseline_data.py"))
    source = tmp_path / "src/qcnn/baselines/sp_qcnn.py"
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    received = []
    def train(config, *, implementation, **kwargs):
        assert config.seeds.seed_count == 3
        assert kwargs["save_training_state"] is True
        received.append(implementation)
        source.write_text("edited during training\n")
        return SimpleNamespace(manifest_path=tmp_path / "manifest.json")
    monkeypatch.setattr(script, "run_mnist_auto_training", train)
    script.main(["--epochs", "2"])
    assert received[0]["source_sha256"]["src/qcnn/baselines/sp_qcnn.py"] == original_hash
    assert script.implementation_record("sp_qcnn")["source_sha256"]["src/qcnn/baselines/sp_qcnn.py"] != original_hash
