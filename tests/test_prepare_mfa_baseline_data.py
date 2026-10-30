from pathlib import Path

from tests.script_loading import load_run_script


prepare_mfa = load_run_script("prepare_mfa_baseline_data_module", "prepare_mfa_baseline_data.py")


def test_mfa_regimes_match_existing_pcs_dataset_protocols(tmp_path: Path) -> None:
    common = {"data_root": tmp_path / "data", "artifacts_root": tmp_path / "artifacts"}
    translated = prepare_mfa.make_config(regime="translated", **common)
    centered = prepare_mfa.make_config(regime="centered", **common)

    assert (
        translated.dataset.image_size,
        translated.dataset.scaled_image_size,
        translated.dataset.max_offset,
        translated.dataset.samples_per_class,
        translated.dataset.train_batch_size,
    ) == (32, 16, 8, 1000, 256)
    assert (
        centered.dataset.image_size,
        centered.dataset.scaled_image_size,
        centered.dataset.max_offset,
        centered.dataset.samples_per_class,
        centered.dataset.train_batch_size,
    ) == (32, 32, 0, None, 512)
    assert translated.training.num_epochs == centered.training.num_epochs == 2000
    assert translated.seeds.seed_count == centered.seeds.seed_count == 3
    assert translated.optimizer.kind == centered.optimizer.kind == "mfa_adam"


def test_mfa_centered_pilot_can_use_small_training_subset(tmp_path: Path) -> None:
    config = prepare_mfa.make_config(
        regime="centered",
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        samples_per_class=256,
        train_batch_size=64,
        epochs=2,
    )
    assert config.dataset.samples_per_class == 256
    assert config.dataset.train_batch_size == 64
    assert config.training.num_epochs == 2
