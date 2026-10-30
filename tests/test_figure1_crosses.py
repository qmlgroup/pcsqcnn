"""Scientific contracts of the small Figure 1 classification experiment."""
import numpy as np
import pytest
import torch

from qcnn.figure1_crosses import (CrossClassifier, audit_dataset, diagnostics,
                                  example_images, make_dataset, verify_model)


def test_all_positions_balanced_brightness_and_no_qcs_label_conflicts():
    data = make_dataset()
    audit = audit_dataset(data)
    assert audit['samples'] == 19*256
    assert audit['unique_images'] == audit['samples']
    assert audit['qcs_label_conflicts'] == 0
    assert audit['raw_summed_within_count_max_difference'] < 1e-14
    for base in range(19):
        assert len(np.unique(data['offsets'][data['base_ids'] == base], axis=0)) == 256
    assert all(abs(value-0.25) < 1e-14 for value in audit['stratum_weight'].values())


@pytest.mark.parametrize('kind', ['pcs', 'qcs'])
def test_layer_symmetry_and_common_head_decomposition(kind, tmp_path):
    torch.set_num_threads(2)
    model = CrossClassifier(kind, 7, dtype=torch.float64)
    verify_model(model)
    result = diagnostics(model, tmp_path/kind)
    assert result['map_sum_logit_error'] < 1e-11
    assert model.parameter_counts()['head'] == 5


@pytest.mark.parametrize('kind', ['pcs', 'qcs'])
def test_classification_gradient_against_finite_difference(kind):
    torch.set_num_threads(2)
    model = CrossClassifier(kind, 5, dtype=torch.float64)
    images = torch.tensor(np.stack(list(example_images().values())), dtype=torch.float64)
    labels = torch.tensor([0,1,1,1,1,0], dtype=torch.float64)
    parameter = model.layer.block_parameters if kind == 'pcs' else model.layer.angles
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(images), labels)
    loss.backward()
    index = int(parameter.grad.abs().argmax())
    derivative = float(parameter.grad.flatten()[index])
    assert abs(derivative) > 1e-8
    epsilon = 1e-5
    with torch.no_grad():
        initial = float(parameter.flatten()[index])
        parameter.flatten()[index] = initial+epsilon
        plus = torch.nn.functional.binary_cross_entropy_with_logits(model(images), labels)
        parameter.flatten()[index] = initial-epsilon
        minus = torch.nn.functional.binary_cross_entropy_with_logits(model(images), labels)
        parameter.flatten()[index] = initial
    assert float((plus-minus)/(2*epsilon)) == pytest.approx(derivative, rel=1e-5, abs=1e-7)
