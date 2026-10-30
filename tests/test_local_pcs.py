"""Check finite support against pixel shifts, independently of the FFT formula."""

import pytest
import torch

from qcnn import LocalModeMultiplexer2D, PCSQCNN, RegisterLayout2D
from qcnn.quantum import _apply_iqft_2d, _apply_qft_2d
from tests.script_loading import load_run_script


@pytest.fixture
def benchmark_matmul_precision():
    # Read the actual benchmark configuration so reverting its precision also
    # regresses the numerical checks; always restore process-wide Torch state.
    script = load_run_script("local_pcs_precision_config", "prepare_local_pcsqcnn_translated_mnist_data.py")
    precision = script.iter_local_pcsqcnn_training_configs()[0].training.torch_matmul_precision
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _coins(coefficients):
    # Independent basis construction, including the identity coefficient.
    paulis = torch.tensor([
        [[1, 0], [0, 1]], [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]], [[1, 0], [0, -1]],
    ], dtype=torch.complex128, device=coefficients.device)
    basis = torch.stack([torch.kron(a, b) for a in paulis for b in paulis])
    generators = torch.einsum("...p,pij->...ij", coefficients.to(basis.dtype), basis)
    return torch.linalg.matrix_exp(1j * generators)


def _conditional_roll(state, *, axis, shift, mask):
    mask = torch.tensor(mask, dtype=torch.bool, device=state.device)
    return torch.where(mask, torch.roll(state, shift, axis), state)


def _spatial_reference(state, coefficients, selected_x, selected_y):
    """Direct S_y* C_y S_y S_x* C_x S_x with explicit condition branches."""
    coins = _coins(coefficients)
    remaining_x = state.shape[2] // selected_x
    remaining_y = state.shape[4] // selected_y
    x_branches = []
    for cx in range(state.shape[2]):
        y_branches = []
        for cy in range(state.shape[4]):
            branch = coins[cx // remaining_x, cy // remaining_y]
            out = state[:, :, cx, :, cy, :]
            for axis, coin, mask in (
                (1, branch[0], (False, False, True, True)),
                (2, branch[1], (False, True, False, True)),
            ):
                out = _conditional_roll(out, axis=axis, shift=1, mask=mask)
                out = out @ coin.T
                out = _conditional_roll(out, axis=axis, shift=-1, mask=mask)
            y_branches.append(out)
        x_branches.append(torch.stack(y_branches, dim=3))
    return torch.stack(x_branches, dim=2)


@pytest.mark.parametrize("conditions,selected", [((0, 0), (0, 0)), ((2, 1), (1, 1)), ((1, 2), (0, 1))])
def test_fourier_layer_and_gradients_match_independent_spatial_shifts(conditions, selected):
    torch.manual_seed(193)
    layout = RegisterLayout2D(image_size=16, feature_qubits=2,
                             x_condition_qubits=conditions[0], y_condition_qubits=conditions[1])
    layer = LocalModeMultiplexer2D(layout, x_condition_qubits_to_use=selected[0],
                                  y_condition_qubits_to_use=selected[1], dtype=torch.float64)
    shape = (2, layout.x_active_dim, 2**conditions[0], layout.y_active_dim, 2**conditions[1], 4)
    state = torch.randn(shape, dtype=torch.complex128, requires_grad=True)
    reference_state = state.detach().clone().requires_grad_()
    reference_parameters = layer.block_parameters.detach().clone().requires_grad_()
    actual = _apply_iqft_2d(layer(_apply_qft_2d(state)))
    expected = _spatial_reference(reference_state, reference_parameters, 2**selected[0], 2**selected[1])
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
    probe = torch.randn(shape, dtype=torch.complex128)
    actual_grad = torch.autograd.grad((actual * probe.conj()).real.sum(), (state, layer.block_parameters))
    expected_grad = torch.autograd.grad((expected * probe.conj()).real.sum(), (reference_state, reference_parameters))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=2e-11, rtol=2e-11)


def test_unitarity_finite_support_and_translation_equivariance():
    torch.manual_seed(12)
    layout = RegisterLayout2D(image_size=16, feature_qubits=2, x_condition_qubits=1, y_condition_qubits=1)
    layer = LocalModeMultiplexer2D(layout, x_condition_qubits_to_use=1,
                                  y_condition_qubits_to_use=1, dtype=torch.float64)
    blocks = layer.unitary_blocks()
    identity = torch.eye(4, dtype=torch.complex128).expand_as(blocks)
    torch.testing.assert_close(blocks.mH @ blocks, identity, atol=2e-13, rtol=2e-13)

    # Every feature and pooling branch is probed at a periodic boundary.
    state = torch.zeros(16, 8, 2, 8, 2, 4, dtype=torch.complex128)
    for b in range(16):
        state[b, 0, b // 8, 0, (b // 4) % 2, b % 4] = 1
    apply = lambda s: _apply_iqft_2d(layer(_apply_qft_2d(s)))
    out = apply(state)
    allowed = torch.zeros(8, 8, dtype=torch.bool)
    for x in (7, 0, 1):
        for y in (7, 0, 1):
            allowed[x, y] = True
    amplitudes = out.permute(0, 2, 4, 5, 1, 3)
    assert amplitudes[..., ~allowed].abs().max() < 2e-13
    assert out[:, 1].abs().max() > 1e-3  # A nontrivial spatial operation.
    torch.testing.assert_close(out.abs().square().sum((1, 2, 3, 4, 5)), torch.ones(16, dtype=torch.float64),
                               atol=2e-13, rtol=2e-13)
    for axis, shift in ((1, 3), (3, -2)):
        torch.testing.assert_close(apply(state.roll(shift, axis)), out.roll(shift, axis), atol=2e-13, rtol=2e-13)


def _model(**kwargs):
    with pytest.warns(UserWarning, match="1/sqrt"):
        return PCSQCNN(image_size=32, num_classes=10, feature_qubits=2, quantum_layers=3,
                       multiplexer_kind="local_3x3", **kwargs)


def test_main_architecture_parameter_counts():
    model = _model()
    assert [m.block_parameters.numel() for m in model.multiplexers] == [32, 128, 128]
    assert [tuple(m.block_parameters.shape) for m in model.multiplexers] == [(1, 1, 2, 16), (2, 2, 2, 16), (2, 2, 2, 16)]
    assert model.readout_spatial_shape == (8, 8, 4)
    assert sum(p.numel() for p in model.classifier.parameters()) == 2570
    assert sum(p.numel() for p in model.parameters()) == 2858


def test_multilayer_pooling_junction_readout_and_gradients():
    torch.manual_seed(19)
    # Also checks that .double() updates the parameterless FRQI encoder.
    reduced = _model().double()
    explicit = _model(dtype=torch.float64, use_reduced_fourier_junction=False)
    explicit.load_state_dict(reduced.state_dict())
    images = torch.rand(2, 32, 32, dtype=torch.float64)
    probabilities = reduced.exact_quantum_readout_probabilities(images)
    torch.testing.assert_close(probabilities.sum((1, 2, 3)), torch.ones(2, dtype=torch.float64), atol=1e-13, rtol=1e-13)
    actual = reduced(images)
    expected = explicit(images)
    assert actual.shape == (2, 10) and actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
    labels = torch.tensor([2, 7])
    torch.nn.functional.cross_entropy(actual, labels).backward()
    torch.nn.functional.cross_entropy(expected, labels).backward()
    for a, b in zip(reduced.parameters(), explicit.parameters()):
        assert a.grad is not None and torch.isfinite(a.grad).all()
        torch.testing.assert_close(a.grad, b.grad, atol=1e-13, rtol=1e-11)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_local_pcs_cuda_matches_cpu(dtype, benchmark_matmul_precision):
    torch.manual_seed(41)
    gpu = _model(dtype=dtype).cuda()
    # Double-precision CPU reference with exactly the same stored coefficients.
    cpu = _model(dtype=torch.float64)
    cpu.load_state_dict(gpu.state_dict())
    images = torch.rand(3, 32, 32)
    expected, actual = cpu(images.double()), gpu(images.to(device="cuda", dtype=dtype))
    atol, rtol = (2e-7, 2e-4) if dtype == torch.float32 else (2e-11, 2e-11)
    torch.testing.assert_close(actual.cpu().double(), expected, atol=atol, rtol=rtol)
    labels = torch.tensor([2, 5, 8])
    torch.nn.functional.cross_entropy(expected, labels).backward()
    torch.nn.functional.cross_entropy(actual, labels.cuda()).backward()
    for a, b in zip(cpu.parameters(), gpu.parameters()):
        assert b.grad is not None and torch.isfinite(b.grad).all()
        torch.testing.assert_close(a.grad, b.grad.cpu().double(), atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_local_pcs_cuda_float32_unitarity_support_and_normalization(benchmark_matmul_precision):
    torch.manual_seed(41)
    model = _model().cuda()
    assert model.classifier.weight.dtype == torch.float32
    with torch.no_grad():
        for layer in model.multiplexers:
            # Audit GPU-produced blocks on CPU in double precision, so the
            # checker itself cannot hide/amplify reduced-precision GEMM errors.
            blocks = layer.unitary_blocks().cpu().to(torch.complex128)
            identity = torch.eye(4, dtype=torch.complex128).expand_as(blocks)
            torch.testing.assert_close(blocks.mH @ blocks, identity, atol=5e-5, rtol=0)
            kernel = torch.fft.ifft2(blocks, dim=(2, 3))
            nx, ny = blocks.shape[2:4]
            inside = torch.zeros(nx, ny, dtype=torch.bool)
            for x in (nx - 1, 0, 1):
                for y in (ny - 1, 0, 1):
                    inside[x, y] = True
            assert kernel[:, :, ~inside].abs().max().item() < 2e-7

        images = torch.rand(3, 32, 32, device="cuda")
        probabilities = model.exact_quantum_readout_probabilities(images)
        assert torch.isfinite(probabilities).all()
        torch.testing.assert_close(probabilities.sum((1, 2, 3)), torch.ones(3, device="cuda"),
                                   atol=5e-5, rtol=0)


@pytest.mark.parametrize("features", [1, 3])
def test_local_requires_two_features(features):
    with pytest.raises(ValueError, match="exactly two"):
        LocalModeMultiplexer2D(RegisterLayout2D(image_size=8, feature_qubits=features))


def test_unknown_multiplexer_rejected():
    with pytest.warns(UserWarning, match="1/sqrt"), pytest.raises(ValueError, match="multiplexer_kind"):
        PCSQCNN(image_size=8, num_classes=10, multiplexer_kind="typo")


def _apply_small_gate(state, gate, wires):
    """Apply a gate to explicit binary axes; axis zero indexes input probes."""
    targets = [wire + 1 for wire in wires]
    order = [i for i in range(state.ndim) if i not in targets] + targets
    moved = state.permute(order)
    result = (moved.reshape(-1, gate.shape[0]) @ gate.T).reshape(moved.shape)
    return result.permute([order.index(i) for i in range(state.ndim)])


@pytest.mark.parametrize("image_size,conditions", [(4, (0, 1)), (8, (0, 1))])
def test_binary_phase_circuit_matches_local_fourier_blocks(image_size, conditions):
    """Check every basis state and pooling branch, including coin phases.

    Phase ramps are compiled into CNOTs and single-qubit gates independently
    of the simulator's diagonal formula. Coins remain exact U(4) matrices;
    their three-CNOT synthesis bound is the cited general two-qubit result.
    """
    resources = load_run_script("local_pcs_resource_counts", "count_pcs_resources.py")
    torch.manual_seed(57)
    layout = RegisterLayout2D(image_size=image_size, feature_qubits=2,
                             x_condition_qubits=conditions[0], y_condition_qubits=conditions[1])
    layer = LocalModeMultiplexer2D(layout, x_condition_qubits_to_use=conditions[0],
                                  y_condition_qubits_to_use=conditions[1], dtype=torch.float64)
    nx, ny = layout.x_active_dim.bit_length() - 1, layout.y_active_dim.bit_length() - 1
    n = nx + ny + 2
    state = torch.eye(2**n, dtype=torch.complex128).reshape(2**n, *([2] * n))
    cnot = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0],
                         [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.complex128)
    hadamard = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / 2**0.5

    def phase(angle):
        return torch.diag(torch.exp(torch.tensor([0j, 1j * angle], dtype=torch.complex128)))

    def ramp(value, start, width, target, inverse=False):
        counts = resources.Gates()

        def apply(value, gate, wires):
            nonlocal counts
            counts += resources.Gates(cnot=1) if len(wires) == 2 else resources.Gates(one_qubit=1)
            return _apply_small_gate(value, gate, wires)

        for position in range(width):
            control = start + position  # Most significant bit first.
            angle = (1 if inverse else -1) * torch.pi / 2**position
            if position == 0:  # CP(+/- pi) is CZ, requiring one CNOT.
                value = apply(value, hadamard, [target])
                value = apply(value, cnot, [control, target])
                value = apply(value, hadamard, [target])
            else:
                value = apply(value, phase(angle / 2), [control])
                value = apply(value, phase(angle / 2), [target])
                value = apply(value, cnot, [control, target])
                value = apply(value, phase(-angle / 2), [target])
                value = apply(value, cnot, [control, target])
        return value, counts

    coins = _coins(layer.block_parameters)
    for bx in range(coins.shape[0]):
        for by in range(coins.shape[1]):
            actual, phase_gates = state, resources.Gates()
            for axis, start, width, target in [(0, 0, nx, nx + ny),
                                               (1, nx, ny, nx + ny + 1)]:
                actual, count = ramp(actual, start, width, target)
                phase_gates += count
                actual = _apply_small_gate(actual, coins[bx, by, axis], [nx + ny, nx + ny + 1])
                actual, count = ramp(actual, start, width, target, inverse=True)
                phase_gates += count
            expected = torch.einsum('xyij,bxyj->bxyi', layer.unitary_blocks()[bx, by],
                                    state.reshape(2**n, 2**nx, 2**ny, 4))
            torch.testing.assert_close(actual.reshape_as(expected), expected, atol=2e-13, rtol=2e-13)
            assert phase_gates == resources.local_phases(nx + ny)
