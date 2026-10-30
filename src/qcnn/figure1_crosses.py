"""Small, reproducible classification experiment for Figure 1.

Coordinates are [sample, x, y, feature]; plots transpose the two spatial axes.
This experiment trains only image labels, never feature-location targets.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from qcnn.baselines.sp_qcnn import SplitParallelStage, _euler
from qcnn.layout import RegisterLayout2D
from qcnn.quantum import LocalModeMultiplexer2D, pauli_coefficients_to_unitary

SIZE = 16
CROSS = {(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    root = Path(__file__).parent
    return {str(p.relative_to(root)): digest(p) for p in
            (Path(__file__), root / 'quantum.py', root / 'layout.py',
             root / 'baselines/sp_qcnn.py')}


def image_at(pattern, center=(2, 2)):
    image = np.zeros((SIZE, SIZE), dtype=np.float32)
    for u, v in pattern:
        image[(u + center[0]) % SIZE, (v + center[1]) % SIZE] = 0.5
    return image


def cross_count(image):
    occupied = image > 0
    matches = np.ones_like(occupied)
    for u, v in CROSS:
        matches &= np.roll(occupied, (-u, -v), axis=(0, 1))
    return int(matches.sum())


def bit_cycle_numpy(array, axis, power=1):
    permutation = np.arange(SIZE)
    for _ in range(power % 4):
        permutation = ((permutation & 1) << 3) | (permutation >> 1)
    return np.take(array, np.argsort(permutation), axis=axis)


def transform(state, kind, power=1):
    """T_x, T_y, S_x, S_y on a [B, X, Y, ...] tensor."""
    axis = 1 if kind.endswith('x') else 2
    if kind.startswith('T'):
        return torch.roll(state, power, dims=axis)
    permutation = torch.arange(SIZE, device=state.device)
    for _ in range(power % 4):
        permutation = ((permutation & 1) << 3) | (permutation >> 1)
    return state.index_select(axis, permutation.argsort())


def make_dataset():
    broken = CROSS - {(-1, 0)} | {(-1, -1)}
    negatives = []
    for _ in range(4):
        negatives.append(set(broken))
        broken = {(-v, u) for u, v in broken}
    bases = [('single_cross', image_at(CROSS), 1)]
    bases += [(f'single_broken_{i}', image_at(p), 0) for i, p in enumerate(negatives)]
    for j, displacement in enumerate(((0, 7), (6, 5))):
        second = tuple((2 + k) % SIZE for k in displacement)
        combinations = [('cc', CROSS, CROSS, 1), ('cn', CROSS, negatives[0], 1),
                        ('nc', negatives[1], CROSS, 1)]
        combinations += [(f'nn{i}', p, negatives[(i + 1) % 4], 0)
                         for i, p in enumerate(negatives)]
        for name, left, right, label in combinations:
            bases.append((f'pair{j}_{name}', np.maximum(image_at(left), image_at(right, second)), label))
    images, labels, base_ids, offsets = [], [], [], []
    for base_id, (_, image, label) in enumerate(bases):
        assert bool(cross_count(image)) == bool(label)
        for u in range(SIZE):
            for v in range(SIZE):
                images.append(np.roll(image, (u, v), axis=(0, 1)))
                labels.append(label)
                base_ids.append(base_id)
                offsets.append((u, v))
    images, labels = np.asarray(images), np.asarray(labels, dtype=np.float32)
    pixels = (images > 0).sum((1, 2))
    weights = np.empty(len(labels), dtype=np.float64)
    # Each label and occupied-pixel-count stratum carries exactly 1/4 mass.
    for count in (5, 10):
        for label in (0, 1):
            selected = (pixels == count) & (labels == label)
            weights[selected] = 1 / (4 * selected.sum())
    return dict(images=images, labels=labels, weights=weights,
                base_ids=np.array(base_ids), offsets=np.array(offsets), pixels=pixels,
                names=np.array([b[0] for b in bases]))


def audit_dataset(data):
    # All S_x^a S_y^b images are generated exactly, without learned features.
    by_image = {}
    duplicates, conflicts, examples = 0, 0, []
    for i, image in enumerate(data['images']):
        key = np.packbits(image > 0).tobytes()
        label = int(data['labels'][i])
        if key in by_image:
            duplicates += 1
            assert by_image[key][0] == label
        by_image[key] = (label, i)
    for label, i in by_image.values():
        image = data['images'][i]
        for a in range(4):
            sx = bit_cycle_numpy(image, 0, a)
            for b in range(4):
                key = np.packbits(bit_cycle_numpy(sx, 1, b) > 0).tobytes()
                other = by_image.get(key)
                if other is not None and other[0] != label:
                    conflicts += 1
                    if len(examples) < 10:
                        examples.append(dict(source=i, target=other[1], sx=a, sy=b))
    # These raw summed FRQI features depend only on count, not arrangement.
    angle = np.pi * data['images'].astype(np.float64)
    raw = np.stack((np.sin(angle)**2, np.cos(angle)**2), -1).mean((1, 2))
    raw_gap = 0.0
    for count in (5, 10):
        values = raw[data['pixels'] == count]
        raw_gap = max(raw_gap, float(np.ptp(values, axis=0).max()))
    return dict(samples=len(data['labels']), base_patterns=len(data['names']),
                all_translations=SIZE**2, unique_images=len(by_image), duplicate_images=duplicates,
                qcs_label_conflicts=conflicts, conflict_examples=examples,
                raw_summed_within_count_max_difference=raw_gap,
                raw_summed_optimal_weighted_accuracy=0.5,
                stratum_weight={f'{c}_pixels_label{y}': float(data['weights'][
                    (data['pixels'] == c) & (data['labels'] == y)].sum())
                    for c in (5, 10) for y in (0, 1)})


class CrossClassifier(nn.Module):
    """One local PCS layer or one short QCS stage; common summed linear head.

The QCS stage reuses the existing sp-QCNN equivariant gates, without splitting
or pooling. Two joint blocks and final feature rotations give 46 coefficients.
The local PCS layer has 32 coefficients. Both heads have five coefficients.
"""
    def __init__(self, kind, seed=0, dtype=torch.float32, qcs_depth=2):
        super().__init__()
        if kind not in ('pcs', 'qcs'):
            raise ValueError(kind)
        self.kind, self.qcs_depth = kind, qcs_depth
        torch.manual_seed(seed)
        if kind == 'pcs':
            self.layer = LocalModeMultiplexer2D(RegisterLayout2D(SIZE, 2), dtype=dtype)
        else:
            self.layer = SplitParallelStage(4, depth=qcs_depth, stage='joint',
                                            init_scale=1.0, dtype=dtype)
            self.final_feature_angles = nn.Parameter(torch.empty(2, 3, dtype=dtype))
        with torch.no_grad():
            for p in self.parameters():
                p.normal_(0, 0.3)
        # The head has the same initialization in both architectures.
        torch.manual_seed(seed + 10000)
        self.head = nn.Linear(4, 1, dtype=dtype)
        nn.init.normal_(self.head.weight, 0, 0.1)
        nn.init.zeros_(self.head.bias)

    def encode(self, images):
        angle = images.to(self.head.weight.dtype) * math.pi
        zero = torch.zeros_like(angle)
        return torch.stack((torch.sin(angle), torch.cos(angle), zero, zero), -1).to(
            torch.complex128 if angle.dtype == torch.float64 else torch.complex64) / SIZE

    def core(self, state):
        if self.kind == 'pcs':
            fourier = torch.fft.fft2(state, dim=(1, 2), norm='ortho')
            blocks = self.layer.unitary_blocks()[0, 0]
            out = torch.einsum('xyij,bxyj->bxyi', blocks, fourier)
            return torch.fft.ifft2(out, dim=(1, 2), norm='ortho')
        out = self.layer(state.reshape(-1, SIZE * SIZE * 4)).reshape(-1, SIZE, SIZE, 4)
        gate = torch.kron(_euler(self.final_feature_angles[0]), _euler(self.final_feature_angles[1]))
        return out @ gate.T

    def probabilities(self, output):
        return output.real.square() + output.imag.square()

    def logits_encoded(self, encoded):
        blank = self.encode(encoded.real.new_zeros(1, SIZE, SIZE))
        output = self.core(torch.cat((blank, encoded)))
        features = self.probabilities(output).sum((1, 2))
        # Fixed scaling/blank centering improves conditioning. With a free
        # bias this is exactly the same affine family on summed probabilities.
        return self.head(SIZE**2 * (features[1:] - features[:1])).squeeze(-1)

    def forward(self, images):
        return self.logits_encoded(self.encode(images))

    def parameter_counts(self):
        head = sum(p.numel() for p in self.head.parameters())
        total = sum(p.numel() for p in self.parameters())
        return dict(quantum=total-head, head=head, total=total)


def direct_pcs(model, state):
    """Independent position-space application of the published shift/coin layer."""
    coins = pauli_coefficients_to_unitary(model.layer.block_parameters)[0, 0]
    out = state
    for axis, coin, selected in ((1, coins[0], (0, 0, 1, 1)),
                                 (2, coins[1], (0, 1, 0, 1))):
        mask = torch.tensor(selected, dtype=torch.bool, device=state.device)
        out = torch.where(mask, torch.roll(out, 1, dims=axis), out)
        out = out @ coin.T
        out = torch.where(mask, torch.roll(out, -1, dims=axis), out)
    return out


@torch.no_grad()
def verify_model(model):
    model = model.to(dtype=torch.float64)
    device = model.head.weight.device
    generator = torch.Generator(device=device).manual_seed(20261026)
    state = torch.randn(3, SIZE, SIZE, 4, dtype=torch.complex128, device=device, generator=generator)
    state /= state.flatten(1).norm(dim=1)[:, None, None, None]
    output = model.core(state)
    result = {'norm_error': float((output.abs().square().sum((1,2,3))-1).abs().max())}
    for action in ('Tx', 'Ty', 'Sx', 'Sy'):
        result[action] = float((model.core(transform(state, action)) - transform(output, action)).abs().max())
    if model.kind == 'pcs':
        result['direct_spatial_error'] = float((output-direct_pcs(model, state)).abs().max())
        impulse = torch.zeros_like(state[:1]); impulse[0, 7, 7, 0] = 1
        response = model.core(impulse)
        outside = torch.ones(SIZE, SIZE, dtype=torch.bool, device=device)
        outside[6:9, 6:9] = False
        result['outside_radius_one'] = float(response[:, outside].abs().max())
    required = ['norm_error'] + (['Tx', 'Ty', 'direct_spatial_error', 'outside_radius_one']
                                if model.kind == 'pcs' else ['Sx', 'Sy'])
    if max(result[k] for k in required) > 1e-10:
        raise AssertionError(result)
    return result


def example_images():
    one = image_at(CROSS)
    two = image_at(CROSS, (2, 13))
    pair = np.maximum(one, two)
    return {'blank': np.zeros_like(pair), 'single_top': one, 'single_bottom': two,
            'pair': pair, 'T3_pair': np.roll(pair, 3, axis=0),
            'S3_pair': bit_cycle_numpy(pair, 0, 3)}


@torch.no_grad()
def diagnostics(model, output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(dtype=torch.float64)
    device = model.head.weight.device
    examples = example_images()
    names = list(examples)
    images = torch.tensor(np.stack(list(examples.values())), dtype=torch.float64, device=device)
    encoded = model.encode(images)
    states = model.core(encoded)
    probs = model.probabilities(states)
    maps = SIZE**2 * torch.einsum('bxyf,f->bxy', probs-probs[:1], model.head.weight[0])
    logits = maps.sum((1,2)) + model.head.bias[0]
    check = verify_model(model)
    pair = names.index('pair')
    for action, name in [('Tx', 'T3_pair'), ('Sx', 'S3_pair')]:
        index = names.index(name)
        target = transform(states[pair:pair+1], action, 3)
        target_map = transform(maps[pair:pair+1], action, 3)
        check[action+'_pictured_state_linf'] = float((states[index:index+1]-target).abs().max())
        check[action+'_pictured_map_linf'] = float((maps[index:index+1]-target_map).abs().max())
        check[action+'_logit_difference'] = float((logits[index]-logits[pair]).abs())
    check['map_sum_logit_error'] = float((model(images)-logits).abs().max())
    # Compare local output near each isolated cross to the same region in the pair.
    for name, u, v in [('single_top', 2, 2), ('single_bottom', 2, 13)]:
        index = names.index(name)
        check[name+'_pair_local_error'] = float((states[pair,u-1:u+2,v-1:v+2]
                                               -states[index,u-1:u+2,v-1:v+2]).abs().max())
    if model.kind == 'pcs' and max(check[k] for k in (
        'Tx_pictured_state_linf', 'Tx_pictured_map_linf', 'single_top_pair_local_error',
        'single_bottom_pair_local_error')) > 1e-9:
        raise AssertionError(check)
    if model.kind == 'qcs' and max(check['Sx_pictured_state_linf'], check['Sx_pictured_map_linf']) > 1e-9:
        raise AssertionError(check)
    check['examples'] = {name: {'logit': float(logits[i]), 'probability_cross': float(logits[i].sigmoid()),
                              'actual_cross_count': cross_count(examples[name])}
                         for i, name in enumerate(names)}
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in
              [('images', images), ('inputs', encoded), ('states', states), ('probabilities', probs),
               ('contributions', maps), ('logits', logits)]}
    arrays['names'] = np.asarray(names)
    for action in ('Tx', 'Sx'):
        arrays[action+'_transported_state'] = transform(states[pair:pair+1], action, 3)[0].cpu().numpy()
        arrays[action+'_transported_map'] = transform(maps[pair:pair+1], action, 3)[0].cpu().numpy()
    np.savez_compressed(output_dir/'states.npz', **arrays)
    write_json(output_dir/'diagnostics.json', check)
    return check
