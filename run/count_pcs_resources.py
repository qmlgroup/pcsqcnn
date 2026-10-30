#!/usr/bin/env python3
"""Reproduce the constructive gate bounds in manuscript Table 1.

Basis: CNOT and arbitrary one-qubit gates; each gate in the stated
factorization counts once, without merging adjacent gates. No routing,
measurement latency, or repeated shots are included. The model has a 32x32
canvas, three layers and two feature qubits. Feed-forward counts bound the
most expensive pooling outcomes. Fourier bit reversals are wire relabelings.

Multiplexers follow Bergholm et al. (PRA 71, 052330) and the cosine-sine
recursion of Shende et al. (TCAD 25, 1000). Uncontrolled two-qubit coins use
the 3-CNOT / 15-one-qubit construction of Vatan and Williams (PRA 69, 032315).
"""

import argparse
from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class Gates:
    cnot: int = 0
    one_qubit: int = 0

    def __add__(self, other):
        return Gates(self.cnot + other.cnot, self.one_qubit + other.one_qubit)

    def __mul__(self, factor):
        return Gates(self.cnot * factor, self.one_qubit * factor)


CP = Gates(2, 3)
CZ = Gates(1, 2)
COIN = Gates(3, 15)


def rotation(controls):
    """Gray-code uniformly controlled Ry/Rz, including zero controls."""
    return Gates(2**controls if controls else 0, 2**controls)


def diagonal(qubits):
    return sum((rotation(j) for j in range(qubits)), Gates())


def multiplexer(controls, targets):
    """Count both gate types, including the residual diagonal unitary."""
    if targets == 1:
        return Gates(2**controls - 1, 2**controls) + diagonal(controls + 1)
    return multiplexer(controls + 1, targets - 1) * 2 + rotation(controls + targets - 1)


def general_layer(controls):
    # Preserve the paper's conservative rounding of the two-target bounds:
    # CNOT <= 14*2**k - 6; one-qubit <= 14*2**k - 2.
    return Gates(14 * 2**controls, 14 * 2**controls)


def local_phases(address_qubits):
    # Four CP(pi) factors, with two Hadamards per CZ implementation.
    return CZ * 4 + CP * (2 * address_qubits - 4)


def local_layer(address_qubits, pooling_controls=0):
    coin = multiplexer(pooling_controls, 2) if pooling_controls else COIN
    return local_phases(address_qubits) + coin * 2


def qft(qubits):
    return Gates(0, qubits) + CP * (qubits * (qubits - 1) // 2)


def resource_rows():
    """Return (label, feed-forward bounds, deferred bounds)."""
    loading = Gates(1024, 1024 + 10)
    fourier = (qft(5) + qft(3)) * 2
    # Two Hadamards per pooling step and 4+4+3+3 conditional phases.
    junction_ff = Gates(0, 4 + 14)
    junction_deferred = Gates(0, 4) + CP * 14
    general_ff = sum((general_layer(k) for k in (10, 8, 6)), Gates())
    general_deferred = sum((general_layer(k) for k in (10, 10, 8)), Gates())
    local_ff = sum((local_layer(k) for k in (10, 8, 6)), Gates())
    local_deferred = local_layer(10) + local_layer(8, 2) + local_layer(6, 2)
    fixed_ff = loading + fourier + junction_ff
    fixed_deferred = loading + fourier + junction_deferred
    return [
        ('FRQI loading', loading, loading),
        ('Initial QFTs and final IQFTs', fourier, fourier),
        ('Reduced pooling junctions', junction_ff, junction_deferred),
        ('General PCS convolutions', general_ff, general_deferred),
        ('Radius-one PCS convolutions', local_ff, local_deferred),
        ('Complete general PCS circuit', fixed_ff + general_ff, fixed_deferred + general_deferred),
        ('Complete radius-one PCS circuit', fixed_ff + local_ff, fixed_deferred + local_deferred),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='Print machine-readable counts.')
    args = parser.parse_args()
    rows = resource_rows()
    if args.json:
        print(json.dumps({name: {'feed_forward': asdict(ff), 'deferred': asdict(dm)}
                          for name, ff, dm in rows}, indent=2))
    else:
        print('| Circuit component | Feed-forward CNOT | Feed-forward 1q | Deferred CNOT | Deferred 1q |')
        print('| --- | ---: | ---: | ---: | ---: |')
        for name, ff, dm in rows:
            print(f'| {name} | {ff.cnot:,} | {ff.one_qubit:,} | {dm.cnot:,} | {dm.one_qubit:,} |')


if __name__ == '__main__':
    main()
