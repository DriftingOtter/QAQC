import math
import numpy as np
import pandas as pd
import matplotlib as mp

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.primitives import StatevectorSampler


def trainableGate(bit_count, theta):
    '''
        IBM reduced gate set: {RZ, SX, CZ}

        We do this due to the combination of {RZ, RX, and X}
        can be made from {RZ, and SX} combinations directly &
        are less error prone (with RZ being strictly 0 noise)

        Since you only need one 2-qubit gate to achieve universal
        quantum computing we choose CZ as its naturally better
        supported in IBM architecture over CX that is build by a
        modulation technique over CZ's natural implementation.
    '''
    train_para = theta
    layer_count = 3  # <-- From |IBM_mininal_gate_set|

    ansatz = QuantumCircuit(bit_count, name="V")
    for layer in range(0, layer_count-1):
        # Per-bit Layer
        for bit in range(bit_count):
            ansatz.rz(phi=train_para, qubit=bit)
        # Entanglement Layer
        for bit in range(0, math.ceil((bit_count-1)/2)):
            ansatz.cz(control_qubit=bit, target_qubit=bit+1)
        # Per-bit Layer
        for bit in range(0, bit_count):
            ansatz.sx(qubit=bit)

    v = ansatz.to_gate()
    return v


def distillationCircuit(teacher_circuit, ansatz):
    teacher_bits = teacher_circuit.num_qubits
    bit_count = teacher_bits * 2

    qc = QuantumCircuit(bit_count)

    # Apply Unitaries
    for bit in range(teacher_bits):
        qc.h(bit)
    for bit in range(teacher_bits):
        qc.cx(bit, bit + teacher_bits)

    # Teacher gate on first register
    qc.append(
        teacher_circuit,
        range(teacher_bits)
    )

    # Trainable gate on second register
    if not (0 <= ansatz <= 2 * math.pi):
        raise ValueError("ansatz must be between 0 and 2π")
    tg = trainableGate(
        bit_count=teacher_bits,
        theta=ansatz
    )
    qc.append(tg, range(teacher_bits, bit_count))

    # Undo Unitaries
    for bit in range(teacher_bits-1, -1, -1):
        qc.cx(bit, bit + teacher_bits)
    for bit in range(teacher_bits):
        qc.h(bit)

    qc.measure_all()

    return qc


if __name__ == "__main__":

    # Ideal Circuit
    teacher_circuit = QuantumCircuit(2, name="U")
    teacher_circuit.h(0)
    teacher_circuit.cx(0, 1)
    tc = teacher_circuit.to_gate()

    qc = distillationCircuit(teacher_circuit=tc, ansatz=math.pi/2)
    print(qc.draw('text'))

    sampler = StatevectorSampler()
    job = sampler.run([qc], shots=2000)
    result = job.result()
    counts = result[0].data.meas.get_counts()

    print(f"Measurement Counts: {counts}")
