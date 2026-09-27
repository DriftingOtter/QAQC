import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as mp

from qiskit import ClassicalRegister, QuantumCircuit, circuit
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

    layer_count = 3  # <-- From |IBM_mininal_gate_set|

    ansatz = QuantumCircuit(bit_count, name="V")
    for layer in range(0, layer_count-1):
        # Per-bit Layer
        for bit in range(bit_count):
            ansatz.rz(phi=theta.numpy().item(), qubit=bit)
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
        ansatz = tf.math.floormod(ansatz, 2 * math.pi)
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


def loss(zero_count, shots_per_epoch):
    loss = 1.0 - (zero_count/shots_per_epoch)
    return loss


def runEpoch(distillationCircuit, shots_per_epoch):
    sampler = StatevectorSampler()
    job = sampler.run([distillationCircuit], shots=shots_per_epoch)
    result = job.result()
    counts = result[0].data.meas.get_counts()

    # | 000...0 > state key mask for any size n
    state_mask = "0" * distillationCircuit.num_qubits
    zero_count = counts.get(state_mask, 0)

    score = loss(zero_count=zero_count, shots_per_epoch=shots_per_epoch)
    return score


def calculateGradient(diff, delta):
    return (diff[0] - diff[1])/(2*delta)


if __name__ == "__main__":
    grad = math.inf
    curr_loss = math.inf
    max_epoch = 10_000

    # Hyper-parameters
    delta = 0.01
    optimizer_learning_rate = 0.01
    curr_ansatz = tf.Variable(tf.constant(math.pi)/2.0, dtype=tf.float32)
    shots_per_epoch = 24000

    # Inital Guess
    curr_loss = math.inf
    optimizer = tf.keras.optimizers.Adam(learning_rate=optimizer_learning_rate)

    # Ideal Circuit
    teacher_circuit = QuantumCircuit(2, name="U")
    teacher_circuit.h(0)
    teacher_circuit.cx(0, 1)
    tc = teacher_circuit.to_gate()

    if tc.definition is not None:
        curr_circuit = QuantumCircuit(tc.num_qubits)
        curr_circuit.append(
            tc, [bit for bit in range(curr_circuit.num_qubits)])

        print("Ideal Circuit:")
        print(curr_circuit.draw("text"))
        print()

    epoch = 0
    while not (curr_loss < 0.01 and abs(grad) < 0.001) and epoch <= max_epoch:
        epoch += 1
        print(f"| Epoch:{epoch} \t Current Guess: {curr_ansatz.read_value()}")

        # Calculate Current Loss
        qc_1 = distillationCircuit(
            teacher_circuit=tc, ansatz=curr_ansatz
        )
        curr_loss = runEpoch(distillationCircuit=qc_1,
                             shots_per_epoch=shots_per_epoch
                             )

        # Calculate Parametric Gradient
        qc_2 = distillationCircuit(
            teacher_circuit=tc, ansatz=curr_ansatz+delta
        )
        forward_loss = runEpoch(distillationCircuit=qc_2,
                                shots_per_epoch=shots_per_epoch
                                )

        qc_3 = distillationCircuit(
            teacher_circuit=tc, ansatz=curr_ansatz-delta
        )
        backward_loss = runEpoch(distillationCircuit=qc_3,
                                 shots_per_epoch=shots_per_epoch
                                 )

        loss_delta = (forward_loss, backward_loss)
        loss_grad = calculateGradient(loss_delta, delta)

        # Update curr_ansatz to next optimized value
        optimizer.apply_gradients(
            [(loss_grad, curr_ansatz)]
        )

        grad = loss_grad

        print(
            f"| Current Loss: {curr_loss} \t Gradient: {loss_grad}"
        )
        print(
            f"| Next Guess: {curr_ansatz.read_value()}"
        )
        print()

    print("\n========= Training Complete! =========\n")

    # Retrieve optimized circuit
    qc = trainableGate(bit_count=teacher_circuit.num_qubits,
                       theta=curr_ansatz)

    if qc.definition is not None:
        curr_circuit = QuantumCircuit(qc.num_qubits)
        curr_circuit.append(
            qc, [bit for bit in range(curr_circuit.num_qubits)])

        print("Optimized Circuit:")
        print(curr_circuit.draw("text"))
        print()
