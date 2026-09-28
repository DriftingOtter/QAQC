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

    layer_count = 3
    theta_index = 0

    ansatz = QuantumCircuit(bit_count, name="V")
    for layer in range(0, layer_count):
        # Per-bit Layer
        for bit in range(bit_count):
            ansatz.rz(phi=theta[theta_index].numpy().item(), qubit=bit)
            theta_index += 1
            ansatz.sx(qubit=bit)
            ansatz.rz(phi=theta[theta_index].numpy().item(), qubit=bit)
            theta_index += 1
            ansatz.sx(qubit=bit)
            ansatz.rz(phi=theta[theta_index].numpy().item(), qubit=bit)
            theta_index += 1
        # Entanglement Layer
        if layer < layer_count-1:
            for bit in range(0, math.ceil((bit_count-1)/2)):
                ansatz.cz(control_qubit=bit, target_qubit=bit+1)

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


def printTrainingStatus(epoch, curr_loss, loss_grad, prev_ansatz, curr_ansatz):
    grad_norm = tf.linalg.norm(loss_grad).numpy().item()
    max_grad = tf.reduce_max(tf.abs(loss_grad)).numpy().item()
    mean_grad = tf.reduce_mean(tf.abs(loss_grad)).numpy().item()
    step_norm = tf.linalg.norm(curr_ansatz - prev_ansatz).numpy().item()
    dominant_param = tf.argmax(tf.abs(loss_grad)).numpy().item()

    print(f"| Epoch: {epoch}")
    print(f"| Loss: {curr_loss:.6f}")
    print(f"| Gradient: norm={grad_norm:.6f}  mean|g|={
          mean_grad:.6f}  max|g|={max_grad:.6f}  param={dominant_param}")
    print(f"| Adam Step Magnitude: {step_norm:.6f}")
    print()


if __name__ == "__main__":
    grad = math.inf
    curr_loss = math.inf
    max_epoch = 10_000

    # Hyper-parameters
    delta = 0.1
    optimizer_learning_rate = 0.01
    shots_per_epoch = 24000

    # Inital Guess
    curr_loss = math.inf
    optimizer = tf.keras.optimizers.Adam(learning_rate=optimizer_learning_rate)

    # Ideal Circuit
    teacher_circuit = QuantumCircuit(2, name="U")
    teacher_circuit.h(0)
    teacher_circuit.cx(0, 1)
    tc = teacher_circuit.to_gate()

    layer_count = 3
    parameter_count = 3 * teacher_circuit.num_qubits * layer_count
    curr_ansatz = tf.Variable(
        tf.fill([parameter_count], tf.constant(math.pi/2, dtype=tf.float32))
    )

    if tc.definition is not None:
        print("Ideal Circuit:")
        print(tc.definition.draw("text"))
        print()

    epoch = 0
    while not (curr_loss < 0.01 and abs(grad) < 0.01) and epoch <= max_epoch:
        epoch += 1

        # Calculate Current Loss
        qc_1 = distillationCircuit(
            teacher_circuit=tc, ansatz=curr_ansatz
        )
        curr_loss = runEpoch(distillationCircuit=qc_1,
                             shots_per_epoch=shots_per_epoch
                             )

        # Calculate Parametric Gradient
        loss_grad = []

        for index in range(parameter_count):
            shift = tf.one_hot(index, parameter_count,
                               dtype=tf.float32) * delta

            qc_2 = distillationCircuit(
                teacher_circuit=tc, ansatz=curr_ansatz+shift
            )
            forward_loss = runEpoch(distillationCircuit=qc_2,
                                    shots_per_epoch=shots_per_epoch
                                    )

            qc_3 = distillationCircuit(
                teacher_circuit=tc, ansatz=curr_ansatz-shift
            )
            backward_loss = runEpoch(distillationCircuit=qc_3,
                                     shots_per_epoch=shots_per_epoch
                                     )

            loss_delta = (forward_loss, backward_loss)
            loss_grad.append(calculateGradient(loss_delta, delta))

        loss_grad = tf.constant(loss_grad, dtype=tf.float32)

        # Update curr_ansatz to next optimized value
        prev_ansatz = curr_ansatz.read_value()
        optimizer.apply_gradients(
            [(loss_grad, curr_ansatz)]
        )

        grad = tf.reduce_max(tf.abs(loss_grad)).numpy().item()

        printTrainingStatus(
            epoch, curr_loss, loss_grad, prev_ansatz, curr_ansatz
        )

    print("\n========= Training Complete! =========\n")

    # Retrieve optimized circuit
    qc = trainableGate(bit_count=teacher_circuit.num_qubits,
                       theta=curr_ansatz)

    if qc.definition is not None:
        print("Optimized Circuit:")
        print(qc.definition.draw("text"))
        print()
