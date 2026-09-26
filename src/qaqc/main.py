import math
import numpy as np
import pandas as pd
import matplotlib as mp
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.primitives import StatevectorSampler

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


def trainableGate(bit_count):
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
    theta = math.pi/2
    layer_count = 3  # <-- From |IBM_mininal_gate_set|

    ansatz = QuantumCircuit(bit_count, name="tG")
    for layer in range(0, layer_count-1):
        # Per-bit Layer
        for bit in range(bit_count-1):
            ansatz.rz(phi=theta, qubit=bit)
        # Entanglement Layer
        for bit in range(0, math.ceil((bit_count-1)/2)):
            ansatz.cz(control_qubit=bit, target_qubit=bit+1)
        # Per-bit Layer
        for bit in range(0, bit_count-1):
            ansatz.sx(qubit=bit)

    tG = ansatz.to_gate()
    return (tG, theta)


qc = QuantumCircuit(2)
cr = ClassicalRegister(2)

# setup unitaries
qc.h(0)
qc.cx(0, 1)

# distillation gate
qc.x(0)

# train-able gate
tg, theta = trainableGate(1)
qc.append(tg, [1])

# undo unitaries
qc.cx(0, 1)
qc.h(0)

qc.measure_all()

print(qc.draw(output="text"))


sampler = StatevectorSampler()
job = sampler.run([qc], shots=2000)
result = job.result()
data = result[0]
counts = data.data.meas.get_counts()

print(f"Measurement Counts: {counts}")
