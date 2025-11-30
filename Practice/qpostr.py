# ------------------------------------------------------------------------------

# Una estructura de datos para circuitos cuánticos a usar es la notación
# binaria posicional para números enteros (pag 4 introQNLP.pdf).

# El protocolo de codificación de string usado es QPOSTR (Quantum positional
# string). En comp clásica un string viene codificado como [3, 1, 2], pero en
# QPOSTR, se usa 1p ⊗ 3c + 2p ⊗ 1c + 3p ⊗ 2c, las P representan posición y las
# c representan el Cth caracter.

# ------------------------------------------------------------------------------

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    # 0 --> pos0
    # 1 --> pos1
    # 2 --> chr2
    # 3 --> chr3
    # 4 --> output0
    # 5 --> output1
    # classical readout (2 output bits)
    qc = QuantumCircuit(6, 2)

    # Superposition for positions qubits
    qc.h(0)
    qc.h(1)

    # Gate for c (11)
    qc.ccx(0, 1, 2, ctrl_state="00")
    qc.ccx(0, 1, 3, ctrl_state="00")

    # Gate for a (01)
    # Remember, big endian notation
    qc.ccx(0, 1, 2, ctrl_state="01")

    # Gate for b (10)
    qc.ccx(0, 1, 3, ctrl_state="10")

    # Multicontroller for output0
    qc.x(1)
    qc.mcx([0, 1, 2], 4)
    # qc.x(1)

    # Multicontroller for output1
    # qc.x(1)
    qc.mcx([0, 1, 3], 5)
    qc.x(1)

    # Measure to classical readout. Recovers character
    # 'a' from position 01 of string "cab"
    qc.measure(4, 0)
    qc.measure(5, 1)

    qc.draw("mpl")
    plt.show()

    sim = AerSimulator()
    qct = transpile(qc, sim)
    results = sim.run(qct, shots=2048).result()

    counts = results.get_counts()
    # Little endian
    # counts_simple = {k[::-1]: v for k, v in counts.items()}
    print(counts)
