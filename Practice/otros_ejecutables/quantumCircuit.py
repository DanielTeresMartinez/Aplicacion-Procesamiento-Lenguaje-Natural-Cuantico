from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
import numpy as np
from qiskit_aer import AerSimulator

def simPrintVals(qc, shotsRun: int=1, text: str="result simulation"):
    sim = AerSimulator()
    results = sim.run(transpile(qc), shots=shotsRun).result()
    cnt = results.get_counts()
    print(text, next(iter(cnt)))


def gatesAND():
    # Use CCNOT to verify both states are |1>
    qc = QuantumCircuit(3, 1)
    qc.ccx(0, 1, 2)
    return qc

def gatesOR(qc):
    # Flip qubit 2 if one of the inputs it's |1>.
    qc.cx(0, 2)
    qc.cx(1, 2)
    # For last case, flip again qubit 2 with CCNOT
    qc.ccx(0, 1, 2)

def gatesXOR(qc):
    # If only 1 qubit is |1>, then qubit num 2 should be |1>.
    qc.cx(0, 1)
    qc.cx(1, 2)


if __name__ == "__main__":
    # Use preety print for arrays
    np.set_printoptions(formatter={'complex_kind': lambda x: f"{x:10.4g}"})
    init_vals = [[0, 0], [0, 1], [1, 0], [1, 1]]
    qc = QuantumCircuit(3, 1)
    qc.h(0)
    qc.h(1)
    # Apply instructions from one circuit to the other circuit specified in this
    # function
    qc = qc.compose(gatesAND())
    qc.save_statevector()
    
    sim = AerSimulator()
    qct = transpile(qc, sim)
    sv = sim.run(qct, shots=1).result().get_statevector()

    for ket in sv.to_dict():
        # Ya que está en little endian
        print(ket[::-1])

