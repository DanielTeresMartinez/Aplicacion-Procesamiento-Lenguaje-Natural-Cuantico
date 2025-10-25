from qiskit import QuantumCircuit, transpile
import numpy as np
from qiskit_aer import AerSimulator

def exQCAndSim():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.z(0)
    qc.h(0)
    qc.measure_all()

    sim = AerSimulator()
    # transpile convierte todo el circuito a "código máquina".
    # El .run envía un job, pero no interesa nada 
    results = sim.run(transpile(qc), shots=10).result()
    counts = results.get_counts()
    print(counts)


def simPrintVals(qc, shotsRun: int=1, text: str="result simulation"):
    sim = AerSimulator()
    results = sim.run(transpile(qc), shots=shotsRun).result()
    cnt = results.get_counts()
    print(text, next(iter(cnt)))

def resetQc(qc):
    # Reset all qubits to |0> at the start of the iteration
    for q in range(qc.num_qubits):
        qc.reset(q)

def applyLogicGate(qc, init_vals, gate_func):
    for i, j in init_vals:
        resetQc(qc)
        # Initialize qubits 0 and 1
        qc.ry(i, 0)
        qc.ry(j, 1)

        # Apply specific gate
        gate_func(qc)

        # Measure qubit 2 in bit 0
        qc.measure(2, 0)
        simPrintVals(qc, text=f"{1 if i == np.pi else 0} {1 if j == np.pi else 0} --> ")


def gatesAND(qc):
    # Use CCNOT to verify both states are |1>
    qc.ccx(0, 1, 2)

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

def andGate(qc, init_vals):
    applyLogicGate(qc, init_vals, gatesAND)

def orGate(qc, init_vals):
    applyLogicGate(qc, init_vals, gatesOR)

def xorGate(qc, init_vals):
    applyLogicGate(qc, init_vals, gatesXOR)

if __name__ == "__main__":
    # Use preety print for arrays
    np.set_printoptions(formatter={'complex_kind': lambda x: f"{x:10.4g}"})
    # exQCAndSim()

    init_vals = [[0, 0], [0, np.pi], [np.pi, 0], [np.pi, np.pi]]
    qc = QuantumCircuit(3, 1)

    print("AND gate simulated:")
    andGate(qc, init_vals)

    print(f"\nXOR gate simulated:")
    xorGate(qc, init_vals)

    print(f"\nOR gate simulated:")
    orGate(qc, init_vals)