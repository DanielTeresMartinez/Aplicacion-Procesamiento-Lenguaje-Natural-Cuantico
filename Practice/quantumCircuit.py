from qiskit.circuit.library import XGate, YGate, ZGate, HGate
import numpy as np

if __name__ == "__main__":
    # Use preety print for arrays
    np.set_printoptions(formatter={'complex_kind': lambda x: f"{x:10.4g}"})

    x_gate = XGate()
    print(f"X.to_matrix():\n", x_gate.to_matrix())

    yh_gate = HGate()
    print(f"\nH.to_matrix():\n", yh_gate.to_matrix())