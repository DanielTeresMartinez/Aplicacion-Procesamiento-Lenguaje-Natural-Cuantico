from qiskit.quantum_info import Statevector
import numpy as np

if __name__ == "__main__":
    # Dims is 2 due to states |0> and |1>
    sv0 = Statevector([1, 0], dims=2)
    # Instead of 4dim, it's parsed as (2, 2)
    sv4 = Statevector([0, 1, 1, 1], dims=4)
    # 2 dimensions and which position will have basis state
    fi = Statevector.from_int(0, 2)
    print(f"Test state vector and from_int:\n", sv0, f"\n", fi)

    # Tensor product of Pauli X, Y, Z, eigenstates.
    fl1 = Statevector.from_label("1")
    fl_plus = Statevector.from_label("-")

    # Set seed value by default
    sv_measures = Statevector.from_label("+")
    sv_measures.seed(42)
    outcome, state = sv_measures.measure()
    print(f"value = {outcome} from state: {state}")

