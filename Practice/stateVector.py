from qiskit.quantum_info import Statevector
from qiskit.quantum_info import Operator
import numpy as np

if __name__ == "__main__":
    # Use preety print for arrays
    np.set_printoptions(formatter={'complex_kind': lambda x: f"{x:10.4g}"})

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
    print(f"\nMueasurment method:\nvalue = {outcome} from state: {state}\n")

    res_prob = sv_measures.probabilities(decimals=3)
    res_prob_dict = sv_measures.probabilities_dict(decimals=3)
    print(f"Probabilities: {res_prob}\nProbabilities dict: {res_prob_dict}\n")

    sv_samples = Statevector.from_label("r")
    sv_samples.seed(42)
    # Use probabilities() for current state and qargs. Do not modify the current state.
    # Predict which values will be computed when measured.
    res_samples = sv_samples.sample_counts(200) 
    # Same as before but returns values in a list instead of dict
    res_samples_list = sv_samples.sample_memory(5)
    print(f"Sample_counts: {res_samples}\nResults list format: {res_samples_list}\n")

    sv_samples.draw("bloch")
    # Show previous function when executed on terminal.
    # plt.show()

    # Evolve a qubit with a matrix operator. In this case use Pauli X gate
    res_evolve = sv_samples.evolve(Operator.from_label("X"))
    print(f"Qubit previous evolve: {sv_samples.data}\nApplying Pauli x Gate...\n"
          f"Qubit after evolve: {res_evolve.data}\n")


