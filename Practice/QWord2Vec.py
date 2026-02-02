import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, partial_trace

# Try importing Bloch directly to support point plotting
try:
    from qiskit.visualization.bloch import Bloch
except ImportError:
    try:
        from qiskit.visualization import Bloch
    except ImportError:
        # If standard imports fail, we might need qutip directly or verify environment
        print("Warning: Could not import Bloch from qiskit.visualization.bloch")
        # In some environments, it might be available differently or require 'pip install qutip'
        pass


def get_entangling_layer(num_qubits):
    """
    Creates the entangling layer U_enta.
    Based on Sim et al. (2019), using circular CNOT configuration.
    """
    qc = QuantumCircuit(num_qubits)
    if num_qubits > 1:
        for i in range(num_qubits):
            qc.cx(i, (i + 1) % num_qubits)
    return qc


def build_qword2vec_circuit(num_qubits, num_layers):
    """
    Implements the parameterized circuit U(θ) for Q-word2vec.
    Structure: Prod ( U_enta * RZ(θz) * RY(θy) * RX(θx) )
    Order in circuit (left to right): RX -> RY -> RZ -> U_enta
    """
    # 3 params (Rx, Ry, Rz) per qubit per layer
    num_params_per_layer = 3 * num_qubits
    total_params = num_params_per_layer * num_layers
    θ = ParameterVector("θ", total_params)

    qc = QuantumCircuit(num_qubits)
    param_idx = 0

    for l in range(num_layers):
        # Layer 1: RX
        for q in range(num_qubits):
            qc.rx(θ[param_idx], q)
            param_idx += 1

        # Layer 2: RY
        for q in range(num_qubits):
            qc.ry(θ[param_idx], q)
            param_idx += 1

        # Layer 3: RZ
        for q in range(num_qubits):
            qc.rz(θ[param_idx], q)
            param_idx += 1

        # Layer 4: Entanglement
        enta_layer = get_entangling_layer(num_qubits)
        qc.compose(enta_layer, inplace=True)

        qc.barrier()

    return qc, θ


def encode_input(qc, k, num_qubits):
    """
    Encodes integer k into state |k> (computational basis).
    Uses standard binary mapping.
    """
    fmt = f"{{0:0{num_qubits}b}}"
    binary_k = fmt.format(k)
    for i, bit in enumerate(reversed(binary_k)):
        if bit == "1":
            qc.x(i)


def get_bloch_vector_for_k(k, theta_values, num_qubits, qc_ansatz):
    """
    Runs the circuit for input k, traces out qubits 1..n-1,
    and returns the [x, y, z] Bloch vector of qubit 0 (embedding qubit).
    """
    sim = AerSimulator()

    qc_full = QuantumCircuit(num_qubits)
    encode_input(qc_full, k, num_qubits)

    qc_bound = qc_ansatz.assign_parameters(theta_values)
    qc_full.compose(qc_bound, inplace=True)

    qc_full.save_statevector()

    t_qc = transpile(qc_full, sim)
    job = sim.run(t_qc, shots=1)
    state = job.result().get_statevector()

    # Partial trace to extract state of qubit 0
    traced_qubits = list(range(1, num_qubits))
    rho_0 = partial_trace(state, traced_qubits)

    # Get Bloch vector (Expectation values of Pauli matrices)
    x = rho_0.expectation_value(np.array([[0, 1], [1, 0]]))
    y = rho_0.expectation_value(np.array([[0, -1j], [1j, 0]]))
    z = rho_0.expectation_value(np.array([[1, 0], [0, -1]]))

    return [x.real, y.real, z.real]


if __name__ == "__main__":
    # Q-word2vec Configuration for Fig 2b
    n_qubits = 5  # System size (32 words)
    n_layers = 1  # Depth

    print(f"--- Q-Word2Vec Full Model (n={n_qubits}, L={n_layers}) ---")

    qc_ansatz, theta_params = build_qword2vec_circuit(n_qubits, n_layers)

    # Random parameters (representing a trained state)
    np.random.seed(42)
    theta_vals = np.random.uniform(0, 2 * np.pi, len(theta_params))

    print("Computing Bloch vectors for all 32 input words...")

    num_words = 2**n_qubits
    xs, ys, zs = [], [], []

    for k in range(num_words):
        vec = get_bloch_vector_for_k(k, theta_vals, n_qubits, qc_ansatz)
        xs.append(vec[0])
        ys.append(vec[1])
        zs.append(vec[2])

    print(f"Vectors computed: {len(xs)}")
    print("Plotting scattered points on Bloch Sphere...")

    # Visualization
    try:
        b = Bloch()
        # Add all 32 points
        # Bloch.add_points expects a list of lists: [[x...],[y...],[z...]] OR list of coords
        # passing [xs, ys, zs] works for points
        b.add_points([xs, ys, zs])

        # Optional: Formatting to look like Figure 2b
        b.vector_color = ["b"]
        b.point_color = ["r"]
        b.point_marker = ["o"]

        # Qiskit/Matplotlib Fix: Use render() and plt.show()
        if hasattr(b, "render"):
            b.render()

        plt.show()

    except NameError:
        print(
            "Error: Bloch class not available. Please install qutip or check qiskit visualization."
        )
    except TypeError as e:
        print(f"Visualization warning: {e}")
        plt.show()
