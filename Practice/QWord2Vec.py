import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.visualization.bloch import Bloch
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
import qiskit_algorithms
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals

# Set random seed directly
qiskit_algorithms.utils.algorithm_globals.random_seed = 42


def get_entangling_layer(num_qubits):
    """
    Creates the entangling layer U_enta.
    Based on Sim et al. (2019), using circular CNOT configuration.
    (explanied at the end of page 3 article)
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
    Remember that is in reverse from that was read???
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


def get_embeddings_batched(params, qc_ansatz, num_qubits):
    """
    Efficiently computes Bloch vectors for ALL inputs k=0..N-1
    using a single batch execution on AerSimulator.
    """
    num_words = 2**num_qubits
    sim = AerSimulator()

    # 1. Bind parameters ONCE to the ansatz
    qc_bound = qc_ansatz.assign_parameters(params)

    # 2. Create batch of circuits (one for each input word k)
    circuits = []
    for k in range(num_words):
        qc = QuantumCircuit(num_qubits)
        encode_input(qc, k, num_qubits)
        qc.compose(qc_bound, inplace=True)
        qc.save_statevector()
        circuits.append(qc)

    # 3. Transpile and Run as a SINGLE Job (Batch Processing)
    # This is much faster than running 32 separate jobs.
    t_circuits = transpile(circuits, sim)
    job = sim.run(t_circuits, shots=1)
    result = job.result()

    embeddings = []
    traced_qubits = list(range(1, num_qubits))

    # Pre-define Pauli matrices for expectation values
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])

    # 4. Process Results
    for i in range(num_words):
        # Retrieve statevector for the i-th circuit in the batch
        state = result.get_statevector(i)

        # Partial trace to extract state of qubit 0
        rho_0 = partial_trace(state, traced_qubits)

        # Calculate Bloch coordinates
        x = rho_0.expectation_value(pauli_x).real
        y = rho_0.expectation_value(pauli_y).real
        z = rho_0.expectation_value(pauli_z).real

        embeddings.append([x, y, z])

    return np.array(embeddings)


def calculate_cross_entropy(embeddings, num_words):
    """
    Calculates the Cross Entropy Loss (Skip-Gram style approximation)
    ideally used in Word2Vec.

    Since we lack a real corpus/pairs (Context, Target), we simulate the math:
    - Positive pairs: We assume for demo that word i and (i+1) are related.
    - Negative pairs: Randomly sampled.

    Formula: -log(sigmoid(u . v)) for positive pairs
             -log(sigmoid(-u . v_neg)) for negative pairs
    """
    loss = 0.0
    # Create Dummy Batch of Positive Pairs (e.g., word i is context for i+1)
    # In real training, this comes from the text corpus window.
    positive_pairs = [(i, (i + 1) % num_words) for i in range(num_words)]

    # Simple sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Compute Loss for Positive Pairs
    for i, j in positive_pairs:
        u = embeddings[i]
        v = embeddings[j]
        # Dot product of embeddings (measures similarity in Hilbert space)
        score = np.dot(u, v)
        loss += -np.log(sigmoid(score) + 1e-9)  # 1e-9 for stability

    return loss / len(positive_pairs)


def calculate_custom_loss(params, qc_ansatz, n_qubits, target_distances, C=1.0):
    """
    Implements the custom loss function from the paper:
    Loss = CrossEntropy - C * CorrelationCoefficient(EmbeddingData, TrainingLabels)
    """
    # 1. Get Quantum Embeddings for all words (Batched & Efficient)
    q_embeddings = get_embeddings_batched(params, qc_ansatz, n_qubits)

    # 2. Calculate Distance Vector for Quantum Embeddings (Euclidean)
    q_dists = pdist(q_embeddings, metric="euclidean")

    # 3. Calculate Correlation with Target Distances
    if len(target_distances) != len(q_dists):
        correlation = 0.0
    else:
        correlation, _ = pearsonr(q_dists, target_distances)

    # 4. Cross Entropy Term (Implemented Math)
    # Calculates the semantic loss based on the current embeddings
    num_words = 2**n_qubits
    cross_entropy = calculate_cross_entropy(q_embeddings, num_words)

    # 5. Total Loss
    total_loss = cross_entropy - (C * correlation)

    return total_loss


def train_qword2vec_spsa(qc_ansatz, initial_params, n_qubits):
    """
    Setup for SPSA optimization based on Parameter Shift Rule.
    Uses a stochastic gradient approach to minimize loss.
    """
    # Prepare Dummy Target Distances for validation of the formula
    num_pairs = (2**n_qubits * (2**n_qubits - 1)) // 2
    dummy_targets = np.random.rand(num_pairs)

    # Wrapper is a closure that captures 'qc_ansatz', 'n_qubits', 'dummy_targets'
    # and let the optimizer to modify the 'params'
    def loss_function(params):
        return calculate_custom_loss(params, qc_ansatz, n_qubits, dummy_targets, C=1.0)

    # Initialize SPSA
    optimizer = SPSA(maxiter=100, learning_rate=0.001)

    print("\nStarting SPSA Optimization...")
    result = optimizer.minimize(fun=loss_function, x0=initial_params)
    print(f"SPSA Optimization Complete. Final Loss: {result.fun:.4f}")
    return result.x


if __name__ == "__main__":
    # Q-word2vec Configuration for Fig 2b
    n_qubits = 5  # System size (32 words)
    # Increase layers to ensure information propagates from all input bits to the embedding qubit
    # With L=1, qubit 0 only sees neighbors, causing clustering around ~2-4 states.
    n_layers = 5  # Depth L=5

    print(f"--- Q-Word2Vec Full Model (n={n_qubits}, L={n_layers}) ---")

    qc_ansatz, theta_params = build_qword2vec_circuit(n_qubits, n_layers)

    # Random parameters
    np.random.seed(42)
    theta_vals = np.random.uniform(0, 2 * np.pi, len(theta_params))

    # Demo of SPSA setup
    # theta_vals = train_qword2vec_spsa(qc_ansatz, theta_vals, n_qubits)

    print("Computing Bloch vectors for all 32 input words...")

    # Efficient Batch Calculation for Visualization
    num_words = 2**n_qubits
    embeddings = get_embeddings_batched(theta_vals, qc_ansatz, n_qubits)

    # Unpack for plotting
    xs = embeddings[:, 0]
    ys = embeddings[:, 1]
    zs = embeddings[:, 2]

    print(f"Vectors computed: {len(xs)}")
    print("Plotting scattered points on Bloch Sphere...")

    # Visualization
    try:
        b = Bloch()

        # Generate distinct colors for 32 points using a colormap
        cmap = plt.get_cmap("hsv")  # 'hsv' gives a nice rainbow cycle
        colors = [cmap(i / num_words) for i in range(num_words)]

        # Clear default point colors/markers to set our own
        b.point_color = []
        b.point_marker = []

        for i in range(num_words):
            # Add point i as a list of lists [[x], [y], [z]]
            b.add_points([[xs[i]], [ys[i]], [zs[i]]])
            # Set style for this point
            b.point_color.append(colors[i])
            b.point_marker.append("o")

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
