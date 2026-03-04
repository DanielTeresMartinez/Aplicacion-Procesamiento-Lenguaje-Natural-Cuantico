import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, partial_trace, Pauli
from qiskit.visualization.bloch import Bloch
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
import qiskit_algorithms
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals


# Set random seed directly
qiskit_algorithms.utils.algorithm_globals.random_seed = 42
np.random.seed(42)


def binary_entropy(p):
    """Calculates the binary entropy H(p)."""
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def get_error_probability(n, ne):
    """
    Estimates error probability p(n, ne) such that ne >= (1 - H(p)) * 2^n.
    We assume the equality to find the bound: H(p) = 1 - ne / (2^n).
    """
    target_h = 1 - ne / (2**n)

    # Binary search for p in [0, 0.5]
    low, high = 0.0, 0.5
    for _ in range(50):
        mid = (low + high) / 2
        if binary_entropy(mid) < target_h:
            low = mid
        else:
            high = mid

    p_val = (low + high) / 2
    print(f"p(n={n}, ne={ne}) = {p_val:.4f}")
    return p_val


def estimate_num_layers(n_qubits, n_embedding_qubits, num_data, ath):
    """
    Heuristic formula for estimating L (number of layers).
    L = ceil( #data / (3 * (n + ne) * Ath * p(n, ne)) )
    """
    p = get_error_probability(n_qubits, n_embedding_qubits)

    denominator = 3 * (n_qubits + n_embedding_qubits) * ath
    L = int(np.ceil(num_data / denominator) * p)
    return L


def get_entangling_layer(num_qubits):
    """

    General pattern for n qubits:
      CX(0, n-1)                    <- jump from top to bottom
      CX(n-1, n-2), CX(n-2, n-3), ..., CX(2, 1)   <- staircase up
    """
    qc = QuantumCircuit(num_qubits)
    if num_qubits > 1:
        # Step 1: top qubit (0) controls bottom qubit (n-1)
        qc.cx(num_qubits - 1, 0)
        # Step 2: staircase upward from bottom
        for i in range(num_qubits - 1, 0, -1):
            qc.cx(i - 1, i)

    return qc


def build_qword2vec_circuit(num_qubits, num_layers):
    """
    Implements the parameterized circuit U(θ) for Q-word2vec.

    The circuit operates on the n system qubits (num_qubits).
    Per the formula:
        U(θ) = Π_{l=1}^{L} [ U_enta · RZ(θz)^⊗n · RY(θy)^⊗n · RX(θx)^⊗n ]

    Reading the product left-to-right (as it appears in the circuit):
        RX -> RY -> RZ -> U_enta   (per layer)

    This matches the Circuit-Block (CB) configuration from the paper figure:
    each block shows RX rotations followed by CNOT entanglement.
    """
    # 3 params (Rx, Ry, Rz) per qubit per layer
    num_params_per_layer = 3 * num_qubits
    total_params = num_params_per_layer * num_layers
    θ = ParameterVector("θ", total_params)

    qc = QuantumCircuit(num_qubits)
    param_idx = 0

    for l in range(num_layers):
        # RX on all qubits
        for q in range(num_qubits):
            qc.rx(θ[param_idx], q)
            param_idx += 1

        # RY on all qubits
        for q in range(num_qubits):
            qc.ry(θ[param_idx], q)
            param_idx += 1

        # RZ on all qubits
        for q in range(num_qubits):
            qc.rz(θ[param_idx], q)
            param_idx += 1

        # U_enta: CNOT block on system qubits
        enta_layer = get_entangling_layer(num_qubits)
        qc.compose(enta_layer, inplace=True)

        # Visual separator between layers
        if l < num_layers - 1:
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


# Para preguntar: Me lía mucho esta parte de generar 32 circuitos para las 32 posibles palabras.
def prepare_transpiled_circuits(qc, n_qubits, sim):
    """
    Pre-compiles the circuits for all possible input words.
    """
    circuits = []
    num_words = 2**n_qubits
    for k in range(num_words):
        temp_qc = QuantumCircuit(n_qubits)
        encode_input(temp_qc, k, n_qubits)
        temp_qc.compose(qc, inplace=True)
        temp_qc.save_statevector()
        circuits.append(temp_qc)
    return transpile(circuits, sim)


def get_embeddings_batched(params, theta_params, t_circuits, sim, num_qubits):
    """
    Efficiently computes Bloch vectors using pre-transpiled circuits and parameter binding.
    """
    num_words = len(t_circuits)

    # Bind parameters efficiently to the transpiled circuits
    # Since we want to apply the SAME global parameters to all circuits:
    # Converted to python floats to avoid AerSimulator errors
    clean_params = [float(p) for p in params]
    bind_dict = dict(zip(theta_params, clean_params))

    # Create bound circuits (assign_parameters returns new copies by default)
    bound_circuits = [c.assign_parameters(bind_dict) for c in t_circuits]

    job = sim.run(bound_circuits, shots=1)
    result = job.result()

    embeddings = []
    traced_qubits = list(range(1, num_qubits))

    pauli_x = Pauli("X")
    pauli_y = Pauli("Y")
    pauli_z = Pauli("Z")

    for i in range(num_words):
        state = result.get_statevector(i)
        rho_0 = partial_trace(state, traced_qubits)
        x = rho_0.expectation_value(pauli_x).real
        y = rho_0.expectation_value(pauli_y).real
        z = rho_0.expectation_value(pauli_z).real

        embeddings.append([x, y, z])

    return np.array(embeddings)


def calculate_cross_entropy(embeddings, num_words, training_data):
    """
    Calculates the Cross Entropy Loss (Skip-Gram style approximation)
    using the provided synthetic training data.

    training_data: dict { target_word_idx: [context_idx1, context_idx2] }

    Formula: -log(sigmoid(u . v)) for positive pairs
    """
    loss = 0.0

    # Create List of Positive Pairs from Training Data
    positive_pairs = []
    for target_word, context_words in training_data.items():
        for context_word in context_words:
            positive_pairs.append((target_word, context_word))

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Compute Loss for Positive Pairs
    for i, j in positive_pairs:
        u = embeddings[i]
        v = embeddings[j]

        # Dot product of embeddings (measures similarity in Hilbert space)
        score = np.dot(u, v)
        loss += -np.log(sigmoid(score))

    if len(positive_pairs) == 0:
        return 0.0

    return loss / len(positive_pairs)


def calculate_custom_loss(
    params, theta_params, t_circuits, sim, n_qubits, target_distances, training_data, C
):
    """
    Implements the custom loss function from the paper:
    Loss = CrossEntropy - C * CorrelationCoefficient(EmbeddingData, TrainingLabels)
    It's implemented in that way to retain the relationships that may not be preserved
    after the measurement. It forces the circuit to "maintain" the relationships and distances.
    """
    q_embeddings = get_embeddings_batched(
        params, theta_params, t_circuits, sim, n_qubits
    )

    # Calculate Distance Vector for Quantum Embeddings (Euclidean)
    q_dists = pdist(q_embeddings, metric="euclidean")

    # Calculate Correlation with Target Distances
    if len(target_distances) != len(q_dists):
        correlation = 0.0
    else:
        correlation, _ = pearsonr(q_dists, target_distances)

    # Calculates the semantic loss based on the current embeddings
    num_words = 2**n_qubits
    cross_entropy = calculate_cross_entropy(q_embeddings, num_words, training_data)

    # Total Loss
    return cross_entropy - (C * correlation)


def train_qword2vec(
    qc, theta_params, initial_params, n_qubits, training_data, epochs=100, c_val=1.0
):
    """
    Setup for SPSA optimization based on Parameter Shift Rule.
    Uses a stochastic gradient approach to minimize loss.
    """
    # Initialize Simulator and Pre-Transpile Circuits ONCE
    sim = AerSimulator()
    t_circuits = prepare_transpiled_circuits(qc, n_qubits, sim)

    # Prepare Dummy Target Distances for validation of the formula
    num_pairs = (2**n_qubits * (2**n_qubits - 1)) // 2
    dummy_targets = np.random.rand(num_pairs)

    def loss_function(params):
        return calculate_custom_loss(
            params,
            theta_params,
            t_circuits,
            sim,
            n_qubits,
            dummy_targets,
            training_data,
            c_val,
        )

    # Initialize SPSA
    optimizer = SPSA(maxiter=epochs, learning_rate=0.001, perturbation=0.1)

    print("\nStarting SPSA Optimization...")
    result = optimizer.minimize(fun=loss_function, x0=initial_params)
    print(f"SPSA Optimization Complete. Final Loss: {result.fun:.4f}")

    # Calculate and print final Error Rate
    final_embeddings = get_embeddings_batched(
        result.x, theta_params, t_circuits, sim, n_qubits
    )
    final_error_rate = calculate_error_rate(final_embeddings, training_data)
    print(f"Final Error Rate on Training Data: {final_error_rate:.4f}")

    return result.x


def load_corpus(file_path):
    """
    Reads the corpus from a text file.
    Returns a list of sentences (lists of words).
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Simple tokenization: lowercase and split
    corpus = [line.strip().lower().split() for line in lines]
    return corpus


def build_vocabulary(corpus, n_qubits):
    """
    Builds a vocabulary from the corpus.
    Maps words to integers 0 to 2^n - 1.
    """
    unique_words = set()
    for sentence in corpus:
        unique_words.update(sentence)

    sorted_words = sorted(list(unique_words))
    vocab_size = len(sorted_words)
    max_vocab = 2**n_qubits

    if vocab_size > max_vocab:
        print(f"Warning: Corpus has {vocab_size} words, truncating to {max_vocab}.")
        sorted_words = sorted_words[:max_vocab]

    word_to_id = {w: i for i, w in enumerate(sorted_words)}
    id_to_word = {i: w for i, w in enumerate(sorted_words)}

    return word_to_id, id_to_word


def generate_training_data_from_text(corpus, word_to_id, window_size=1):
    """
    Generates training data pairs using Skip-Gram approach.
    Returns: { target_id: [context_id1, context_id2, ...] }
    """
    training_data = {}

    # Initialize entries for all vocab words
    for idx in word_to_id.values():
        training_data[idx] = []

    for sentence in corpus:
        indices = [word_to_id[w] for w in sentence if w in word_to_id]

        for i, target_idx in enumerate(indices):
            # Window range
            start = max(0, i - window_size)
            end = min(len(indices), i + window_size + 1)

            for j in range(start, end):
                if i != j:
                    context_idx = indices[j]
                    training_data[target_idx].append(context_idx)

    # Remove duplicates
    final_data = {}
    for t, contexts in training_data.items():
        if contexts:
            final_data[t] = contexts

    return final_data


def calculate_error_rate(embeddings, training_data):
    """
    Calculates the 'error rate' metric:
    Sum of mismatches between the two surrounding words of the model output
    and the training data labels, divided by (2 * total number of training samples).

    Model output for word w is the vector of dot products [w . v_0, w . v_1, ... w . v_N].
    """
    mismatches = 0
    num_samples = len(training_data)

    for word_idx, targets in training_data.items():
        u = embeddings[word_idx]

        # Scores with all other embeddings
        scores = np.dot(embeddings, u)
        scores[word_idx] = -np.inf

        # Find the indices of the top 2 scores
        top_2_indices = np.argsort(scores)[-2:]

        # Mismatch if a predicted top-2 index is NOT in the target list
        current_mismatches = 0
        for pred_idx in top_2_indices:
            if pred_idx not in targets:
                current_mismatches += 1

        mismatches += current_mismatches

    return mismatches / (2 * num_samples)


if __name__ == "__main__":
    # System size
    n_qubits = 4
    # Embedding size
    n_embedding = 2

    print(f"--- Q-Word2Vec Full Model (n={n_qubits}, ne={n_embedding}) ---")

    try:
        corpus = load_corpus("smallCorpora.txt")
        print("Corpus loaded successfully.")
    except FileNotFoundError:
        print("Error: smallCorpora.txt not found. Please create it.")
        exit(1)

    word_to_id, id_to_word = build_vocabulary(corpus, n_qubits)
    print(f"Vocabulary Size: {len(word_to_id)}/{2**n_qubits}")

    # Generate Training Data from Text
    training_data = generate_training_data_from_text(corpus, word_to_id, window_size=1)

    # Heuristic Estimation of L using actual number pairs of training data
    num_data = len(corpus)
    ath = 0.02

    n_layers = estimate_num_layers(n_qubits, n_embedding, num_data, ath)
    print(f"Heuristic L = {n_layers} (using #pairs={num_data}, Ath={ath})")

    qc, theta_params = build_qword2vec_circuit(n_qubits, n_layers)
    theta_vals = np.random.uniform(0, 2 * np.pi, len(theta_params))

    # --- Draw the circuit to verify the entangling block structure ---
    print("\n--- Circuit Diagram (U_enta verification) ---")
    # Build a small example for visualization (max 2 layers for readability)
    qc_draw, _ = build_qword2vec_circuit(n_qubits, min(n_layers, 2))
    fig = qc_draw.draw(output="mpl", fold=-1, style="iqp")
    fig.suptitle(
        f"QWord2Vec Circuit Block (n={n_qubits}, ne={n_embedding}, L={min(n_layers, 2)})",
        fontsize=12,
    )
    fig.savefig("qword2vec_circuit.png", dpi=150, bbox_inches="tight")
    print("Circuit diagram saved to qword2vec_circuit.png")
    plt.show()

    # Demo of SPSA setup
    theta_vals = train_qword2vec(
        qc, theta_params, theta_vals, n_qubits, training_data, epochs=1000, c_val=0
    )

    print(f"Computing Bloch vectors for all {2**n_qubits} input words...")

    # Efficient Batch Calculation for Visualization
    num_words = 2**n_qubits
    sim = AerSimulator()
    t_circuits = prepare_transpiled_circuits(qc, n_qubits, sim)
    embeddings = get_embeddings_batched(
        theta_vals, theta_params, t_circuits, sim, n_qubits
    )

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
