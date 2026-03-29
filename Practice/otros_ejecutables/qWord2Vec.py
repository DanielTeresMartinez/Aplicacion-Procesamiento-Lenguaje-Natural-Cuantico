"""
qWord2Vec.py — Implementación del modelo Q-Word2Vec.

Basado en: Ohno et al., "Q-word2vec: Quantum Natural Language Processing",
Quantum Machine Intelligence, 2025.

Las funciones siguen el orden de las secciones del paper:
    Datos  →  §3.4 Profundidad  →  §3.1 Circuito  →  §3.3 Pérdida
    →  §4 Error rate  →  §3.1 Entrenamiento  →  §3.2 Evaluación
"""

import os
from collections import Counter
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit.quantum_info import partial_trace, Pauli
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
import qiskit_algorithms
from qiskit_algorithms.utils import algorithm_globals
from qvisualization import plot_circuit, plot_embeddings_2d, plot_bloch_sphere


# Semilla aleatoria para reproducibilidad
qiskit_algorithms.utils.algorithm_globals.random_seed = 42
np.random.seed(42)


# =============================================================================
# DATOS Y CORPUS
# Carga y preparación de los datos de entrenamiento.
# =============================================================================


def load_corpus(file_path):
    """Lee el corpus de frases. Cada línea es una frase independiente."""
    with open(file_path, "r") as f:
        lines = f.readlines()
    return [line.strip().lower().split() for line in lines]


def load_word_list(file_path):
    """
    Lee una lista de palabras (una por línea) y la trata como una única secuencia
    sin fronteras de frase, de modo que la ventana de contexto cruza líneas libremente.
    """
    with open(file_path, "r") as f:
        words = [line.strip().lower() for line in f if line.strip()]
    print(f"Word list cargado: {len(words)} palabras desde '{file_path}'")
    return [words]


def load_word2vec_embeddings(filepath):
    """
    Carga los embeddings de Word2Vec generados por word2Vec.py.
    Lanza FileNotFoundError si el fichero no existe.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Fichero de embeddings no encontrado: '{filepath}'\n"
            "Ejecuta word2Vec.py primero para generarlo."
        )

    embeddings = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "  #" in line:
                line = line[: line.index("  #")]
            parts = line.split()
            embeddings[parts[0]] = np.array([float(v) for v in parts[1:]], dtype=float)

    dim = next(iter(embeddings.values())).shape[0] if embeddings else 0
    print(
        f"Cargados {len(embeddings)} embeddings Word2Vec (dim={dim}) desde '{filepath}'"
    )
    return embeddings


def build_vocabulary(corpus, n_qubits):
    """
    Construye el vocabulario asignando un índice 0..2^n-1 a cada palabra única.
    Si hay más palabras que 2^n_qubits se trunca alfabéticamente.
    """
    unique_words = sorted(set(w for sentence in corpus for w in sentence))
    max_vocab = 2**n_qubits

    if len(unique_words) > max_vocab:
        print(f"Aviso: {len(unique_words)} palabras, truncando a {max_vocab}.")
        unique_words = unique_words[:max_vocab]

    word_to_id = {w: i for i, w in enumerate(unique_words)}
    id_to_word = {i: w for i, w in enumerate(unique_words)}
    return word_to_id, id_to_word


def generate_training_data_from_text(corpus, word_to_id, window_size=1):
    """
    Genera pares Skip-Gram con ventana deslizante.
    Devuelve { target_id: [context_id, ...] } para palabras con al menos un contexto.
    """
    training_data = {idx: [] for idx in word_to_id.values()}

    for sentence in corpus:
        indices = [word_to_id[w] for w in sentence if w in word_to_id]
        for i, target_idx in enumerate(indices):
            start = max(0, i - window_size)
            end = min(len(indices), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    training_data[target_idx].append(indices[j])

    return {t: ctx for t, ctx in training_data.items() if ctx}


# =============================================================================
# SECCIÓN 4 · Experimentos — Vectores de etiquetas
# "Two elements [...] were set to 0.5, while all other elements were set to zero."
# =============================================================================


def generate_label_vectors(training_data, n_qubits):
    """
    Convierte training_data en vectores de etiqueta de tamaño N=2^n_qubits.

    Las dos palabras de contexto más frecuentes de cada palabra objetivo reciben
    el valor 0.5 y el resto 0, de modo que el vector suma 1 (Sección 4).
    """
    N = 2**n_qubits
    label_vectors = {}
    for k, contexts in training_data.items():
        label = np.zeros(N)
        for ctx, _ in Counter(contexts).most_common(2):
            label[ctx] = 0.5
        label_vectors[k] = label
    return label_vectors


# =============================================================================
# SECCIÓN 3.4 · Estimación de la profundidad del circuito
# Fórmulas (ec. 6–7): L = ⌈num_data / (3·(n+ne)·Ath)⌉ · p(n,ne)
# donde p se obtiene invirtiendo H(p) = 1 − ne/2^n.
# =============================================================================


def binary_entropy(p):
    """Entropía binaria H(p) = -p·log₂(p) - (1-p)·log₂(1-p)."""
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def get_error_probability(n, ne):
    """
    Resuelve numéricamente p tal que H(p) = 1 - ne/2^n (ec. 7).
    Usa búsqueda binaria en [0, 0.5] porque H(p) no tiene inversa cerrada.
    """
    target_h = 1 - ne / (2**n)
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
    """Aplica la fórmula heurística (ec. 6) para estimar la profundidad L del circuito."""
    p = get_error_probability(n_qubits, n_embedding_qubits)
    denominator = 3 * (n_qubits + n_embedding_qubits) * ath
    return int(np.ceil(num_data / denominator) * p)


# =============================================================================
# SECCIÓN 3.1 · Q-word2vec — Construcción del circuito
# Arquitectura (ec. 4, Fig. 1):  |k⟩ → U(θ_u) → reset(ne..n-1) → V(θ_v) → |ψ⟩
# Cada bloque aplica L capas de  RX^⊗n → RY^⊗n → RZ^⊗n → U_enta.
# =============================================================================


def get_entangling_layer(num_qubits):
    """
    Capa de entrelazamiento U_enta con puertas CX en configuración circuit-block
    (Sim et al. 2019), tal como especifica el paper.
    """
    qc = QuantumCircuit(num_qubits)
    if num_qubits > 1:
        qc.cx(num_qubits - 1, 0)
        for i in range(num_qubits - 1, 0, -1):
            qc.cx(i - 1, i)
    return qc


def build_parameterized_block(num_qubits, num_layers, param_name):
    """
    Construye un bloque U(θ) o V(θ) con L capas (ec. 4).
    Cada capa: RX^⊗n → RY^⊗n → RZ^⊗n → U_enta.
    Devuelve (QuantumCircuit, ParameterVector de longitud 3·n·L).
    """
    num_params = 3 * num_qubits * num_layers
    theta = ParameterVector(param_name, num_params)
    qc = QuantumCircuit(num_qubits)
    param_idx = 0

    for l in range(num_layers):
        for q in range(num_qubits):
            qc.rx(theta[param_idx], q)
            param_idx += 1
        for q in range(num_qubits):
            qc.ry(theta[param_idx], q)
            param_idx += 1
        for q in range(num_qubits):
            qc.rz(theta[param_idx], q)
            param_idx += 1
        qc.compose(get_entangling_layer(num_qubits), inplace=True)
        if l < num_layers - 1:
            qc.barrier()

    return qc, theta


def build_full_qword2vec_circuit(n_qubits, n_embedding, num_layers):
    """
    Ensambla el circuito completo QWord2Vec (Fig. 1):
        |k⟩ → U(θ_u) → reset(qubits ne..n-1) → V(θ_v) → |ψ⟩

    El reset tras U hace que sólo los ne qubits de embedding pasen información
    de U a V, actuando como cuello de botella (análogo al autoencoder de Word2Vec).

    Devuelve (QuantumCircuit, theta_u, theta_v).
    """
    u_circuit, theta_u = build_parameterized_block(n_qubits, num_layers, "θ_u")
    v_circuit, theta_v = build_parameterized_block(n_qubits, num_layers, "θ_v")

    qc = QuantumCircuit(n_qubits)
    qc.barrier(label="U(θ_u)")
    qc.compose(u_circuit, inplace=True)
    for q in range(n_embedding, n_qubits):
        qc.reset(q)
    qc.barrier(label="V(θ_v)")
    qc.compose(v_circuit, inplace=True)
    qc.barrier(label="|ψ⟩")

    return qc, theta_u, theta_v


def prepare_transpiled_circuits(qc, n_qubits, sim):
    """
    Pre-compila N=2^n circuitos, uno por cada palabra de entrada |k⟩.
    Cada palabra k se codifica aplicando puertas X en los qubits correspondientes
    a los bits a '1' de k. Pre-transpilar evita repetir este coste en cada
    evaluación de la función de pérdida.

    Ejemplo (n=3, k=5):
        k=5 en binario → |101⟩  (puertas X en qubits 0 y 2)

        El vector estado resultante en el espacio de 2^3=8 dimensiones es:
            |101⟩ = [0, 0, 0, 0, 0, 1, 0, 0]

        Es decir, un vector one-hot con el 1 en la posición k=5.
        Esto es el equivalente cuántico del vector one-hot de Word2Vec clásico.
    """
    circuits = []
    fmt = f"{{0:0{n_qubits}b}}"
    for k in range(2**n_qubits):
        temp_qc = QuantumCircuit(n_qubits)
        for i, bit in enumerate(reversed(fmt.format(k))):
            if bit == "1":
                temp_qc.x(i)
        temp_qc.compose(qc, inplace=True)
        temp_qc.save_statevector()
        circuits.append(temp_qc)
    return transpile(circuits, sim)


# =============================================================================
# SECCIÓN 3.3 · Función de pérdida
# (ec. 5):  Loss = CrossEntropy(p, y) − C · PearsonR(d_Q, d_W2V)
# =============================================================================


def get_output_probabilities_batched(params, theta_params, t_circuits, sim, n_qubits):
    """
    Ejecuta el circuito completo (U → reset → V) para cada palabra y devuelve
    la distribución de probabilidad de salida sobre los N estados base:
        prob_array[k][j] = |⟨j | V(θ_v) U(θ_u) | k⟩|²

    Es la salida que el paper usa para la cross-entropy (§3.3) y el error rate (§4).
    """
    N = 2**n_qubits
    bind_dict = dict(zip(theta_params, [float(p) for p in params]))
    bound_circuits = [c.assign_parameters(bind_dict) for c in t_circuits]

    result = sim.run(bound_circuits, shots=1).result()

    prob_array = np.zeros((N, N))
    for k in range(N):
        prob_array[k] = result.get_statevector(k).probabilities()
    return prob_array


def calculate_cross_entropy(prob_distributions, label_vectors):
    """
    Término de cross-entropy de la pérdida (§3.3):
        H(y, p) = −Σ_j  y_k[j] · log(p_k[j])   promediado sobre las muestras.
    """
    eps = 1e-10
    loss = 0.0
    count = 0
    for k, label in label_vectors.items():
        loss += -np.sum(label * np.log(prob_distributions[k] + eps))
        count += 1
    return loss / count if count > 0 else 0.0


def calculate_custom_loss(
    params, theta_params, t_circuits, sim, n_qubits, target_distances, label_vectors, C
):
    """
    Función de pérdida completa (ec. 5):
        Loss = CrossEntropy(p, y) − C · PearsonR(d_Q, d_W2V)

    La cross-entropy acerca la salida del circuito a los vectores de etiqueta.
    El término de correlación de Pearson regulariza la geometría para que las
    distancias entre distribuciones cuánticas correlacionen con las de Word2Vec.
    """
    prob_distributions = get_output_probabilities_batched(
        params, theta_params, t_circuits, sim, n_qubits
    )

    q_dists = pdist(prob_distributions, metric="euclidean")

    if len(target_distances) != len(q_dists):
        correlation = 0.0
    else:
        correlation, _ = pearsonr(q_dists, target_distances)

    cross_entropy = calculate_cross_entropy(prob_distributions, label_vectors)
    return cross_entropy - (C * correlation)


# =============================================================================
# SECCIÓN 4 · Experimentos — Métrica error rate
# "The error rate [...] is the sum of the number of mismatches between the two
# peak positions [...] divided by twice the total number of training samples."
# =============================================================================


def calculate_error_rate(prob_distributions, label_vectors):
    """
    Error rate (Sección 4):
        error_rate = Σ_k mismatches(pred_peaks_k, label_peaks_k) / (2 · |muestras|)

    pred_peaks  = top-2 estados base por probabilidad en la salida del circuito.
    label_peaks = top-2 posiciones del vector de etiqueta (las 2 palabras de contexto).
    mismatches  = picos predichos que no están en los picos del label (0, 1 ó 2).

    El paper exige error_rate == 0 en los datos de entrenamiento para la convergencia.
    """
    if not label_vectors:
        return 0.0

    mismatches = 0
    for k, label in label_vectors.items():
        pred_peaks = set(np.argsort(prob_distributions[k])[-2:])
        label_peaks = set(np.argsort(label)[-2:])
        mismatches += len(pred_peaks - label_peaks)

    return mismatches / (2 * len(label_vectors))


# =============================================================================
# SECCIÓN 3.1 · Q-word2vec — Entrenamiento
# Optimizador: SGD con momentum (lr=0.001, momentum=0.9) + SPSA para el gradiente.
# SPSA sólo necesita 2 evaluaciones del circuito por paso (Gacon et al. 2021).
# Parada anticipada cuando error_rate == 0 en los datos de entrenamiento.
# =============================================================================


def train_qword2vec(
    qc,
    theta_params,
    initial_params,
    n_qubits,
    label_vectors,
    target_distances,
    epochs=10000,
    c_val=2.0,
    learning_rate=0.001,
    momentum=0.9,
):
    """
    Entrena Q-Word2Vec optimizando θ_u y θ_v con SGD + SPSA.

    SPSA aproxima el gradiente con dos evaluaciones de la pérdida:
        ĝ = (L(θ + c·δ) − L(θ − c·δ)) / (2·c·δ),   δ ∈ {−1, +1}^d aleatorio.

    Devuelve (params_optimizados, historial_de_pérdida).
    """
    sim = AerSimulator()
    t_circuits = prepare_transpiled_circuits(qc, n_qubits, sim)

    def loss_fn(p):
        return calculate_custom_loss(
            p,
            theta_params,
            t_circuits,
            sim,
            n_qubits,
            target_distances,
            label_vectors,
            c_val,
        )

    params = np.array(initial_params, dtype=float)
    velocity = np.zeros_like(params)
    loss_history = []
    c = 0.1  # constante de perturbación SPSA

    print(f"\nIniciando SGD + SPSA  (lr={learning_rate}, momentum={momentum})")

    check_every = 10
    final_error_rate = None

    for epoch in range(epochs):
        delta = np.random.choice([-1.0, 1.0], size=len(params))

        loss_plus = loss_fn(params + c * delta)
        loss_minus = loss_fn(params - c * delta)
        grad = (loss_plus - loss_minus) / (2.0 * c * delta)

        velocity = momentum * velocity + learning_rate * grad
        params = params - velocity

        current_loss = (loss_plus + loss_minus) / 2.0
        loss_history.append(current_loss)

        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(f"  Época {epoch + 1:>4}/{epochs}  |  Pérdida ≈ {current_loss:.6f}")

        if (epoch + 1) % check_every == 0:
            probs = get_output_probabilities_batched(
                params, theta_params, t_circuits, sim, n_qubits
            )
            error_rate = calculate_error_rate(probs, label_vectors)
            if error_rate == 0.0:
                print(f"\n  Parada anticipada en época {epoch + 1}: error rate = 0.0")
                final_error_rate = 0.0
                break

    print(f"\nEntrenamiento completado.  Pérdida final ≈ {loss_history[-1]:.4f}")

    if final_error_rate is None:
        probs = get_output_probabilities_batched(
            params, theta_params, t_circuits, sim, n_qubits
        )
        final_error_rate = calculate_error_rate(probs, label_vectors)
    print(f"Error rate final: {final_error_rate:.4f}")

    return params, loss_history


# =============================================================================
# SECCIÓN 3.2 · Evaluación de los embeddings
# El paper usa la correlación de Pearson entre d_W2V y d_Q como métrica
# principal: una correlación alta indica que Q-Word2Vec ha preservado la
# geometría semántica de Word2Vec.
# =============================================================================


def get_embeddings_batched(
    params, theta_params, t_circuits, sim, num_qubits, n_embedding
):
    """
    Extrae los embeddings de Bloch tras el entrenamiento (post-training).

    Para cada palabra |k⟩ calcula el vector de Bloch (⟨X⟩, ⟨Y⟩, ⟨Z⟩) de cada
    qubit de embedding mediante traza parcial. El embedding de cada palabra tiene
    dimensión 3·ne y es el análogo cuántico de los vectores de Word2Vec.
    """
    num_words = len(t_circuits)
    bind_dict = dict(zip(theta_params, [float(p) for p in params]))
    bound_circuits = [c.assign_parameters(bind_dict) for c in t_circuits]

    result = sim.run(bound_circuits, shots=1).result()

    pauli_x, pauli_y, pauli_z = Pauli("X"), Pauli("Y"), Pauli("Z")
    embeddings = []
    for i in range(num_words):
        state = result.get_statevector(i)
        bloch_coords = []
        for q in range(n_embedding):
            trace_out = [j for j in range(num_qubits) if j != q]
            rho_q = partial_trace(state, trace_out)
            bloch_coords.extend(
                [
                    rho_q.expectation_value(pauli_x).real,
                    rho_q.expectation_value(pauli_y).real,
                    rho_q.expectation_value(pauli_z).real,
                ]
            )
        embeddings.append(bloch_coords)
    return np.array(embeddings)


def build_target_distances(w2v_embeddings, word_to_id, n_qubits):
    """
    Construye el vector de distancias d_W2V (§3.2, ec. 5).
    Contiene las distancias euclídeas entre todos los pares de embeddings Word2Vec,
    ordenados por índice de vocabulario. Las palabras ausentes se tratan como cero.
    """
    num_words = 2**n_qubits
    w2v_dim = next(iter(w2v_embeddings.values())).shape[0]
    w2v_ordered = np.zeros((num_words, w2v_dim))
    for word, idx in word_to_id.items():
        if word in w2v_embeddings:
            w2v_ordered[idx] = w2v_embeddings[word]
    return pdist(w2v_ordered, metric="euclidean")


def evaluate_embedding_quality(w2v_embeddings, qw2v_array, word_to_id):
    """
    Calcula la correlación de Pearson entre d_W2V y d_Q (§3.2).
    Una correlación cercana a 1 indica que Q-Word2Vec reproduce la geometría
    semántica de Word2Vec — criterio principal de éxito del paper.
    """
    common_words = sorted(set(w2v_embeddings) & set(word_to_id))
    if len(common_words) < 2:
        print("No hay suficientes palabras comunes para la evaluación.")
        return None, None

    w2v_vecs = np.array([w2v_embeddings[w] for w in common_words])
    qw2v_vecs = np.array([qw2v_array[word_to_id[w]] for w in common_words])

    corr, pval = pearsonr(
        pdist(w2v_vecs, metric="euclidean"),
        pdist(qw2v_vecs, metric="euclidean"),
    )

    print(f"\n--- Evaluación de calidad (§3.2) ---")
    print(f"Palabras comunes : {len(common_words)}")
    print(f"Pearson r        : {corr:.4f}  (p={pval:.4e})")
    return corr, pval


def save_qword2vec_embeddings(embeddings, id_to_word, filepath):
    """Guarda los embeddings de Bloch en un fichero de texto plano."""
    with open(filepath, "w") as f:
        f.write("# word  x  y  z\n")
        for idx, vec in enumerate(embeddings):
            if idx in id_to_word:
                f.write(
                    f"{id_to_word[idx]}  {vec[0]:.8f}  {vec[1]:.8f}  {vec[2]:.8f}\n"
                )
    print(f"Guardados {len(embeddings)} embeddings en '{filepath}'")


# =============================================================================
# PUNTO DE ENTRADA
# El flujo del main sigue exactamente el orden de las secciones del paper:
#   Datos  →  §3.4  →  §3.1  →  §3.3 + §4 (entrenamiento)  →  §3.2
# =============================================================================

if __name__ == "__main__":
    # False para omitir las visualizaciones y acelerar la ejecución
    SHOW_VISUALIZATIONS = True
    n_qubits = 4
    n_embedding = 2

    print(f"--- Q-Word2Vec (n={n_qubits}, ne={n_embedding}) ---")

    # ── Datos y corpus ───────────────────────────────────────────────────────
    try:
        corpus = load_corpus("smallCorpora.txt")
        print("Corpus de frases cargado.")
    except FileNotFoundError:
        print("Error: smallCorpora.txt no encontrado.")
        exit(1)

    if os.path.isfile("smallWordList.txt"):
        corpus = corpus + load_word_list("smallWordList.txt")
    else:
        print("[INFO] smallWordList.txt no encontrado — entrenando solo con frases.")

    w2v_embeddings = load_word2vec_embeddings("word2vec_embeddings.txt")
    word_to_id, id_to_word = build_vocabulary(corpus, n_qubits)
    print(f"Vocabulario: {len(word_to_id)}/{2**n_qubits} palabras")

    # ── §4 · Vectores de etiqueta ────────────────────────────────────────────
    # window_size=1 → los dos vecinos inmediatos, coincidiendo con los
    # "two surrounding words" de la Sección 4.
    training_data = generate_training_data_from_text(corpus, word_to_id, window_size=1)
    label_vectors = generate_label_vectors(training_data, n_qubits)
    print(f"Muestras de entrenamiento: {len(label_vectors)}")

    # ── §3.4 · Estimación de la profundidad L ────────────────────────────────
    num_data = len(word_to_id)
    # Mejor valor encontrado indicado por el paper
    ath = 0.02
    n_layers = estimate_num_layers(n_qubits, n_embedding, num_data, ath)
    print(f"L heurístico = {n_layers}  (pares={num_data}, Ath={ath})")

    # ── §3.1 · Construcción del circuito ────────────────────────────────────
    qc, theta_u, theta_v = build_full_qword2vec_circuit(n_qubits, n_embedding, n_layers)
    all_theta = list(theta_u) + list(theta_v)
    theta_vals = np.random.uniform(0, 2 * np.pi, len(all_theta))

    # ── §3.2 · Distancias objetivo Word2Vec ──────────────────────────────────
    target_distances = build_target_distances(w2v_embeddings, word_to_id, n_qubits)

    # ── §3.1 + §3.3 · Entrenamiento ──────────────────────────────────────────
    theta_vals, loss_history = train_qword2vec(
        qc,
        all_theta,
        theta_vals,
        n_qubits,
        label_vectors,
        target_distances,
        epochs=2000,
        c_val=3,
        learning_rate=0.001,
        momentum=0.9,
    )

    # ── §3.2 · Evaluación ────────────────────────────────────────────────────
    print(f"Calculando vectores de Bloch para {2**n_qubits} palabras...")
    sim = AerSimulator()
    t_circuits = prepare_transpiled_circuits(qc, n_qubits, sim)
    embeddings = get_embeddings_batched(
        theta_vals, all_theta, t_circuits, sim, n_qubits, n_embedding
    )
    print(f"Vectores calculados: {len(embeddings)}")

    save_qword2vec_embeddings(embeddings, id_to_word, "qword2vec_embeddings.txt")

    if w2v_embeddings:
        evaluate_embedding_quality(w2v_embeddings, embeddings, word_to_id)
    else:
        print("[INFO] Sin embeddings Word2Vec — omitiendo evaluación de calidad.")

    # ── Visualizaciones ───────────────────────────────────────────────────────
    if SHOW_VISUALIZATIONS:
        qc_draw, _, _ = build_full_qword2vec_circuit(
            n_qubits, n_embedding, min(n_layers, 2)
        )
        plot_circuit(qc_draw, n_qubits, n_embedding, min(n_layers, 2))
        plot_embeddings_2d(embeddings, id_to_word, word_to_id)
        plot_bloch_sphere(embeddings, n_qubits)
