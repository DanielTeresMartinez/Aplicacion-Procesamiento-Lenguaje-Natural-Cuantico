from collections import Counter
import os
import numpy as np
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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
                # ELIMINO REPETIDOS
                if i != j and indices[j] not in training_data[target_idx]:
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
    print(f"p(n={n}, ne={ne}) = {p_val:.3f}")
    return p_val


def estimate_num_layers(n_qubits, n_embedding_qubits, num_data, ath):
    """Aplica la fórmula heurística (ec. 6) para estimar la profundidad L del circuito."""
    p = round(get_error_probability(n_qubits, n_embedding_qubits), 3)
    denominator = 3 * (n_qubits + n_embedding_qubits) * ath
    return int(np.ceil(num_data * p / denominator))


def build_target_distances(w2v_embeddings, word_to_id):
    """
    Construye el vector de distancias d_W2V (§3.2, ec. 5).
    Contiene las distancias euclídeas entre todos los pares de embeddings Word2Vec,
    ordenados por índice de vocabulario. Las palabras ausentes se tratan como cero.
    """
    num_words = len(word_to_id)
    w2v_dim = next(iter(w2v_embeddings.values())).shape[0]
    w2v_ordered = np.zeros((num_words, w2v_dim))
    for word, idx in word_to_id.items():
        if word in w2v_embeddings:
            w2v_ordered[idx] = w2v_embeddings[word]
    dist = pdist(w2v_ordered, metric="euclidean")

    return dist


# =========================================================
# SECCION 4: Implementar el custom error rate que se indica
# =========================================================


def calculate_error_rate(prob_distributions, label_vectors):
    """
    Tasa de error entre las posiciones pico del label y las del modelo (Sección 4).
    """
    total_mismatches = 0
    for k, label in label_vectors.items():
        label_peaks = set(np.argsort(label)[-2:])
        model_peaks = set(np.argsort(prob_distributions[k])[-2:])
        # Convertido a set para hacer una resta de conjuntos, es decir
        # los elementos que están en `label_peaks` pero no en `model_peaks`
        total_mismatches += len(label_peaks - model_peaks)
    return total_mismatches / (2 * len(label_vectors))


def plot_embeddings_comparison(
    final_probs, w2v_embeddings, word_to_id, id_to_word, save_path="embeddings_comparison.png"
):
    """
    Visualización side-by-side de los embeddings Word2Vec y Q-word2vec (como Fig. 8 del paper).
    Q-word2vec: PCA de las distribuciones de probabilidad a 2D.
    Word2Vec:   embeddings directos (ya son 2D).
    """
    words = [id_to_word[i] for i in range(len(word_to_id))]

    # Q-word2vec: PCA de prob_distributions (n_words x 2^n_qubits) → 2D
    pca = PCA(n_components=2)
    qw2v_2d = pca.fit_transform(final_probs)

    # Word2vec: ordenar embeddings por índice de vocabulario
    w2v_dim = next(iter(w2v_embeddings.values())).shape[0]
    w2v_matrix = np.zeros((len(word_to_id), w2v_dim))
    for word, idx in word_to_id.items():
        if word in w2v_embeddings:
            w2v_matrix[idx] = w2v_embeddings[word]

    # Si Word2Vec tiene más de 2 dimensiones, reducir también con PCA
    if w2v_dim > 2:
        w2v_2d = PCA(n_components=2).fit_transform(w2v_matrix)
    else:
        w2v_2d = w2v_matrix

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, coords, title in [
        (axes[0], w2v_2d, "Word2vec"),
        (axes[1], qw2v_2d, "Q-word2vec"),
    ]:
        ax.scatter(coords[:, 0], coords[:, 1])
        for i, word in enumerate(words):
            ax.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=9)
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Gráfica guardada en '{save_path}'")
    plt.show()
