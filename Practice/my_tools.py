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
    final_probs,
    w2v_embeddings,
    word_to_id,
    id_to_word,
    save_path="embeddings_comparison.png",
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

    colors = plt.cm.hsv(np.linspace(0, 0.9, len(words)))
    _offsets_w2v = [
        (10, 6),
        (-55, 6),
        (10, -16),
        (-55, -16),
        (10, 18),
        (-55, 18),
        (10, -28),
        (-55, -28),
    ]
    _offsets_qw2v = [
        (22, 4),
        (-72, 4),
        (22, -24),
        (-72, -24),
        (22, 28),
        (-72, 28),
        (22, -44),
        (-72, -44),
        (6, 42),
        (-20, 42),
        (6, -50),
        (-20, -50),
        (40, 14),
        (-88, 14),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, coords, title, offsets in [
        (axes[0], w2v_2d, "Word2vec", _offsets_w2v),
        (axes[1], qw2v_2d, "Q-word2vec", _offsets_qw2v),
    ]:
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=80, zorder=3)
        for i, word in enumerate(words):
            ox, oy = offsets[i % len(offsets)]
            ax.annotate(
                word,
                (coords[i, 0], coords[i, 1]),
                textcoords="offset points",
                xytext=(ox, oy),
                fontsize=9,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            )
        ax.set_title(title)

    plt.tight_layout()
    for p in ([save_path] if isinstance(save_path, str) else save_path):
        plt.savefig(p, dpi=150)
        print(f"Gráfica guardada en '{p}'")
    plt.show()


def plot_loss_history(filepath, save_path=None, title_info=None):
    """
    Lee un fichero loss_history_*.txt y dibuja pérdida y error rate en ejes duales.
    Formato esperado (generado por my_qword2vec_qnspsa.py y BayesianFinetuningQNSPSA.py):
        # cabecera opcional con hiperparámetros
        iter,loss,error_rate
        1,4.80,...
    """
    iters, losses, error_rates = [], [], []
    header_info = ""

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                header_info = line[1:].strip()
                continue
            if not line:
                continue
            parts = line.split(",")
            try:
                it = int(parts[0])
            except ValueError:
                continue  # salta cabeceras como "epoch,loss,error_rate"
            if len(parts) >= 3 and it % 10 == 0:
                iters.append(it)
                losses.append(float(parts[1]))
                error_rates.append(float(parts[2]))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(
        iters, losses, color="steelblue", linewidth=1.0, alpha=0.8, label="Pérdida"
    )
    ax1.set_xlabel("Iteración")
    ax1.set_ylabel("Pérdida", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(iters, error_rates, color="darkorange", linewidth=1.2, label="Error rate")
    ax2.set_ylabel("Error rate", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    # Marcar el mejor error rate
    best_er = min(error_rates)
    best_it = iters[error_rates.index(best_er)]
    ax2.axhline(best_er, color="darkorange", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.annotate(
        f"best={best_er:.4f} (it={best_it})",
        xy=(best_it, best_er),
        xytext=(best_it + len(iters) * 0.02, best_er - 0.015),
        fontsize=8,
        color="darkorange",
        va="top",
    )

    title = "Historial de entrenamiento Q-Word2Vec"
    info = title_info if title_info is not None else header_info
    if info:
        title += f"\n{info}"
    ax1.set_title(title, fontsize=10)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    plt.tight_layout()
    if save_path:
        for p in ([save_path] if isinstance(save_path, str) else save_path):
            plt.savefig(p, dpi=150)
            print(f"Gráfica guardada en '{p}'")
    plt.show()


def plot_bloch_sphere(
    qc_data,
    thetas_pv,
    theta_values,
    n_embedding,
    word_to_id,
    id_to_word,
    save_path="bloch_sphere.png",
):
    """
    Visualiza todos los embeddings sobre una única esfera de Bloch.
    Se usa el estado reducido del primer qubit de embedding (trazando el resto)
    y se etiqueta cada punto con el nombre de la palabra.
    """
    from qiskit.quantum_info import Statevector, partial_trace

    words = [id_to_word[i] for i in range(len(word_to_id))]
    colors = plt.cm.hsv(np.linspace(0, 0.9, len(words)))

    # ── Calcular vector de Bloch del qubit 0 de embedding para cada palabra ──
    bvecs = []
    for k in range(len(word_to_id)):
        qc_bound = qc_data[k].assign_parameters({thetas_pv: theta_values})
        qc_no_meas = qc_bound.remove_final_measurements(inplace=False).decompose()
        sv = Statevector(qc_no_meas)
        n_total = qc_no_meas.num_qubits
        trace_out = [i for i in range(n_total) if i != 0]
        rho = partial_trace(sv, trace_out)
        r = rho.data
        bx = 2.0 * r[0, 1].real
        by = -2.0 * r[0, 1].imag
        bz = (r[0, 0] - r[1, 1]).real
        bvecs.append([bx, by, bz])

    # ── Dibujar esfera única ─────────────────────────────────────────────────
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(xs, ys, zs, alpha=0.08, color="lightblue", linewidth=0)

    for d in range(3):
        start, end = [0, 0, 0], [0, 0, 0]
        start[d], end[d] = -1.3, 1.3
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            "k-",
            alpha=0.25,
            linewidth=0.7,
        )

    ax.text(0, 0, 1.45, "|0⟩", ha="center", va="bottom", fontsize=9)
    ax.text(0, 0, -1.50, "|1⟩", ha="center", va="top", fontsize=9)
    ax.text(1.45, 0, 0, "|+⟩", ha="left", fontsize=8, alpha=0.6)
    ax.text(0, 1.45, 0, "|i+⟩", ha="left", fontsize=8, alpha=0.6)

    for k, word in enumerate(words):
        bv = bvecs[k]
        ax.quiver(
            0,
            0,
            0,
            bv[0],
            bv[1],
            bv[2],
            color=colors[k],
            alpha=0.85,
            arrow_length_ratio=0.15,
            linewidth=1.2,
        )
        ax.scatter(*bv, color=colors[k], s=40, zorder=5)
        ax.text(
            bv[0] * 1.15,
            bv[1] * 1.15,
            bv[2] * 1.15,
            word,
            fontsize=8,
            color=colors[k],
            fontweight="bold",
        )

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Q-Word2Vec — Esfera de Bloch (qubit embedding 0)", fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Esfera de Bloch guardada en '{save_path}'")
    plt.show()
