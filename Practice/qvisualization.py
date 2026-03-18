"""
qvisualization.py — Visualization utilities for Q-Word2Vec.

All matplotlib/Bloch-sphere rendering is isolated here so that qWord2Vec.py
can skip it entirely when SHOW_VISUALIZATIONS = False, keeping training runs fast.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization.bloch import Bloch
from sklearn.decomposition import PCA


def plot_circuit(qc, n_qubits, n_embedding, n_layers):
    """
    Draws the QWord2Vec circuit, saves it as PNG, and displays it interactively.

    Saves to:
      - qword2vec_circuit.png  (local working directory)
      - ../memoria/imagenes/qword2vec_circuit.png  (TFG document directory, if it exists)

    Parameters
    ----------
    qc          : QuantumCircuit to draw (typically built with min(n_layers, 2) for clarity)
    n_qubits    : total system qubits
    n_embedding : embedding qubits
    n_layers    : number of layers shown in the title
    """
    cols_per_row = max(1, qc.depth() // 2 + 1)
    fig = qc.draw(output="mpl", fold=cols_per_row)
    fig.suptitle(
        f"QWord2Vec Full Circuit  [U(θ_u) → reset → V(θ_v)]\n"
        f"n={n_qubits} qubits,  ne={n_embedding} embedding,  L={n_layers} layers",
        fontsize=15,
        fontweight="bold",
        y=1.0,
    )

    local_path = "qword2vec_circuit.png"
    fig.savefig(local_path, dpi=150, bbox_inches="tight")
    print(f"Circuit diagram saved to: {local_path}")

    try:
        memoria_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "memoria", "imagenes"
        )
        os.makedirs(memoria_dir, exist_ok=True)
        memoria_path = os.path.join(memoria_dir, "qword2vec_circuit.png")
        fig.savefig(memoria_path, dpi=150, bbox_inches="tight")
        print(f"Circuit diagram saved to: {memoria_path}")
    except Exception:
        print(
            "[Aviso] No se ha guardado la imagen en memoria/imagenes: "
            "esta ruta solo existe dentro del proyecto del TFG."
        )

    plt.show()


def plot_embeddings_2d(
    embeddings, id_to_word, word_to_id, filepath="qword2vec_embeddings_2d.png"
):
    """
    Projects qWord2Vec Bloch vectors to 2D with PCA and displays a labelled scatter plot.

    Parameters
    ----------
    embeddings  : np.ndarray (N, 3*n_embedding) – Bloch-vector embeddings per word
    id_to_word  : dict {int: str}
    word_to_id  : dict {str: int}
    filepath    : output PNG path
    """
    words = [id_to_word[i] for i in range(len(embeddings)) if i in id_to_word]
    vecs = np.array([embeddings[word_to_id[w]] for w in words])

    coords = PCA(n_components=2).fit_transform(vecs)
    xs, ys = coords[:, 0], coords[:, 1]

    _offsets = [
        (10, 6),
        (-55, 6),
        (10, -16),
        (-55, -16),
        (10, 18),
        (-55, 18),
        (10, -28),
        (-55, -28),
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(range(len(words)))
    ax.scatter(xs, ys, c=colors, s=80, zorder=3)
    for i, (word, xi, yi) in enumerate(zip(words, xs, ys)):
        ox, oy = _offsets[i % len(_offsets)]
        ax.annotate(
            word,
            (xi, yi),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=9,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )
    ax.set_title("2D qWord2Vec Embeddings (PCA 3D→2D)")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    print(f"2D embedding plot saved to: {filepath}")
    plt.show()


def plot_bloch_sphere(embeddings, n_qubits):
    """
    Plots all word-embedding Bloch vectors as coloured points on the Bloch sphere.

    Parameters
    ----------
    embeddings : np.ndarray (N, 3) – first three Bloch coordinates (x, y, z)
    n_qubits   : system qubits (used to derive N = 2^n_qubits)
    """
    num_words = 2**n_qubits
    xs = embeddings[:, 0]
    ys = embeddings[:, 1]
    zs = embeddings[:, 2]

    print("Plotting scattered points on Bloch Sphere...")

    try:
        b = Bloch()

        cmap = plt.get_cmap("hsv")
        colors = [cmap(i / num_words) for i in range(num_words)]

        b.point_color = []
        b.point_marker = []

        for i in range(num_words):
            b.add_points([[xs[i]], [ys[i]], [zs[i]]])
            b.point_color.append(colors[i])
            b.point_marker.append("o")

        if hasattr(b, "render"):
            b.render()

        plt.show()

    except NameError:
        print(
            "Error: Bloch class not available. "
            "Please install qutip or check qiskit visualization."
        )
    except TypeError as e:
        print(f"Visualization warning: {e}")
        plt.show()
