import random
import time
import shutil

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt

# ── Hiperparámetros ───────────────────────────────────────────────────────────
FINAL_EPOCHS = 2000
PRINT_EVERY = 200
PATIENCE = 5
MIN_DELTA = 0.01

# Mejores parámetros obtenidos mediante grid search (grid_search_word2vec.py)
BEST_PARAMS = {
    "vector_size": 2,
    "window": 1,
    "alpha": 0.25,
    "negative": 5,
    "ns_exponent": 0.0,
}

# ── Ground truth de clústeres (para evaluación con K-Means) ──────────────────
_clusters = {
    "animal": ["dog", "cat", "animal", "eyes"],
    "food": ["apple", "fish", "milk"],
    "culture": ["book", "music", "movie"],
    "sentiment": ["i", "like", "hate"],
}
_word_to_cluster = {w: c for c, words in _clusters.items() for w in words}
_cluster_names = list(_clusters.keys())
_cluster_to_int = {c: i for i, c in enumerate(_cluster_names)}
_all_gt_words = list(_word_to_cluster.keys())
_all_gt_ints = [_cluster_to_int[_word_to_cluster[w]] for w in _all_gt_words]


class EarlyStopping(Exception):
    pass


def evaluate(model):
    """K-Means accuracy contra el ground truth de _clusters.

    K-Means asigna etiquetas arbitrarias (0-3), así que se usa el algoritmo
    húngaro para encontrar la mejor correspondencia entre clústeres y categorías
    antes de calcular el accuracy.
    """
    valid = [(w, gt) for w, gt in zip(_all_gt_words, _all_gt_ints) if w in model.wv]
    words, gt_ints = zip(*valid)

    vectors = np.array([model.wv[w] for w in words])
    n_clusters = len(_cluster_names)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    predicted = km.fit_predict(vectors)

    cm = confusion_matrix(gt_ints, predicted, labels=list(range(n_clusters)))
    _, col_ind = linear_sum_assignment(-cm)
    return float(cm[range(n_clusters), col_ind].sum() / len(words))


class LossCallback(CallbackAny2Vec):
    def __init__(self):
        self._epoch = 0
        self.train_losses = []
        self._prev_loss = 0.0
        self.val_scores = []
        self._best_val_score = float("-inf")
        self._no_improve_count = 0
        self.converged_at = None
        self.model = None
        self._best_checkpoint = "_best_checkpoint.model"

    def on_epoch_end(self, model):
        self.model = model
        self._epoch += 1

        cumulative = model.get_latest_training_loss()
        delta = cumulative - self._prev_loss
        self.train_losses.append(delta)
        self._prev_loss = cumulative

        if self._epoch % PRINT_EVERY == 0:
            val_score = evaluate(model)
            self.val_scores.append((self._epoch, val_score))

            if val_score > self._best_val_score + MIN_DELTA:
                self._best_val_score = val_score
                self._no_improve_count = 0
                model.save(self._best_checkpoint)
                improved_tag = " (new best)"
            else:
                self._no_improve_count += 1
                improved_tag = ""

            print(
                f"  época {self._epoch:>6} | pérdida_delta = {delta:.4f}"
                f" | kmeans_acc = {val_score:.4f}{improved_tag}"
                f"  (sin_mejora={self._no_improve_count}/{PATIENCE})"
            )

            if self._no_improve_count >= PATIENCE:
                self.converged_at = self._epoch
                raise EarlyStopping(
                    f"Sin mejora en {PATIENCE} comprobaciones — parada en época {self._epoch}"
                )

    def on_train_end(self, model):
        if self.val_scores:
            last_epoch, last_score = self.val_scores[-1]
            print(
                f"K-Means accuracy (último checkpoint, época {last_epoch}): {last_score:.4f}"
            )
        if self.converged_at:
            print(f"Convergido en época {self.converged_at} (parada temprana).")


def most_similar(model, word, topn=3):
    if word not in model.wv:
        return []
    return [(w, float(s)) for w, s in model.wv.most_similar(word, topn=topn)]


def _save_embeddings(filepath, word_vec_dict, header_comment=""):
    """Guarda un dict {word: np.ndarray} en un fichero de texto plano."""
    words = list(word_vec_dict.keys())
    dim = len(word_vec_dict[words[0]])
    col_header = "  ".join(f"d{i}" for i in range(dim))
    with open(filepath, "w") as f:
        f.write(f"# word  {col_header}")
        if header_comment:
            f.write(f"  # {header_comment}")
        f.write("\n")
        for word, vec in word_vec_dict.items():
            vec_str = "  ".join(f"{v:.8f}" for v in vec)
            f.write(f"{word}  {vec_str}\n")
    print(f"Embeddings guardados → {filepath}  ({len(words)} palabras, dim={dim})")


if __name__ == "__main__":

    sentences = list(LineSentence("smallCorpora.txt"))
    random.seed(42)
    random.shuffle(sentences)
    print(f"Corpus cargado: {len(sentences)} frases")

    loss_cb = LossCallback()
    print(f"\nEntrenando Skip-Gram (negative sampling) hasta {FINAL_EPOCHS} épocas…")
    print(f"Parámetros: {BEST_PARAMS}\n")

    t_start = time.time()
    try:
        model = Word2Vec(
            sentences=sentences,
            sg=1,
            hs=0,
            min_count=1,
            workers=1,
            seed=42,
            compute_loss=True,
            epochs=FINAL_EPOCHS,
            callbacks=[loss_cb],
            **BEST_PARAMS,
        )
    except EarlyStopping as e:
        print(f"\n[Early stopping] {e}")
        model = loss_cb.model

    elapsed = time.time() - t_start
    print(f"Tiempo de entrenamiento: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # ── 3. Cargar el mejor checkpoint ─────────────────────────────────────────
    model = Word2Vec.load(loss_cb._best_checkpoint)

    # ── 4. Evaluación final ───────────────────────────────────────────────────
    final_score = evaluate(model)
    print(f"\nK-Means accuracy final: {final_score:.4f}")
    print("\nPalabras más similares:")
    for word in ["dog", "cat", "book", "fish", "music"]:
        if word in model.wv:
            nbrs = [(w, round(s, 3)) for w, s in most_similar(model, word, topn=3)]
            print(f"  {word:<8} → {nbrs}")

    # ── 5. Gráficas ───────────────────────────────────────────────────────────
    # Figura 1: curvas de entrenamiento
    fig1, ax0 = plt.subplots(figsize=(8, 6))

    if loss_cb.val_scores:
        ck_epochs = [e for e, _ in loss_cb.val_scores]
        ck_scores = [s for _, s in loss_cb.val_scores]
        ax0.plot(
            ck_epochs,
            ck_scores,
            color="darkorange",
            linewidth=1.4,
            marker="o",
            markersize=4,
            label="K-Means accuracy",
        )
        best_epoch = ck_epochs[int(np.argmax(ck_scores))]
        ax0.axvline(
            best_epoch,
            color="green",
            linestyle="--",
            linewidth=1,
            label=f"best (epoch {best_epoch})",
        )
        ax0.legend(fontsize=8)

    ax0.set_title(
        "K-Means accuracy at each checkpoint\n(vs. ground truth, higher = better)"
    )
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("K-Means accuracy")

    stride = max(1, len(loss_cb.train_losses) // 200)
    train_epochs = list(range(1, len(loss_cb.train_losses) + 1, stride))
    train_vals = loss_cb.train_losses[::stride]

    ax0b = ax0.twinx()
    ax0b.plot(
        train_epochs,
        train_vals,
        color="steelblue",
        linewidth=0.6,
        alpha=0.35,
        label="train loss Δ",
    )
    ax0b.set_ylabel("Train loss delta", color="steelblue", fontsize=8)
    ax0b.tick_params(axis="y", labelcolor="steelblue", labelsize=7)

    fig1.tight_layout()
    fig1.savefig("lossCurvesWord2Vec.png", dpi=150)
    shutil.copy("lossCurvesWord2Vec.png", "../memoria/imagenes/lossCurvesWord2Vec.png")

    # Figura 2: embeddings 2D
    fig2, ax1 = plt.subplots(figsize=(8, 6))

    vocab = model.wv.index_to_key
    xs = [model.wv[w][0] for w in vocab]
    ys = [model.wv[w][1] for w in vocab]
    colors = plt.cm.hsv(np.linspace(0, 0.9, len(vocab)))

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

    ax1.scatter(xs, ys, c=colors, s=80, zorder=3)
    for i, (word, xi, yi) in enumerate(zip(vocab, xs, ys)):
        ox, oy = _offsets[i % len(_offsets)]
        ax1.annotate(
            word,
            (xi, yi),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=9,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )

    bp = BEST_PARAMS
    ax1.set_title(
        f"2D Word Embeddings\n"
        f"vsize={bp['vector_size']}  win={bp['window']}  α={bp['alpha']}  "
        f"neg={bp['negative']}  ns_exp={bp.get('ns_exponent', 0.75)}"
    )
    ax1.set_xlabel("Dimension 1")
    ax1.set_ylabel("Dimension 2")
    ax1.grid(True, linestyle="--", alpha=0.4)

    fig2.tight_layout()
    fig2.savefig("embeddingsWord2Vec.png", dpi=150)
    shutil.copy("embeddingsWord2Vec.png", "../memoria/imagenes/embeddingsWord2Vec.png")

    # ── 6. Exportar embeddings para Q-word2vec ────────────────────────────────
    orig_embs = {w: model.wv[w] for w in vocab}
    _save_embeddings(
        "word2vec_embeddings.txt",
        orig_embs,
        header_comment=f"vector_size={BEST_PARAMS['vector_size']}",
    )
