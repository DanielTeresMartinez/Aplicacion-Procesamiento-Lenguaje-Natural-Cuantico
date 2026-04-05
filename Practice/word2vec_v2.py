import random
import time
import shutil

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.callbacks import CallbackAny2Vec
import numpy as np
import matplotlib.pyplot as plt
from my_tools import kmeans_cluster_accuracy

# ── Hiperparámetros ───────────────────────────────────────────────────────────
FINAL_EPOCHS = 2000
PRINT_EVERY = 10  # imprime cada N épocas
PATIENCE = 15  # checks sin mejora antes de parar
MIN_DELTA = 0.001  # mejora mínima en loss_delta para contar como mejora
LOSS_FILE = "loss_history_word2vec.txt"

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
_all_gt_words = list(_word_to_cluster.keys())


class EarlyStopping(Exception):
    pass


def evaluate(model):
    """K-Means accuracy contra el ground truth de _clusters."""
    word_vectors = {w: model.wv[w] for w in _all_gt_words if w in model.wv}
    return kmeans_cluster_accuracy(word_vectors, _word_to_cluster, _cluster_names)


class LossCallback(CallbackAny2Vec):
    """Callback que calcula el loss por época sin interferir en Gensim.

    Gensim acumula `running_training_loss` a lo largo de todo el entrenamiento.
    En lugar de resetearlo (que podría interferir con el estado interno),
    calculamos el loss de cada época como:

        loss_época = acumulado_ahora - acumulado_época_anterior

    Es aritméticamente equivalente al reset pero sin tocar ningún atributo
    interno de la librería.

    Early stopping: guardamos checkpoint cuando el loss de la época es menor
    que el mejor hasta ahora (MIN_DELTA de margen).  Las épocas que Gensim
    reporta como 0.0 (corpus pequeño, sin pares válidos) se ignoran para
    no disparar la parada prematuramente.
    El K-Means accuracy se calcula solo una vez al final sobre el mejor
    checkpoint.
    """

    def __init__(self):
        self._epoch = 0               # épocas reales de Gensim
        self.train_losses = []        # (epoch_gensim, loss) solo para épocas válidas
        self._cumulative_prev = 0.0   # acumulado al final de la época anterior
        self._best_loss = float("inf")
        self._no_improve_count = 0
        self.converged_at = None
        self.best_epoch = None
        self.model = None
        self._best_checkpoint = "_best_checkpoint.model"
        self._loss_file = open(LOSS_FILE, "w")
        self._loss_file.write("epoch,loss\n")

    # ── Sin on_epoch_begin: no tocamos ningún atributo interno de Gensim ──────

    def on_epoch_end(self, model):
        self.model = model
        self._epoch += 1

        # Loss de esta época = diferencia de acumulados
        cumulative_now = model.get_latest_training_loss()
        epoch_loss = cumulative_now - self._cumulative_prev
        self._cumulative_prev = cumulative_now

        # Guardar solo épocas válidas (para las gráficas)
        if epoch_loss > 0.0:
            self.train_losses.append((self._epoch, epoch_loss))

        # Solo actuar en múltiplos exactos de PRINT_EVERY (10, 20, 30...)
        if self._epoch % PRINT_EVERY != 0:
            return

        # Si en esta época Gensim no procesó pares, la saltamos en silencio
        if epoch_loss == 0.0:
            return

        # ¿El loss mejora (baja)?
        if epoch_loss < self._best_loss - MIN_DELTA:
            self._best_loss = epoch_loss
            self._no_improve_count = 0
            model.save(self._best_checkpoint)
            self.best_epoch = self._epoch
            improved_tag = " (new best)"
        else:
            self._no_improve_count += 1
            improved_tag = ""

        self._loss_file.write(f"{self._epoch},{epoch_loss:.6f}\n")
        self._loss_file.flush()
        print(
            f"  época {self._epoch:>6} | loss = {epoch_loss:>10.4f}{improved_tag}"
            f"  (sin_mejora={self._no_improve_count}/{PATIENCE})"
        )

        if self._no_improve_count >= PATIENCE:
            self.converged_at = self._epoch
            raise EarlyStopping(
                f"Sin mejora en {PATIENCE} comprobaciones — parada en época {self._epoch}"
            )

    def on_train_end(self, model):
        self._loss_file.close()
        if self.converged_at:
            print(f"\nConvergido en época {self.converged_at} (parada temprana).")


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
    import os

    if os.path.exists(loss_cb._best_checkpoint):
        model = Word2Vec.load(loss_cb._best_checkpoint)
        print(f"\nMejor checkpoint cargado (época {loss_cb.best_epoch})")
    else:
        print("\n[AVISO] No se encontró checkpoint; usando modelo actual.")

    # ── 4. Evaluación final (K-Means sobre el mejor checkpoint) ───────────────
    final_score = evaluate(model)
    print(f"K-Means accuracy (mejor época={loss_cb.best_epoch}): {final_score:.4f}")

    # ── 5. Gráficas ───────────────────────────────────────────────────────────
    # Figura 1: curvas de entrenamiento
    fig1, ax0 = plt.subplots(figsize=(8, 6))

    # train_losses contiene solo épocas válidas: [(epoch_gensim, loss), ...]
    if loss_cb.train_losses:
        all_epochs, all_vals = zip(*loss_cb.train_losses)
    else:
        all_epochs, all_vals = [], []

    # Puntos de la gráfica: épocas múltiplo de PRINT_EVERY con loss > 0
    pairs_ck = [(e, l) for e, l in zip(all_epochs, all_vals) if e % PRINT_EVERY == 0]
    ck_epochs_f = [e for e, _ in pairs_ck]
    ck_losses_f = [l for _, l in pairs_ck]

    if ck_losses_f:
        ax0.plot(
            ck_epochs_f,
            ck_losses_f,
            color="darkorange",
            linewidth=1.4,
            marker="o",
            markersize=4,
            label="Train loss (checkpoint epochs)",
        )
        if loss_cb.best_epoch is not None:
            ax0.axvline(
                loss_cb.best_epoch,
                color="green",
                linestyle="--",
                linewidth=1,
                label=f"best (epoch {loss_cb.best_epoch})",
            )
        ax0.legend(fontsize=8)

    ax0.set_title("Train loss per valid epoch\n(lower = better)")
    ax0.set_xlabel("Epoch (Gensim)")
    ax0.set_ylabel("Loss")

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
