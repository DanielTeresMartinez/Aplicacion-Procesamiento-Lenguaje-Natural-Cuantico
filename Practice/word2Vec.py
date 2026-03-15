import random
from itertools import product

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

_all_sentences = list(LineSentence("smallCorpora.txt"))
random.seed(42)
random.shuffle(_all_sentences)
_split = int(0.8 * len(_all_sentences))
train_sentences = _all_sentences[:_split]
val_sentences = _all_sentences[_split:]

# `sentences` alias keeps the grid-search branch (which uses `sentences=sentences`)
# pointing at training data only — it never sees the validation split.
sentences = train_sentences

FINE_TUNING = False
FINAL_EPOCHS = 2000
PRINT_EVERY = 200
PATIENCE = 5
MIN_DELTA = 0.01


class EarlyStopping(Exception):
    pass


_clusters = {
    "animal": ["dog", "cat", "animal", "wild", "pet", "eyes"],
    "food": ["fish", "milk", "food", "water", "apple"],
    "culture": ["book", "music", "song", "read", "art"],
    "movie": ["movie", "screen", "watch", "hate", "bad"],
}
_word_to_cluster = {w: c for c, words in _clusters.items() for w in words}
_train_words = {w for sent in train_sentences for w in sent}
_val_words = {w for sent in val_sentences for w in sent}
_eval_vocab = [w for w in _word_to_cluster if w in _val_words and w in _train_words]
_eval_labels = [_word_to_cluster[w] for w in _eval_vocab]


def _project_2d(model):
    vocab = model.wv.index_to_key
    vecs = np.array([model.wv[w] for w in vocab])
    if vecs.shape[1] > 2:
        vecs = PCA(n_components=2).fit_transform(vecs)
    return {w: vecs[i] for i, w in enumerate(vocab)}


def evaluate(model):
    return float(silhouette_score(model.wv[_eval_vocab], _eval_labels, metric="cosine"))


class LossCallback(CallbackAny2Vec):
    def __init__(self):
        self._epoch = 0
        self.train_losses = []  # training loss delta per epoch (kept for plot)
        self._prev_loss = 0.0  # used only to compute the per-epoch train delta

        self.val_scores = []  # (epoch, silhouette_score) at each checkpoint
        self._best_val_score = float("-inf")
        self._no_improve_count = 0
        self.converged_at = None
        self.model = None
        self._best_checkpoint = "_best_checkpoint.model"

    def on_epoch_end(self, model):
        self.model = model
        self._epoch += 1

        # --- training loss delta (kept only for the plot) ---
        cumulative = model.get_latest_training_loss()
        delta = cumulative - self._prev_loss
        self.train_losses.append(delta)
        self._prev_loss = cumulative

        # --- validation check every PRINT_EVERY epochs ---
        if self._epoch % PRINT_EVERY == 0:
            val_score = evaluate(model)  # silhouette score — validation proxy
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
                f"  epoch {self._epoch:>6} | train_loss_delta = {delta:.4f}"
                f" | val_score = {val_score:.4f}{improved_tag}"
                f"  (no_improve={self._no_improve_count}/{PATIENCE})"
            )

            if self._no_improve_count >= PATIENCE:
                self.converged_at = self._epoch
                raise EarlyStopping(
                    f"No improvement for {PATIENCE} checks — stopping at epoch {self._epoch}"
                )

    def on_train_end(self):
        if self.val_scores:
            last_epoch, last_score = self.val_scores[-1]
            print(
                f"Validation score (last checkpoint, epoch {last_epoch}): {last_score:.4f}"
            )
        if self.converged_at:
            print(f"Converged at epoch {self.converged_at} (early stopping).")


def most_similar(model, word, topn=3):
    if word not in model.wv:
        return []
    return [(w, float(s)) for w, s in model.wv.most_similar(word, topn=topn)]


if FINE_TUNING:
    param_grid = {
        "vector_size": [3, 4, 5],
        "window": [1, 2],
        "alpha": [0.2, 0.25, 0.3],
        "negative": [6, 7, 8, 9],
    }
    SEARCH_EPOCHS = 500

    total = 1
    for v in param_grid.values():
        total *= len(v)
    print(f"Grid search: {total} combinations × {SEARCH_EPOCHS} epochs\n")

    best_score, best_params = float("-inf"), {}
    for combo in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combo))
        m = Word2Vec(
            sentences=sentences,
            sg=1,
            hs=0,
            workers=1,
            min_count=1,
            seed=42,
            compute_loss=False,
            epochs=SEARCH_EPOCHS,
            **params,
        )
        score = evaluate(m)
        if score > best_score:
            best_score, best_params = score, params
            print(f"  new best  score={score:.4f}  params={params}")

    print(
        f"\nBest params → vector_size={best_params['vector_size']}, "
        f"window={best_params['window']}, alpha={best_params['alpha']}, "
        f"negative={best_params['negative']} |  score = {best_score:.4f}"
    )
else:
    best_params = {
        "vector_size": 3,
        "window": 1,
        "alpha": 0.25,
        "negative": 6,
    }

loss_cb = LossCallback()
print(f"\nTraining Skip-Gram (negative sampling) for up to {FINAL_EPOCHS} epochs…")
print(f"Params: {best_params}\n")
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
        **best_params,
    )
except EarlyStopping as e:
    print(f"\n[Early stopping] {e}")
    model = loss_cb.model

# Load the best checkpoint regardless of whether early stopping fired.
# Without this, model has last-epoch weights, not best-epoch weights.
model = Word2Vec.load(loss_cb._best_checkpoint)

final_score = evaluate(model)
print(f"\nFinal evaluate() score: {final_score:.4f}")
print("\nmost_similar() results:")
for word in ["dog", "cat", "book", "fish", "music"]:
    if word in model.wv:
        nbrs = [(w, round(s, 3)) for w, s in most_similar(model, word, topn=3)]
        print(f"  {word:<8} → {nbrs}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1.6]})

# --- left panel: validation score at each checkpoint ---
if loss_cb.val_scores:
    ck_epochs = [e for e, _ in loss_cb.val_scores]
    ck_scores = [s for _, s in loss_cb.val_scores]
    axes[0].plot(
        ck_epochs,
        ck_scores,
        color="darkorange",
        linewidth=1.4,
        marker="o",
        markersize=4,
        label="val silhouette score",
    )
    best_idx = int(np.argmax(ck_scores))
    best_epoch = ck_epochs[best_idx]
    axes[0].axvline(
        best_epoch,
        color="green",
        linestyle="--",
        linewidth=1,
        label=f"best (epoch {best_epoch})",
    )
    axes[0].legend(fontsize=8)

axes[0].set_title("Validation score at each checkpoint\n(silhouette, higher = better)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Silhouette score")

# faint training-loss overlay for reference (secondary y-axis)
stride = max(1, len(loss_cb.train_losses) // 200)
# epoch numbers are 1-indexed: epoch 1 = train_losses[0]
train_epochs = list(range(1, len(loss_cb.train_losses) + 1, stride))
train_deltas = loss_cb.train_losses[::stride]

ax0b = axes[0].twinx()
ax0b.plot(
    train_epochs,
    train_deltas,
    color="steelblue",
    linewidth=0.6,
    alpha=0.35,
    label="train loss Δ",
)
ax0b.set_ylabel("Train loss delta", color="steelblue", fontsize=8)
ax0b.tick_params(axis="y", labelcolor="steelblue", labelsize=7)

wv2d = _project_2d(model)
vocab = model.wv.index_to_key
xs = [wv2d[w][0] for w in vocab]
ys = [wv2d[w][1] for w in vocab]
colors = plt.cm.hsv(np.linspace(0, 0.9, len(vocab)))
vsize = best_params["vector_size"]
pca_note = f" (PCA {vsize}D→2D)" if vsize > 2 else ""

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

axes[1].scatter(xs, ys, c=colors, s=80, zorder=3)
for i, (word, xi, yi) in enumerate(zip(vocab, xs, ys)):
    ox, oy = _offsets[i % len(_offsets)]
    axes[1].annotate(
        word,
        (xi, yi),
        textcoords="offset points",
        xytext=(ox, oy),
        fontsize=9,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
    )
bp = best_params
axes[1].set_title(
    f"2D Word Embeddings{pca_note}\n"
    f"vsize={bp['vector_size']}  win={bp['window']}  α={bp['alpha']}  "
    f"neg={bp['negative']}  ns_exp={bp.get('ns_exponent', 0.75)}"
)
axes[1].set_xlabel("Dimension 1")
axes[1].set_ylabel("Dimension 2")
axes[1].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("loss_curves.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# Export embeddings to disk so that qWord2Vec can load them.
#
# Two files are saved:
#   word2vec_embeddings.txt     – original high-dim vectors (one word per line)
#   word2vec_embeddings_2d.txt  – PCA-projected 2-D vectors  (one word per line)
#
# Format (both files):
#   Line 0: "# word  <col_headers>"  (comment / header)
#   Lines 1+: "<word>  v1  v2  ...  vN"
# ---------------------------------------------------------------------------


def _save_embeddings(filepath, word_vec_dict, header_comment=""):
    """Save a {word: np.ndarray} dict to a plain-text file."""
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
    print(f"Embeddings saved → {filepath}  ({len(words)} words, dim={dim})")


vocab = model.wv.index_to_key
orig_embs = {w: model.wv[w] for w in vocab}
_save_embeddings(
    "word2vec_embeddings.txt",
    orig_embs,
    header_comment=f"vector_size={best_params['vector_size']}",
)
