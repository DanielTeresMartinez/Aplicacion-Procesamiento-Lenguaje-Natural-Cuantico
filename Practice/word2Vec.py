from itertools import product

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

sentences = list(LineSentence("smallCorpora.txt"))

FINE_TUNING = True
FINAL_EPOCHS = 2000
PRINT_EVERY = 200


class LossCallback(CallbackAny2Vec):
    def __init__(self):
        self._prev_loss = 0.0
        self._epoch = 0
        self.train_losses = []

    def on_epoch_end(self, model):
        self._epoch += 1
        cumulative = model.get_latest_training_loss()
        delta = cumulative - self._prev_loss
        self.train_losses.append(delta)
        self._prev_loss = cumulative
        if self._epoch % PRINT_EVERY == 0:
            print(f"  epoch {self._epoch:>6} | loss_delta = {delta:.4f}")

    def on_train_end(self, model):
        print(f"Training loss (last epoch): {self.train_losses[-1]:.4f}")


_pos_pairs = [
    ("dog", "cat"),
    ("dog", "animal"),
    ("cat", "animal"),
    ("book", "music"),
    ("fish", "milk"),
]
_neg_pairs = [("book", "dog"), ("movie", "cat"), ("fish", "hate")]


def _project_2d(model):
    vocab = model.wv.index_to_key
    vecs = np.array([model.wv[w] for w in vocab])
    if vecs.shape[1] > 2:
        vecs = PCA(n_components=2).fit_transform(vecs)
    return {w: vecs[i] for i, w in enumerate(vocab)}


def evaluate_2d(model):
    """Mean cosine sim of positive pairs minus negative pairs, evaluated in 2D (PCA if needed)."""
    wv2d = _project_2d(model)

    def mean_sim(pairs):
        valid = [(a, b) for a, b in pairs if a in wv2d and b in wv2d]
        if not valid:
            return 0.0
        # cosine_similarities accepts plain numpy arrays, works on PCA vectors
        sims = [
            model.wv.cosine_similarities(wv2d[a], np.array([wv2d[b]]))[0]
            for a, b in valid
        ]
        return float(np.mean(sims))

    return mean_sim(_pos_pairs) - mean_sim(_neg_pairs)


def most_similar_2d(model, word, topn=3):
    wv2d = _project_2d(model)
    if word not in wv2d:
        return []
    others = [(w, v) for w, v in wv2d.items() if w != word]
    words, vecs = zip(*others)
    scores = model.wv.cosine_similarities(wv2d[word], np.array(vecs))
    ranked = sorted(zip(words, scores), key=lambda x: x[1], reverse=True)
    return [(w, float(s)) for w, s in ranked[:topn]]


if FINE_TUNING:
    param_grid = {
        "vector_size": [5, 10, 15, 20, 30],
        "window": [2, 3, 4, 5],
        "alpha": [0.075, 0.1, 0.15, 0.2],
        # Num of diff words is 13 in that training dataset
        "negative": [2, 3, 4, 5, 7],
    }
    SEARCH_EPOCHS = 1000

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
            ns_exponent=0.75,
            sample=0.0,
            seed=42,
            compute_loss=False,
            epochs=SEARCH_EPOCHS,
            **params,
        )
        score = evaluate_2d(m)
        if score > best_score:
            best_score, best_params = score, params
            print(f"  ↑ new best  score={score:.4f}  params={params}")

    print(
        f"\nBest params → vector_size={best_params['vector_size']}, "
        f"window={best_params['window']}, alpha={best_params['alpha']}, "
        f"negative={best_params['negative']} |  score = {best_score:.4f}"
    )
else:
    best_params = {
        "vector_size": 10,
        "window": 5,
        "alpha": 0.1,
        "negative": 2,
        "ns_exponent": 0.75,
        "sample": 0.0,
    }

loss_cb = LossCallback()
print(f"\nTraining Skip-Gram (negative sampling) for {FINAL_EPOCHS} epochs…")
print(f"Params: {best_params}\n")
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

final_score = evaluate_2d(model)
print(f"\nFinal evaluate_2d() score: {final_score:.4f}")
print("\nmost_similar_2d() results:")
for word in ["dog", "cat", "book", "fish", "music"]:
    if word in model.wv:
        nbrs = [(w, round(s, 3)) for w, s in most_similar_2d(model, word, topn=3)]
        print(f"  {word:<8} → {nbrs}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1.6]})

stride = max(1, len(loss_cb.train_losses) // 200)
x_plot = range(0, len(loss_cb.train_losses), stride)
y_plot = loss_cb.train_losses[::stride]
axes[0].plot(x_plot, y_plot, color="steelblue", linewidth=1.2)
axes[0].set_title("Training loss per epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (delta)")

wv2d = _project_2d(model)
vocab = model.wv.index_to_key
xs = [wv2d[w][0] for w in vocab]
ys = [wv2d[w][1] for w in vocab]
colors = plt.cm.tab10(range(len(vocab)))
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
