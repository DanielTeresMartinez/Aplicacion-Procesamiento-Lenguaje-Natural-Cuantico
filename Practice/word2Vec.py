from itertools import product

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.callbacks import CallbackAny2Vec
import matplotlib.pyplot as plt

sentences = list(LineSentence("smallCorpora.txt"))

FINE_TUNING = False
FINAL_EPOCHS = 500  # matches SEARCH_EPOCHS: grid search optimised for this epoch count
PRINT_EVERY = 50


class LossCallback(CallbackAny2Vec):
    """Records per-epoch training loss (delta) and prints it every PRINT_EVERY epochs."""

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


# ── Evaluation metric (always available, used both in grid search and after training) ──
# mean similarity of positive pairs minus mean similarity of negative pairs.
_pos_pairs = [
    ("dog", "cat"),
    ("dog", "animal"),
    ("cat", "animal"),
    ("book", "music"),
    ("fish", "milk"),
]
_neg_pairs = [("book", "dog"), ("movie", "cat"), ("fish", "hate")]


def evaluate(model):
    def mean_sim(pairs):
        sims = [
            model.wv.similarity(a, b)
            for a, b in pairs
            if a in model.wv and b in model.wv
        ]
        return sum(sims) / len(sims) if sims else 0.0

    return mean_sim(_pos_pairs) - mean_sim(_neg_pairs)


# ── Hyperparameter selection ──────────────────────────────────────────────────
if FINE_TUNING:
    # vector_size fixed to 2; sg=1 (skip-gram) mandatory; hs=0 (negative sampling) mandatory.
    #
    # ns_exponent (Gensim docs): shapes the negative-sampling distribution.
    #   0.75 → proportional to frequency (Word2Vec paper default)
    #   0.0  → uniform over all words (useful for tiny vocabularies)
    #   0.5  → intermediate
    #
    # sample: threshold for downsampling high-frequency words.
    #   'like' appears in 63 % of sentences and pulls all vectors toward it.
    #   sample=1e-4 randomly drops some of its occurrences so rarer words get
    #   fairer gradient signal.
    param_grid = {
        "window": [1, 2, 3],
        "alpha": [0.025, 0.05, 0.075, 0.1],
        "negative": [3, 5, 10],
        "ns_exponent": [0.1, 0.25, 0.5, 0.75],
        "sample": [0.0, 1e-5, 1e-4, 1e-3],
    }
    SEARCH_EPOCHS = 1_000  # ~6-7 % of FINAL_EPOCHS; enough to see convergence trends

    total = 1
    for v in param_grid.values():
        total *= len(v)
    print(f"Grid search: {total} combinations × {SEARCH_EPOCHS} epochs each…")

    best_score, best_params = float("-inf"), {}
    for combo in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combo))
        m = Word2Vec(
            sentences=sentences,
            vector_size=2,
            sg=1,
            hs=0,  # negative sampling (mandatory)
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
            print(f"  ↑ new best  score={score:.4f}  params={params}")

    best_params["vector_size"] = 2
    print(
        f"\nBest params → window={best_params['window']}, alpha={best_params['alpha']}, "
        f"negative={best_params['negative']}, ns_exponent={best_params['ns_exponent']}, "
        f"sample={best_params['sample']}  |  score = {best_score:.4f}"
    )
else:
    # Best params found by grid search (score = 1.1207):
    best_params = {
        "vector_size": 2,
        "window": 2,
        "alpha": 0.1,
        "negative": 3,
        "ns_exponent": 0.75,
        "sample": 0.0,
    }

# ── Final model ───────────────────────────────────────────────────────────────
# NOTE: compute_loss=True only works reliably when sentences= and epochs= are
# passed directly to the Word2Vec constructor. Using the explicit build_vocab()
# + train() API does NOT propagate compute_loss to the training workers in
# Gensim 4.x, causing get_latest_training_loss() to always return 0.
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

# ── Semantic quality check ────────────────────────────────────────────────────
# The loss delta is noisy with only 30 sentences (small stochastic gradient).
# The real quality indicator is whether similar words end up close in 2D space.
final_score = evaluate(model)
print(
    f"\nFinal evaluate() score: {final_score:.4f}  (higher = better semantic separation)"
)
print("\nmost_similar() results:")
for word in ["dog", "cat", "book", "fish", "music"]:
    if word in model.wv:
        nbrs = [(w, round(s, 3)) for w, s in model.wv.most_similar(word, topn=3)]
        print(f"  {word:<8} → {nbrs}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1.6]})

# Training loss subsampled to ~200 points for readability
stride = max(1, len(loss_cb.train_losses) // 200)
x_plot = range(0, len(loss_cb.train_losses), stride)
y_plot = loss_cb.train_losses[::stride]
axes[0].plot(x_plot, y_plot, color="steelblue", linewidth=1.2)
axes[0].set_title("Training loss per epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (delta)")

# Word embeddings scatter
vocab = model.wv.index_to_key
xs = [model.wv[w][0] for w in vocab]
ys = [model.wv[w][1] for w in vocab]
colors = plt.cm.tab10(range(len(vocab)))

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
    f"2D Word Embeddings\n"
    f"win={bp['window']}  α={bp['alpha']}  neg={bp['negative']}  "
    f"ns_exp={bp.get('ns_exponent', 0.75)}  sample={bp.get('sample', 0.0)}"
)
axes[1].set_xlabel("Dimension 1")
axes[1].set_ylabel("Dimension 2")
axes[1].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("loss_curves.png", dpi=150)
plt.show()
print("Plot saved → loss_curves.png")
