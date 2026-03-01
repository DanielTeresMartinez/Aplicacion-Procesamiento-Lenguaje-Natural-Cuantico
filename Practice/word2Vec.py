from itertools import product

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.callbacks import CallbackAny2Vec
import matplotlib.pyplot as plt

sentences = list(LineSentence("smallCorpora.txt"))

FINE_TUNING = True
FINAL_EPOCHS = 15_000  # 30 sentences × 15k ≈ same total signal as 13 × 30k


class LossCallback(CallbackAny2Vec):
    """Records per-epoch training loss (delta)."""

    def __init__(self):
        self._prev_loss = 0.0
        self.train_losses = []

    def on_epoch_end(self, model):
        cumulative = model.get_latest_training_loss()
        self.train_losses.append(cumulative - self._prev_loss)
        self._prev_loss = cumulative

    def on_train_end(self, model):
        print(f"Training loss (last epoch): {self.train_losses[-1]:.4f}")


# ── Hyperparameter selection ──────────────────────────────────────────────────
if FINE_TUNING:
    # vector_size is fixed to 2; tune window, alpha, negative
    param_grid = {
        "window": [1, 2, 3, 4],
        "alpha": [0.001, 0.005, 0.007, 0.009, 0.01, 0.012, 0.015, 0.02, 0.05],
        "negative": [5, 8, 10, 12, 15, 20],
    }
    SEARCH_EPOCHS = 500

    # Evaluation metric: mean similarity of similar pairs minus dissimilar pairs
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

    best_score, best_params = float("-inf"), {}
    for combo in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combo))
        m = Word2Vec(
            sentences=sentences,
            vector_size=2,
            sg=1,
            min_count=1,
            seed=42,
            compute_loss=False,
            epochs=SEARCH_EPOCHS,
            **params,
        )
        score = evaluate(m)
        if score > best_score:
            best_score, best_params = score, params

    best_params["vector_size"] = 2
    print(
        f"Best params → window={best_params['window']}, alpha={best_params['alpha']}, "
        f"negative={best_params['negative']}  |  score = {best_score:.4f}"
    )
else:
    best_params = {"vector_size": 2, "window": 1, "alpha": 0.012, "negative": 5}

# ── Final model ───────────────────────────────────────────────────────────────
loss_cb = LossCallback()
model = Word2Vec(
    sentences=sentences,
    sg=1,
    min_count=1,
    seed=42,
    compute_loss=True,
    epochs=FINAL_EPOCHS,
    callbacks=[loss_cb],
    **best_params,
)

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
    f"2D Word Embeddings\n" f"win={bp['window']}  α={bp['alpha']}  neg={bp['negative']}"
)
axes[1].set_xlabel("Dimension 1")
axes[1].set_ylabel("Dimension 2")
axes[1].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("loss_curves.png", dpi=150)
plt.show()
print("Plot saved → loss_curves.png")
