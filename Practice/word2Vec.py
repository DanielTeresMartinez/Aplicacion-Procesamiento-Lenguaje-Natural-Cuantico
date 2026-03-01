from itertools import product

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.callbacks import CallbackAny2Vec
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

sentences = list(LineSentence("smallCorpora.txt"))

FINE_TUNING = False
FINAL_EPOCHS = 30_000  # article spec

ground_truth = [
    ("dog", "cat", 1.0),
    ("dog", "animal", 0.8),
    ("cat", "animal", 0.8),
    ("book", "music", 0.9),
    ("like", "hate", 0.4),
    ("apple", "dog", 0.5),
    ("fish", "milk", 0.6),
    ("book", "dog", 0.1),
    ("movie", "cat", 0.1),
    ("fish", "hate", 0.0),
]


class LossCallback(CallbackAny2Vec):
    """Records per-epoch training loss (delta) and validation loss (1 - Spearman ρ)."""

    def __init__(self, ground_truth):
        self.ground_truth = ground_truth
        self._prev_loss = 0.0
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, model):
        cumulative = model.get_latest_training_loss()
        self.train_losses.append(cumulative - self._prev_loss)
        self._prev_loss = cumulative

        human_scores, model_scores = [], []
        for w1, w2, human in self.ground_truth:
            if w1 in model.wv and w2 in model.wv:
                human_scores.append(human)
                model_scores.append(model.wv.similarity(w1, w2))

        rho, _ = (
            spearmanr(human_scores, model_scores)
            if len(model_scores) >= 2
            else (float("nan"), None)
        )
        self.val_losses.append(1 - rho)

    def on_train_end(self, model):
        human_scores, model_scores = [], []
        for w1, w2, human in self.ground_truth:
            if w1 in model.wv and w2 in model.wv:
                human_scores.append(human)
                model_scores.append(model.wv.similarity(w1, w2))
        rho, p = spearmanr(human_scores, model_scores)
        print(f"Training loss: {self.train_losses[-1]:.4f}")
        print(f"Spearman ρ = {rho:.4f}  (p = {p:.4f})")


# ── Hyperparameter selection ──────────────────────────────────────────────────
if FINE_TUNING:
    # vector_size is fixed to 2 (article spec); tune the remaining hyperparams
    param_grid = {
        "window": [1, 2, 3, 4],
        "alpha": [0.001, 0.005, 0.007, 0.009, 0.01, 0.012, 0.015, 0.02, 0.05],
        "negative": [5, 8, 10, 12, 15, 20],
    }
    SEARCH_EPOCHS = 500

    def evaluate(model):
        human_scores, model_scores = [], []
        for w1, w2, human in ground_truth:
            if w1 in model.wv and w2 in model.wv:
                human_scores.append(human)
                model_scores.append(model.wv.similarity(w1, w2))
        if len(model_scores) < 2:
            return float("-inf")
        rho, _ = spearmanr(human_scores, model_scores)
        return rho

    best_rho, best_params = float("-inf"), {}
    for combo in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combo))
        m = Word2Vec(
            sentences=sentences,
            vector_size=2,
            sg=1,
            min_count=1,
            seed=42,
            workers=1,
            compute_loss=False,
            epochs=SEARCH_EPOCHS,
            **params,
        )
        rho = evaluate(m)
        if rho > best_rho:
            best_rho, best_params = rho, params

    best_params["vector_size"] = 2
    print(
        f"Best params → window={best_params['window']}, alpha={best_params['alpha']}, "
        f"negative={best_params['negative']}  |  ρ = {best_rho:.4f}"
    )
else:
    best_params = {"vector_size": 2, "window": 3, "alpha": 0.01, "negative": 12}

# ── Final model ───────────────────────────────────────────────────────────────
loss_cb = LossCallback(ground_truth)
model = Word2Vec(
    sentences=sentences,
    sg=1,
    min_count=1,
    seed=42,
    workers=1,
    compute_loss=True,
    epochs=FINAL_EPOCHS,
    callbacks=[loss_cb],
    **best_params,
)

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Training loss: subsample to ~200 points so the curve is readable
stride = max(1, len(loss_cb.train_losses) // 200)
epochs = range(0, len(loss_cb.train_losses), stride)
sampled = loss_cb.train_losses[::stride]
axes[0].plot(epochs, sampled, color="steelblue", linewidth=1.2)
axes[0].set_title("Training loss per epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (delta)")

axes[1].plot(loss_cb.val_losses, color="tomato", linewidth=0.8)
axes[1].axhline(
    y=0.5, color="gray", linestyle="--", linewidth=0.8, label="ρ = 0.5 baseline"
)
axes[1].set_title("Validation loss per epoch  (1 − Spearman ρ)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("1 − ρ")
axes[1].legend()

# Word embeddings: one color per word + alternating label offsets
vocab = model.wv.index_to_key
xs = [model.wv[w][0] for w in vocab]
ys = [model.wv[w][1] for w in vocab]
colors = plt.cm.tab10(range(len(vocab)))

_offsets = [
    (8, 4),
    (-50, 4),
    (8, -14),
    (-50, -14),
    (8, 14),
    (-50, 14),
    (8, -24),
    (-50, -24),
]

axes[2].scatter(xs, ys, c=colors, s=60, zorder=3)
for i, (word, xi, yi) in enumerate(zip(vocab, xs, ys)):
    ox, oy = _offsets[i % len(_offsets)]
    axes[2].annotate(
        word,
        (xi, yi),
        textcoords="offset points",
        xytext=(ox, oy),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.4),
    )
bp = best_params
axes[2].set_title(
    f"2D Word Embeddings\n" f"win={bp['window']}  α={bp['alpha']}  neg={bp['negative']}"
)
axes[2].set_xlabel("Dimension 1")
axes[2].set_ylabel("Dimension 2")
axes[2].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("loss_curves.png", dpi=150)
plt.show()
print("Plot saved → loss_curves.png")
