from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from scipy.stats import spearmanr

sentences = LineSentence("smallCorpora.txt")

# Passing `sentences` to the constructor calls build_vocab() + train() internally.
model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=3,
    sg=1,
    negative=5,
    # Force all words to be used while training
    min_count=1,
    alpha=0.025,
    min_alpha=1e-4,
    epochs=200,
    seed=42,
    compute_loss=True,
)

loss = model.get_latest_training_loss()
print(f"Training loss: {loss:.4f}")

# Ground truth similarity scores (0.0–1.0) derived from the corpus co-occurrences.
# Pairs that share many contexts → high score; unrelated pairs → low score.
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

print("\nGround truth evaluation (Spearman correlation)")
print(f"{'Pair':<22} {'Human':>6}  {'Model':>6}")
print("-" * 38)

human_scores, model_scores = [], []
for w1, w2, human in ground_truth:
    # .wv attr is to access trained for vectors
    if w1 in model.wv and w2 in model.wv:
        model_sim = model.wv.similarity(w1, w2)
        human_scores.append(human)
        model_scores.append(model_sim)
        print(
            f"  ({w1}, {w2}){'':<{18 - len(w1) - len(w2)}} {human:>6.2f}  {model_sim:>6.4f}"
        )

rho, pvalue = spearmanr(human_scores, model_scores)
print("-" * 38)
print(f"  Spearman ρ = {rho:.4f}  (p = {pvalue:.4f})")
