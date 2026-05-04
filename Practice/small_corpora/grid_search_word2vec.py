"""
Grid search para encontrar los mejores hiperparámetros de Word2Vec.
Los resultados se usan para fijar BEST_PARAMS en word2vec.py.

Métrica de selección: cosine_delta = media_coseno(pares similares) − media_coseno(pares disimilares).
Cuanto más alto, mejor: el modelo separa correctamente palabras cercanas de lejanas.
"""

from itertools import product

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from word2vec import evaluate

# ── Corpus ────────────────────────────────────────────────────────────────────
sentences = list(LineSentence("smallCorpora.txt"))

# ── Grid ──────────────────────────────────────────────────────────────────────
SEARCH_EPOCHS = 500

param_grid = {
    "vector_size": [2, 3, 4],
    "window": [1, 2, 3],
    "alpha": [0.2, 0.25, 0.275, 0.3],
    "negative": [2, 3, 5, 6],
    "ns_exponent": [0.0, 0.3, 0.75],
}

if __name__ == "__main__":
    total = 1
    for v in param_grid.values():
        total *= len(v)
    print(f"Grid search: {total} combinaciones × {SEARCH_EPOCHS} épocas\n")

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
            print(f"  new best  cosine_delta={score:.4f}  params={params}")

    print(
        f"\nBest params → vector_size={best_params['vector_size']}, "
        f"window={best_params['window']}, alpha={best_params['alpha']}, "
        f"negative={best_params['negative']}, "
        f"ns_exponent={best_params['ns_exponent']} "
        f"|  cosine_delta = {best_score:.4f}"
    )
    print("\nCopia estos valores en BEST_PARAMS de word2vec.py")
