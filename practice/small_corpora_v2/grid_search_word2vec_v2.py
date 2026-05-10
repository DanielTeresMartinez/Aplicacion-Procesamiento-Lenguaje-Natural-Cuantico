"""
Grid search para encontrar los mejores hiperparámetros de Word2Vec (corpus v2).
Los resultados se usan para fijar BEST_PARAMS en word2vec_v2.py.

Métrica de selección: cosine_delta = media_coseno(pares similares) − media_coseno(pares disimilares).
Cuanto más alto, mejor: el modelo separa correctamente palabras cercanas de lejanas.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from itertools import product

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np

from word2vec_v2 import evaluate

# ── Corpus ────────────────────────────────────────────────────────────────────
sentences = list(LineSentence("smallCorporaV2.txt"))


# Grid refinado: vector_size y epochs se expanden donde llegaron al límite anterior.
# Alpha se afina con linspace más estrecho alrededor del 0.15 óptimo.
param_grid = {
    "vector_size": [2, 4, 6],
    "window": [2, 3],
    "alpha": np.logspace(np.log10(0.005), np.log10(0.25), 10),
    "epochs": [300, 400, 500],
    "negative": [2, 3],
    "ns_exponent": [0.2, 0.4],
}

if __name__ == "__main__":
    total = 1
    for v in param_grid.values():
        total *= len(v)
    print(f"Grid search: {total} combinaciones\n")

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
            **params,
        )
        score = evaluate(m)
        if score > best_score:
            best_score, best_params = score, params
            print(f"  new best  cosine_delta={score:.4f}  params={params}")

    print(
        f"\nBest params → vector_size={best_params['vector_size']}, "
        f"window={best_params['window']}, alpha={best_params['alpha']:.4f}, "
        f"epochs={int(best_params['epochs'])}, "
        f"negative={best_params['negative']}, "
        f"ns_exponent={best_params['ns_exponent']} "
        f"|  cosine_delta = {best_score:.4f}"
    )
    print("\nCopia estos valores en BEST_PARAMS de word2vec_v2.py")
