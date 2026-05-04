"""
Grid search para encontrar los mejores hiperparámetros de Word2Vec.
Los resultados se usan para fijar BEST_PARAMS en word2vec.py.

Métrica de selección: cosine_delta = media_coseno(pares similares) − media_coseno(pares disimilares).
Cuanto más alto, mejor: el modelo separa correctamente palabras cercanas de lejanas.
"""

from itertools import product

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np

from word2vec import evaluate

# ── Corpus ────────────────────────────────────────────────────────────────────
sentences = list(LineSentence("smallCorpora.txt"))

# ===================================================================================
# Se ha ajustado esto y se va a quitar el EarlyStopping del código principal.
# Comentarlo en el pdf de correcciones todo esto. Falta imprimir el mejor número de
# épocas encontrado también.
# ===================================================================================
param_grid = {
    "vector_size": [2, 4],
    "window": [1, 2],
    "alpha": np.linspace(0.35, 0.75, 6),
    "epochs": np.arange(300, 600, 100),
    "negative": [2, 3],
    "ns_exponent": [0.0, 0.2],
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
        f"window={best_params['window']}, alpha={best_params['alpha']}, "
        f"negative={best_params['negative']}, "
        f"ns_exponent={best_params['ns_exponent']} "
        f"|  cosine_delta = {best_score:.4f}"
    )
    print("\nCopia estos valores en BEST_PARAMS de word2vec.py")
