# Cambio de métrica: error_rate → cosine_delta

## Motivación

La evaluación final de Q-Word2Vec se hacía con **cosine_delta** (igual que Word2Vec),
pero el criterio de parada del entrenamiento usaba **error_rate** (tasa de error en top-2 picos).
Esto crea una desalineación: el modelo para cuando deja de mejorar en una métrica pero se
evalúa con otra, sin garantía de que ambas sean óptimas a la vez.

Al unificar criterio de parada y métrica final en **cosine_delta** para ambos modelos, la
comparación Word2Vec vs Q-Word2Vec es más justa: los dos se guían y se miden con la misma señal.

## Qué es cosine_delta

```
cosine_delta = mean_cosine(pares similares) - mean_cosine(pares disimilares)
```

- **Pares similares**: palabras del mismo clúster semántico (animal, food, culture, sentiment).
- **Pares disimilares**: palabras de clústeres distintos.
- **Rango teórico**: [-2, 2]. Cuanto más alto, mejor agrupación semántica.
- **Funciona igual** para vectores Word2Vec (R²) y distribuciones de probabilidad Q-Word2Vec (R¹⁶).

## Ficheros modificados

### `small_corpora/qword2vec.py` y `small_corpora_v2/qword2vec.py`

| Elemento | Antes | Después |
|---|---|---|
| Variable de control | `best_er = [np.inf]` | `best_cd = [-np.inf]` |
| Condición de mejora | `er < best_er[0] - MIN_DELTA` | `cd > best_cd[0] + MIN_DELTA` |
| Métrica calculada | `calculate_error_rate(probs, label_vectors)` | `evaluate_cosine_delta(word_vectors)` |
| Cabecera del log | `epoch,loss,error_rate` | `epoch,loss,cosine_delta` |
| Rama TRAIN=False | busca mínimo de error_rate | busca máximo de cosine_delta |
| Salida final | `Error rate` + `Top-2 accuracy` | `Cosine delta (mejor época)` + `Cosine delta (evaluación)` |

### `my_tools.py` — `plot_loss_history`

- Variable `error_rates` → `cosine_deltas`
- Eje Y derecho: "Error rate" → "Cosine delta"
- Anotación del mejor punto: `min` → `max`

## Resumen del impacto

```
Antes:
  Word2Vec    → parada: cosine_delta  | evaluación: cosine_delta  ✓
  Q-Word2Vec  → parada: error_rate    | evaluación: cosine_delta  ✗ (desalineado)

Después:
  Word2Vec    → parada: cosine_delta  | evaluación: cosine_delta  ✓
  Q-Word2Vec  → parada: cosine_delta  | evaluación: cosine_delta  ✓ (alineado)
```

El criterio de parada sigue siendo distinto en mecanismo (checkpointing en W2V,
early stopping en QW2V) por razones del optimizador, pero el **criterio** es el mismo.
