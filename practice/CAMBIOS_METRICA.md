# Experimento: uso de cosine_delta como métrica de entrenamiento en Q-Word2Vec

## Motivación

Se intentó sustituir **error_rate** por **cosine_delta** en el early stopping de Q-Word2Vec,
para que el criterio de parada y la evaluación final usaran la misma métrica y la
comparación con Word2Vec fuera más directa.

## Resultados del grid search con cosine_delta como objetivo

| Versión | Mejor época | Mejor cosine_delta |
|---|---|---|
| V1 (13 frases) | 502 | **0.1360** |
| V2 (64 frases) | 800 (límite) | **0.3923** |

Comparación con Word2Vec:

| Versión | Word2Vec cosine_delta | Q-Word2Vec cosine_delta |
|---|---|---|
| V1 | ~0.8927 | 0.1360 |
| V2 | ~0.9360 | 0.3923 |

## Por qué se descartó

Los valores obtenidos (0.14 y 0.39) son muy inferiores a los de Word2Vec (~0.89 y ~0.94),
lo que se traduce en una representación visual de embeddings muy pobre: el circuito
cuántico no logra separar los clústeres semánticos cuando se entrena optimizando cosine_delta.

## Estado actual

Se volvió al esquema original con error_rate para el early stopping. El cosine_delta
se sigue calculando y mostrando como métrica adicional en la evaluación final del modelo.
