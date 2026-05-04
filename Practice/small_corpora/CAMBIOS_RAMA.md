# Cambios realizados en la rama `cosine-eval` — Word2Vec y QWord2Vec

> Este documento describe los cambios implementados respecto a la rama principal y el razonamiento detrás de cada decisión. Está preparado para ser usado como contexto por el agente de redacción de la memoria.

---

## 1. Reorganización del directorio

El directorio `Practice/` se ha dividido en dos subdirectorios:

- `small_corpora/` → experimentos con el corpus pequeño original (13 frases, 13 palabras)
- `small_corpora_v2/` → experimentos con el corpus alternativo (corpus más grande)

**Razón:** separar claramente los dos conjuntos de experimentos independientes para evitar confusión entre ficheros con y sin sufijo `_v2`.

---

## 2. Cambio de métrica de evaluación en Word2Vec

### Qué se ha cambiado

**Antes:** la métrica de evaluación y criterio de *early stopping* era la **K-Means accuracy** calculada sobre todos los embeddings del vocabulario. Se asignaba cada embedding a un clúster mediante K-Means (k=4) y se comparaba con los clústeres de referencia usando el algoritmo húngaro.

**Ahora:** la métrica es el **cosine_delta**, definido como:

```
cosine_delta = media(cosine_sim(pares similares)) − media(cosine_sim(pares disimilares))
```

donde los pares están anotados manualmente según la pertenencia semántica a los 4 clústeres del corpus.

### Pares utilizados

**Similares** (mismo clúster, similitud esperada alta):
- Animal: dog↔cat, dog↔animal, cat↔animal, dog↔eyes, cat↔eyes
- Food: apple↔fish, apple↔milk, fish↔milk
- Culture: book↔music, book↔movie, music↔movie
- Sentiment: i↔like, i↔hate, like↔hate

**Disimilares** (distinto clúster, similitud esperada baja):
- dog↔apple, dog↔book, dog↔like, cat↔music, apple↔book, apple↔i, fish↔movie, milk↔hate, book↔dog, music↔apple

### Razón del cambio

K-Means como métrica de evaluación tiene varias limitaciones:

1. **Indirección**: K-Means no mide directamente la similitud semántica, sino si los vectores se agrupan geométricamente de una forma que coincida con los clústeres de referencia. Puede dar resultados engañosos con vocabularios pequeños.
2. **Evaluación sobre train**: al evaluar sobre el mismo conjunto de entrenamiento y con los mismos clústeres usados para guiar el early stopping, existe una dependencia circular que dificulta la interpretación.
3. **Défense académica**: el cosine_delta es más directo y más fácilmente justificable como métrica de calidad semántica: mide exactamente si las palabras relacionadas están más cerca entre sí (coseno alto) que las no relacionadas (coseno bajo) en el espacio de embeddings.

El cosine_delta es coherente con la motivación de Word2Vec: el modelo aprende representaciones donde la cercanía semántica se mide con similitud coseno. Usar esa misma distancia para evaluarlo es conceptualmente correcto y está en línea con benchmarks estándar del campo (WordSim353, SimLex-999).

### Diferencia con overfitting

El cosine_delta **no** mide el ajuste a los datos de entrenamiento. El modelo se entrena para minimizar la pérdida de *Negative Sampling* (una tarea de predicción de contextos), mientras que cosine_delta evalúa una propiedad semántica sobre un conjunto de pares anotados externamente. El modelo no puede "hackear" cosine_delta optimizando directamente sobre él.

### Impacto en el early stopping

- El entrenamiento sigue monitorizando la pérdida de *Negative Sampling* como señal de optimización.
- El early stopping ahora maximiza cosine_delta (en lugar de K-Means accuracy) cada 200 épocas con PATIENCE=5 y MIN_DELTA=0.01.
- El comportamiento es análogo: se busca el punto en que los embeddings tienen mayor coherencia semántica, no donde el loss es mínimo.

### Nuevos hiperparámetros encontrados

Se ha re-ejecutado el grid search (`grid_search_word2vec.py`) con la nueva métrica. Los mejores hiperparámetros encontrados son:

| Parámetro | Valor anterior | Valor nuevo |
|-----------|---------------|-------------|
| vector_size | 2 | 2 |
| window | 1 | **2** |
| alpha | 0.25 | **0.3** |
| negative | 5 | **2** |
| ns_exponent | 0.0 | 0.0 |

**cosine_delta obtenido en grid search: 0.5588**

> **NOTA PARA EL AGENTE DE REDACCIÓN:** Los resultados finales del entrenamiento de Word2Vec (salida del terminal) se añadirán a continuación cuando el usuario los proporcione. Actualizar con esos valores la sección de resultados del capítulo de experimentación.

### Resultados del entrenamiento Word2Vec

```
[PENDIENTE — pegar aquí la salida del terminal de python word2vec.py]
```

---

## 3. Cambio en el grid search de Word2Vec

**Fichero:** `grid_search_word2vec.py`

El grid search ahora:
- Importa `evaluate` directamente de `word2vec.py`
- **Maximiza** cosine_delta (en lugar de minimizar error_rate)
- Imprime `cosine_delta` en lugar de `spearman_corr`

Se ha eliminado la dependencia de `most_similar_error_rate` (función que ya no existe en word2vec.py).

---

## 4. Adición de cosine_delta a QWord2Vec

### Qué se ha cambiado

**Fichero:** `my_qword2vec_qnspsa.py`

Se han añadido al inicio del fichero:
- `_SIMILAR_PAIRS` y `_DISSIMILAR_PAIRS` — exactamente los mismos pares que en `word2vec.py`
- `cosine_sim()` — función auxiliar de similitud coseno entre vectores numpy
- `evaluate_cosine_delta()` — calcula cosine_delta sobre las distribuciones de probabilidad del circuito cuántico

Al final del script (sección de evaluación final), se imprime:
```
Cosine delta (≈W2V):   X.XXXX
```

### Razón del cambio

La comparación final entre Word2Vec y QWord2Vec en la memoria utiliza varias métricas:
- Correlación de Pearson (ya existía)
- K-Means accuracy (ya existía en QWord2Vec)
- **cosine_delta (nuevo)** — permite comparar ambos modelos con la misma métrica numérica de forma directa y sin ambigüedad

### Qué NO se ha cambiado en QWord2Vec

- El **early stopping** sigue basándose en `error_rate` (tasa de error en la predicción de contextos). Esto es correcto porque el circuito se entrena para aproximar distribuciones de probabilidad específicas, y el error_rate mide directamente esa tarea.
- La **función de pérdida** (entropía cruzada + correlación de Pearson) no ha cambiado.
- El cosine_delta en QWord2Vec se calcula sobre las distribuciones de probabilidad medidas (vectores de dimensión 2^n = 16), no sobre embeddings 2D como en Word2Vec. Ambas son representaciones válidas del espacio semántico aprendido por cada modelo.

### Interpretación

El cosine_delta sobre las distribuciones cuánticas mide si las distribuciones de probabilidad de palabras semánticamente similares son más parecidas (coseno alto) que las de palabras sin relación (coseno bajo). Es la traducción directa de la misma pregunta semántica al espacio de Hilbert.

### Resultados de QWord2Vec (cargando pesos)

```
[PENDIENTE — pegar aquí la salida del terminal de python my_qword2vec_qnspsa.py con TRAIN=False]
```

---

## 5. Eliminación de ficheros de validación y test

Los ficheros `smallCorpora_val.txt` y `smallCorpora_test.txt` han sido eliminados.

**Razón:** Word2Vec es un modelo inductivo en vocabulario. No puede generar embeddings para palabras no vistas, y todas las palabras del vocabulario aparecen en el corpus de entrenamiento. Los ficheros de validación/test no aportaban valor diferencial:
- Las palabras de val/test son las mismas que las de train (mismo vocabulario de 13 palabras)
- La evaluación con cosine_delta usa pares anotados manualmente, independientes del corpus
- Mantener los ficheros generaba confusión sobre su rol real

El entrenamiento se realiza con el **100% del corpus** (13 frases), igual que antes y justificado en la memoria.

---

## Resumen de cambios por fichero

| Fichero | Cambio |
|---------|--------|
| `word2vec.py` | K-Means → cosine_delta; nuevos BEST_PARAMS; eliminada carga de val/test |
| `grid_search_word2vec.py` | Métrica: error_rate → cosine_delta; eliminada dependencia de `most_similar_error_rate` |
| `my_qword2vec_qnspsa.py` | Añadidos `_SIMILAR_PAIRS`, `_DISSIMILAR_PAIRS`, `evaluate_cosine_delta()`; nuevo print de cosine_delta al final |
| `smallCorpora_val.txt` | Eliminado |
| `smallCorpora_test.txt` | Eliminado |
| `smallCorpora.txt` | Sin cambios (13 frases originales) |

---

## Secciones de la memoria que requieren actualización

1. **§ Hiperparámetros de Word2Vec** — actualizar los valores de `window` (1→2), `alpha` (0.25→0.3), `negative` (5→2)
2. **§ Entrenamiento de Word2Vec** — reemplazar la descripción de K-Means accuracy por cosine_delta como métrica de early stopping; actualizar gráficas y valores numéricos con los resultados del terminal
3. **§ Correlación de Pearson y precisión K-Means** — añadir cosine_delta como tercera métrica de comparación; actualizar valores numéricos
4. **§ Discusión** — adaptar el argumento de diferencia entre modelos incorporando cosine_delta como métrica común

> **NOTA IMPORTANTE:** Los valores numéricos concretos (cosine_delta de Word2Vec, cosine_delta de QWord2Vec, K-Means accuracy actualizada) deben tomarse de los resultados del terminal que el usuario proporcionará. Los placeholders `[PENDIENTE]` deben sustituirse por esos valores antes de redactar.
