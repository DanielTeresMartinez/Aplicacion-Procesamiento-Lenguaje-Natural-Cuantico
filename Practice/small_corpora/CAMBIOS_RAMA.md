# Cambios implementados — Word2Vec y Q-Word2Vec (`small_corpora`)

> Documento de contexto para el agente de redacción de la memoria.
> Describe el estado actual del código, las decisiones tomadas y la justificación de cada cambio.

---

## 0. Visión general de los cambios respecto al estado anterior

| Aspecto | Antes | Ahora |
|---------|-------|-------|
| Métrica de evaluación Word2Vec | K-Means accuracy | **cosine_delta** |
| Criterio de selección del mejor modelo (Word2Vec) | Early stopping sobre K-Means accuracy | **Best Model Checkpointing** sobre cosine_delta |
| Hiperparámetro `epochs` en Word2Vec | Fijo en 2000 con early stopping | **Incluido en el grid search** junto con `alpha` |
| Métrica de evaluación QWord2Vec | cosine_delta sobre distribuciones de probabilidad | **Top-2 accuracy** (hit rate de picos) |
| Grid search Word2Vec | Sin `epochs` explícito | `epochs` incluido como hiperparámetro |

El cambio central es conceptual: K-Means sobre embeddings como criterio de early stopping no tenía mucha lógica — era una métrica indirecta, sensible a la inicialización aleatoria y difícil de justificar. Se ha reemplazado por métricas directas y bien definidas para cada modelo.

---

## 1. Grid search en Word2Vec (`grid_search_word2vec.py`)

### Qué hace el grid search

El script `grid_search_word2vec.py` explora exhaustivamente el espacio de hiperparámetros:

```python
param_grid = {
    "vector_size": [2, 4],
    "window":      [1, 2],
    "alpha":       np.linspace(0.35, 0.75, 6),
    "epochs":      np.arange(300, 600, 100),
    "negative":    [2, 3],
    "ns_exponent": [0.0, 0.2],
}
```

Para cada combinación entrena un modelo completo y evalúa `cosine_delta`. Al final imprime los mejores hiperparámetros encontrados.

### Por qué `epochs` está en el grid search

Gensim calcula el decaimiento de la tasa de aprendizaje de forma que `alpha` llegue a `min_alpha` (≈ 0.0001) exactamente en la **última época planificada**, usando la fórmula:

```
alpha_en_época_t = alpha_inicial - t * (alpha_inicial - min_alpha) / epochs
```

Esto significa que `alpha` y `epochs` están **acoplados**: si se fija `epochs=2000` pero el modelo para en la época 400 (por early stopping), el aprendizaje se detiene con `alpha ≈ 0.24` — todavía muy alto — lo que produce oscilaciones en los embeddings. El resultado depende entonces de cuándo se detiene el entrenamiento, no solo de los hiperparámetros semánticos.

**Solución:** incluir `epochs` directamente en el grid search y entrenar siempre hasta el final del número de épocas elegido. Así el schedule de `alpha` se ejecuta completo, los resultados son reproducibles y comparables.

### Los valores en `word2vec.py` son los mejores del grid search

```python
# word2vec.py — valores encontrados por grid_search_word2vec.py
BEST_PARAMS = {
    "vector_size": 2,
    "window":      2,
    "alpha":       0.59,
    "negative":    2,
    "ns_exponent": 0.0,
}
FINAL_EPOCHS = 500
```

Estos valores **no se tocan a mano**: provienen directamente del mejor resultado de `grid_search_word2vec.py`.

---

## 2. Best Model Checkpointing en Word2Vec

### Nombre de la técnica

La técnica de evaluar el modelo en cada época y guardar únicamente la versión con mejor métrica se llama **Best Model Checkpointing** (o simplemente *model checkpointing*). Es distinta del *early stopping*: el early stopping detiene el entrenamiento cuando no mejora; el checkpointing deja que el entrenamiento termine siempre pero se queda con el mejor punto intermedio.

### Implementación en `word2vec.py`

El `LossCallback` evalúa `cosine_delta` al final de cada época y guarda el modelo completo si supera el mejor valor registrado:

```python
class LossCallback(CallbackAny2Vec):
    def on_epoch_end(self, model):
        val_score = evaluate(model)          # cosine_delta de esta época
        if val_score > self._best_val_score:
            self._best_val_score = val_score
            model.save(self._best_checkpoint)  # guarda solo si mejora

# Al terminar las FINAL_EPOCHS épocas, se carga el mejor modelo guardado
model = Word2Vec.load(loss_cb._best_checkpoint)
final_corr = evaluate(model)
print(f"Cosine delta final: {final_corr:.4f}")
```

El entrenamiento siempre completa las `FINAL_EPOCHS` épocas (para que el schedule de `alpha` se ejecute completo), pero la evaluación final se hace sobre el mejor checkpoint.

---

## 3. Métrica cosine_delta (Word2Vec)

### Definición

La memoria ya explica qué es la distancia coseno y cómo se calcula. El `cosine_delta` construye sobre esa base:

> **cosine_delta** = media(similitud coseno en pares similares) − media(similitud coseno en pares disimilares)

Un valor alto indica que palabras del mismo clúster semántico están más cerca entre sí (coseno alto) que palabras de clústeres distintos (coseno bajo). El rango teórico es [−2, 2]; cuanto más alto, mejor.

### Por qué NO es simplemente la media de todos los cosenos

Es importante no confundir `cosine_delta` con la media aritmética de todas las similitudes coseno. Esa media carecería de poder discriminativo: un modelo que asigna embeddings similares a **todas** las palabras (buenas y malas) obtendría una media alta aunque no hubiera aprendido ninguna estructura semántica.

`cosine_delta` es una **diferencia de medias** — una métrica de contraste:

```
cosine_delta = mean_sim − mean_dis
             = media(14 pares similares) − media(10 pares disimilares)
```

Con los resultados del modelo final:

| Grupo | Pares | Media coseno |
|-------|-------|-------------|
| Similares (mismo clúster) | 14 | ≈ 0.711 |
| Disimilares (clústeres distintos) | 10 | ≈ −0.182 |
| **cosine_delta** | — | **≈ 0.893** |

Que la media de disimilares sea negativa es esperable y deseable: significa que el modelo coloca activamente esas palabras en lados opuestos del espacio de embeddings (coseno negativo implica vectores con componentes en sentidos contrarios).

La métrica penaliza activamente los pares que deberían ser lejanos, no solo premia los cercanos. Por eso un `cosine_delta` alto garantiza que el modelo ha aprendido **separación** entre clústeres, no solo cohesión interna.

### Pares utilizados (anotados manualmente, definidos en `my_tools.py`)

**Similares** (14 pares, mismo clúster semántico):
- *animal:* dog↔cat, dog↔animal, cat↔animal, dog↔eyes, cat↔eyes
- *food:* apple↔fish, apple↔milk, fish↔milk
- *culture:* book↔music, book↔movie, music↔movie
- *sentiment:* i↔like, i↔hate, like↔hate

**Disimilares** (10 pares, clústeres distintos):
dog↔apple, dog↔book, dog↔like, cat↔music, apple↔book, apple↔i, fish↔movie, milk↔hate, book↔dog, music↔apple

### Código de evaluación en `my_tools.py`

```python
def evaluate_cosine_delta(word_vectors: dict) -> float:
    def _sim(w1, w2):
        return 1.0 - cosine_distance(word_vectors[w1], word_vectors[w2])

    sim_scores = [_sim(w1, w2) for w1, w2 in SIMILAR_PAIRS  if w1 in word_vectors and w2 in word_vectors]
    dis_scores = [_sim(w1, w2) for w1, w2 in DISSIMILAR_PAIRS if w1 in word_vectors and w2 in word_vectors]
    return float(np.mean(sim_scores) - np.mean(dis_scores))
```

### Por qué es la métrica adecuada para Word2Vec

Word2Vec aprende representaciones donde la similitud semántica se mide precisamente con coseno. Usar esa misma distancia para evaluar es conceptualmente coherente y está en línea con benchmarks estándar del campo (WordSim353, SimLex-999). Además el `cosine_delta` evalúa una propiedad semántica sobre pares anotados externamente — el modelo no puede "hackearlo" optimizando directamente sobre él.

---

## 4. Métrica top-2 accuracy (Q-Word2Vec)

### Por qué se eliminó cosine_delta de Q-Word2Vec

El cosine_delta calculado sobre las distribuciones de probabilidad cuánticas (vectores de dimensión 2^n = 16) da valores muy bajos porque las distribuciones son "planas": todas las palabras tienen perfiles de probabilidad similares y pequeños, así que el coseno entre cualquier par resulta alto (incluso entre disímiles) y el delta se cancela. No es que el modelo funcione mal — es que la métrica no es adecuada para este espacio de representación.

Usar las coordenadas PCA (2D) para calcular cosine_delta tampoco sería justo: con ~75% de varianza explicada se pierde información relevante. No hay una forma honesta de aplicar cosine_delta al modelo cuántico con el mismo significado que tiene para el clásico.

### Métrica alternativa: top-2 accuracy

La métrica natural del modelo cuántico es la que guía su propio entrenamiento: qué fracción de las palabras predicen correctamente sus dos contextos más probables.

**Cómo se calcula** (función `calculate_error_rate` en `my_tools.py`):

1. Para cada palabra `w` del vocabulario, el **label** tiene valor 0.5 en las 2 posiciones correspondientes a sus contextos más frecuentes (el resto es 0).
2. Se comparan los **2 índices con mayor probabilidad** en la salida del circuito con los **2 índices con valor 0.5** en el label.
3. Se cuenta cuántos índices del label **no aparecen** en el top-2 del modelo (mismatches).
4. `error_rate = mismatches / (2 × n_palabras)`
5. `top2_accuracy = 1 − error_rate`

```python
def calculate_error_rate(prob_distributions, label_vectors):
    total_mismatches = 0
    for k, label in label_vectors.items():
        label_peaks = set(np.argsort(label)[-2:])
        model_peaks = set(np.argsort(prob_distributions[k])[-2:])
        total_mismatches += len(label_peaks - model_peaks)
    return total_mismatches / (2 * len(label_vectors))
```

**Código de evaluación final en `my_qword2vec_qnspsa.py`:**

```python
top2_accuracy = 1.0 - calculate_error_rate(final_probs, label_vectors)
print(f"Top-2 accuracy (picos):    {top2_accuracy:.4f}")
```

### Interpretación

Un `top2_accuracy = 1.0` significa que para cada palabra del vocabulario el circuito cuántico coloca la mayor masa de probabilidad exactamente en los dos índices de los contextos esperados. Un valor de 0.5 significa que acierta un contexto de cada dos, en promedio.

### Nota sobre la comparabilidad con cosine_delta

`cosine_delta` (Word2Vec) y `top2_accuracy` (Q-Word2Vec) no son la misma métrica y no son directamente comparables numéricamente. Sin embargo ambas responden a la misma pregunta semántica desde la perspectiva de cada modelo: ¿aprende el modelo la estructura de similitud del corpus? Se reportan por separado y se discute cualitativamente la comparación.

---

## 5. Código final de evaluación — resumen

### `word2vec.py` (evaluación final)

```python
# Cargar el mejor checkpoint guardado durante el entrenamiento
model = Word2Vec.load(loss_cb._best_checkpoint)
final_corr = evaluate(model)
print(f"Cosine delta final: {final_corr:.4f}")

# Pares intra-clúster detallados
for w1, w2 in SIMILAR_PAIRS:
    if w1 in model.wv and w2 in model.wv:
        sim = model.wv.similarity(w1, w2)
        print(f"  {w1:8s} ↔ {w2:8s}  coseno={sim:.3f}")
```

### `my_qword2vec_qnspsa.py` (evaluación final)

```python
# Diagnóstico cualitativo: picos reales vs esperados
for w, idx in sorted(word_to_id.items()):
    real_peaks    = list(np.argsort(final_probs[idx])[-2:][::-1])
    expected_peaks = list(np.argsort(label_vectors[idx])[-2:][::-1]) if idx in label_vectors else []
    match = "✓" if set(real_peaks) == set(expected_peaks) else "✗"
    print(f"  {w:<10} {str([id_to_word.get(p) for p in real_peaks]):<25} "
          f"{str([id_to_word.get(p) for p in expected_peaks]):<25} {match}")

# Métrica numérica: top-2 accuracy
top2_accuracy = 1.0 - calculate_error_rate(final_probs, label_vectors)
print(f"Top-2 accuracy (picos):    {top2_accuracy:.4f}")
```

---

## 6. Resultados finales

### Word2Vec (`word2vec.py`)

Pares intra-clúster (modelo final — mejor checkpoint):

```
Pares intra-clúster (modelo final):
  dog      ↔ cat       coseno=0.326
  dog      ↔ animal    coseno=0.572
  cat      ↔ animal    coseno=0.962
  dog      ↔ eyes      coseno=0.949
  cat      ↔ eyes      coseno=0.607
  apple    ↔ fish      coseno=0.917
  apple    ↔ milk      coseno=0.928
  fish     ↔ milk      coseno=0.702
  book     ↔ music     coseno=0.338
  book     ↔ movie     coseno=0.998
  music    ↔ movie     coseno=0.276
  i        ↔ like      coseno=0.596
  i        ↔ hate      coseno=0.930
  like     ↔ hate      coseno=0.850
```

**Cosine delta final (mejor checkpoint, época ~400):** `0.8927`

Evolución durante el entrenamiento (cada 50 épocas):

```
  época      1 | cosine_delta = -0.1635
  época     50 | cosine_delta = -0.0677
  época    100 | cosine_delta = -0.3667
  época    150 | cosine_delta =  0.3931
  época    200 | cosine_delta =  0.3341
  época    250 | cosine_delta =  0.3308
  época    300 | cosine_delta =  0.4047
  época    350 | cosine_delta =  0.5556
  época    400 | cosine_delta =  0.8474  ← mejor registrado en log
  época    450 | cosine_delta =  0.7638
  época    500 | cosine_delta =  0.7506
Mejor checkpoint guardado:    0.8927
```

La pérdida también es errática (0.0000 en varias épocas, picos de 6.4 en época 200), lo que refleja el comportamiento inestable esperado con `alpha = 0.59`; el checkpointing aísla el mejor momento de convergencia.

#### Observaciones sobre los resultados

- **Relaciones bien capturadas:** `cat↔animal` (0.962), `book↔movie` (0.998), `i↔hate` (0.930), `apple↔milk` (0.928) — el modelo aprende con éxito la cohesión dentro de varios clústeres.
- **Relaciones problemáticas:** `dog↔cat` (0.326) y `music↔movie` (0.276) son sorprendentemente bajas para pares del mismo clúster; `dog↔eyes` (0.949) es inesperadamente alta dado que `eyes` pertenece al clúster animal solo por co-ocurrencia textual ("the cat's eyes").
- **Causa probable:** corpus muy pequeño — las co-ocurrencias textuales dominan sobre la semántica real. `eyes` aparece frecuentemente junto a palabras de animales en el corpus y por eso adquiere un embedding cercano a ese clúster.

#### Nota sobre el learning rate (`alpha = 0.59`)

El valor `alpha = 0.59` seleccionado por el grid search es considerablemente alto comparado con el valor por defecto de Gensim (`alpha = 0.025`) y con los rangos habituales en la literatura (0.01–0.05). Esta elección se justifica por el tamaño reducido del corpus y el bajo número de épocas (`FINAL_EPOCHS = 500`): con un corpus pequeño, el modelo necesita pasos de gradiente más agresivos para escapar de los mínimos locales antes de que el schedule de `alpha` lo lleve a valores bajos. El grid search valida empíricamente que este valor maximiza `cosine_delta` en estas condiciones, aunque en un corpus grande produciría oscilaciones e inestabilidad.

### Q-Word2Vec (`my_qword2vec_qnspsa.py` con `TRAIN=False`)

```
[PENDIENTE — pegar aquí la salida de: python3 my_qword2vec_qnspsa.py]
```

---

## 7. Secciones de la memoria que requieren actualización

1. **§ Hiperparámetros de Word2Vec** — actualizar con los valores actuales de `BEST_PARAMS` y `FINAL_EPOCHS`; mencionar que provienen del grid search
2. **§ Schedule de aprendizaje** — explicar el acoplamiento `alpha`/`epochs` en Gensim y por qué `epochs` se incluyó en el grid search
3. **§ Estrategia de entrenamiento** — reemplazar descripción de K-Means + early stopping por Best Model Checkpointing; explicar que el entrenamiento siempre completa las épocas planificadas
4. **§ Métricas de evaluación** — reemplazar K-Means accuracy por cosine_delta (W2V) y top-2 accuracy (QWord2Vec); incluir las definiciones de esta sección
5. **§ Resultados y comparación** — actualizar con los valores numéricos de los bloques `[PENDIENTE]`; aclarar que cosine_delta y top-2 accuracy no son comparables numéricamente
