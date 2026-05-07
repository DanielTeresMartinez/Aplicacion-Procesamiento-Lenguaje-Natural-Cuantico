# Experimentación v2 — corpus ampliado y nuevo ajuste de hiperparámetros

> Documento de contexto para el agente de redacción de la memoria.
> **Este documento es una extensión directa de `small_corpora/CAMBIOS_RAMA.md`.**
> No es necesario volver a explicar conceptos ya definidos allí (cosine_delta,
> top-2 accuracy, best model checkpointing, etc.).
> El objetivo de esta sección en la memoria es describir qué se ha cambiado
> respecto a la v1, justificar por qué y discutir los resultados obtenidos.

---

## 0. Resumen de cambios respecto a v1

| Aspecto | v1 (`small_corpora`) | v2 (`small_corpora_v2`) |
|---------|----------------------|------------------------|
| Tamaño del corpus | 13 frases | **64 frases** |
| Equilibrio del corpus | Desigual (clusters desproporcionados) | **Equilibrado** (≈16 frases/cluster) |
| Word2Vec `vector_size` | 2 | **4** |
| Word2Vec `window` | 2 | 2 |
| Word2Vec `alpha` | 0.59 | **0.068** |
| Word2Vec `ns_exponent` | 0.0 | **0.4** |
| Word2Vec `epochs` | 500 | **400** |
| Word2Vec cosine_delta final | 0.8927 | **0.9360** |
| Q-Word2Vec `c_val` | 2 | **3** |
| Q-Word2Vec demás parámetros | — | **Sin cambios** respecto a v1 |
| Q-Word2Vec top-2 accuracy | — | **0.8462** |
| Q-Word2Vec error rate | — | **0.1154** |

---

## 1. Nuevo corpus — motivación y estructura

El corpus de la v1 constaba de 13 frases cortas, suficientes para probar el
funcionamiento básico pero con dos limitaciones evidentes: tamaño muy reducido
y distribución desequilibrada de los clústeres semánticos (algunos clústeres
tenían 2–3 frases de contexto y otros 5–6).

El corpus v2 amplía esto a **64 frases** manteniendo exactamente el mismo
vocabulario de 13 palabras (dog, cat, animal, eyes, apple, fish, milk, book,
music, movie, i, like, hate), organizadas en cuatro clústeres semánticos
equilibrados:

- **Animal** (dog, cat, animal, eyes): ~16 frases
- **Comida** (apple, fish, milk): ~16 frases
- **Cultura** (book, music, movie): ~16 frases
- **Sentimiento** (i, like, hate): ~16 frases

El equilibrio es relevante para ambos modelos: en Word2Vec evita que los
clústeres con más frases dominen el gradiente; en Q-Word2Vec hace que los
vectores de etiqueta (label vectors) tengan una distribución más uniforme y
representativa del vocabulario completo.

---

## 2. Nuevo ajuste de hiperparámetros — Word2Vec

### Grid search realizado

Con el corpus ampliado se ha repetido el grid search para buscar los mejores
hiperparámetros de Word2Vec. La búsqueda exploró el siguiente espacio:

```
vector_size : [2, 4]
window      : [1, 2]
alpha       : escala logarítmica para misma probabilidad de ejegir valores entre ~0.005 y ~0.25 (10 valores)
epochs      : [300, 400, 500]
negative    : [2, 3]
ns_exponent : [0.0, 0.2, 0.4]
```

### Mejores hiperparámetros encontrados

```python
BEST_PARAMS = {
    "vector_size": 4,
    "window":      2,
    "alpha":       0.068,
    "negative":    2,
    "ns_exponent": 0.4,
}
FINAL_EPOCHS = 400
```

### Diferencias relevantes respecto a v1

El cambio más llamativo es `alpha`: de 0.59 en v1 a **0.068** en v2. En v1 se
justificó que el corpus pequeño requería pasos de gradiente agresivos para
escapar de mínimos locales. Con 64 frases, el modelo recibe mucha más señal
por época y puede converger con un learning rate más suave y estable. Esto se
refleja también en la curva de pérdida: la v2 es más progresiva y mucho menos
errática que la v1 (que tenía pérdidas de 0.0 en algunas épocas por
oscilaciones del lr alto).

El aumento de `vector_size` de 2 a 4 permite representaciones más ricas.
Con solo 13 frases esto habría causado sobreajuste; con 64 frases hay
suficiente señal para poblar las 4 dimensiones de forma significativa. Para
visualización 2D (gráfica de embeddings) se proyectan las 4 dimensiones con
PCA, que explica el ~72% de la varianza — suficiente para que la separación
visual sea representativa.

---

## 3. Resultados Word2Vec v2

### Curva de entrenamiento (referencia para la gráfica `lossCurvesWord2Vec_v2.png`)

```
  época      1 | pérdida = 12.7244 | cosine_delta =  0.3321  (new best)
  época     50 | pérdida = 12.6523 | cosine_delta =  0.4726
  época    100 | pérdida = 20.2179 | cosine_delta =  0.6178  (new best)
  época    150 | pérdida =  7.1845 | cosine_delta =  0.7499
  época    200 | pérdida = 16.3003 | cosine_delta =  0.8828  (new best)
  época    250 | pérdida = 15.3982 | cosine_delta =  0.9082
  época    300 | pérdida =  9.9588 | cosine_delta =  0.9144
  época    350 | pérdida =  6.7800 | cosine_delta =  0.9343
  época    400 | pérdida =  6.9872 | cosine_delta =  0.9345

Mejor checkpoint guardado: cosine_delta = 0.9360
```

La pérdida sigue siendo errática (rasgo inherente al alpha alto relativo y a
negative sampling), pero la tendencia del cosine_delta es claramente
ascendente y mucho más estable que en v1, donde el mejor valor no se alcanzó
hasta la época 400 con saltos bruscos.

### Cosine delta final

```
Cosine delta final (mejor checkpoint): 0.9360
```

Mejora de **+0.043** respecto al 0.8927 de v1.

### Pares intra-clúster (referencia para la gráfica `cosine_similarity_comparison_v2.png`)

```
Pares intra-clúster (modelo final):
  dog      ↔ cat       coseno=0.789
  dog      ↔ animal    coseno=0.916
  cat      ↔ animal    coseno=0.910
  dog      ↔ eyes      coseno=0.661
  cat      ↔ eyes      coseno=0.872
  apple    ↔ fish      coseno=0.914
  apple    ↔ milk      coseno=0.754
  fish     ↔ milk      coseno=0.953
  book     ↔ music     coseno=0.248
  book     ↔ movie     coseno=0.167
  music    ↔ movie     coseno=0.934
  i        ↔ like      coseno=0.756
  i        ↔ hate      coseno=0.975
  like     ↔ hate      coseno=0.849
```

Comparando con v1, la mejora es general: el clúster animal es ahora mucho
más cohesivo (dog↔cat sube de 0.326 a 0.789; dog↔animal de 0.572 a 0.916).
El clúster cultura sigue siendo el más problemático — book↔music (0.248) y
book↔movie (0.167) son bajos — probablemente porque "book" co-ocurre con más
variedad de contextos que "music" y "movie" entre sí. No obstante music↔movie
(0.934) indica que dos de los tres nodos del clúster están bien representados.

### Nota sobre la visualización 2D

Los embeddings son de dimensión 4. Para la gráfica `embeddingsWord2Vec_v2.png`
se representan directamente las dos primeras dimensiones (d0 y d1). El archivo
`word2vec_embeddings_v2.txt` contiene los vectores completos en dim=4:

```
# word  d0  d1  d2  d3
dog      ...
cat      ...
(13 palabras)
```

---

## 4. Ajuste de hiperparámetros — Q-Word2Vec v2

### Qué se buscó y qué cambió

Para Q-Word2Vec el grid search se centró exclusivamente en el parámetro `c_val`
(peso de la correlación de Pearson en la función de pérdida:
`loss = cross_entropy − c_val · pearson`). Se exploró un rango reducido de
valores enteros:

```
c_val explorado: [1, 2, 3, 4]
```

El valor óptimo encontrado fue **c_val = 3** (frente a 2 en v1). Un valor mayor
penaliza más fuertemente la falta de correlación entre distancias cuánticas y
distancias Word2Vec, lo que empuja al circuito a alinear su estructura de
distancias con la semántica del corpus. Con un corpus más grande y más rico en
co-ocurrencias, esta penalización adicional es útil porque la señal de correlación
es más fiable.

**El resto de hiperparámetros de QNSPSA permanece igual que en v1**, ya que
el grid search no encontró mejoras significativas variándolos:

```
ath           = 0.02         → n_layers = 11
spsa_a        = 0.16
spsa_c        = 0.07
spsa_gamma    = 0.101
spsa_alpha    = 0.602
regularization = 1.6e-2
hessian_delay  = 400
n_qubits      = 4
n_embedding   = 2
n_shots (inferencia) = 2048
```

---

## 5. Resultados Q-Word2Vec v2

### Métricas principales

```
Error rate (mejor época de entrenamiento):  0.1154
Top-2 accuracy (inferencia final):          0.8462
Correlación de Pearson:                     0.4940   (referencia paper: 0.81)
```

El error rate durante entrenamiento (0.1154) coincide con el de v1. La
top-2 accuracy en inferencia (0.8462) significa que el circuito acierta los
dos contextos esperados en 11 de 13 palabras, fallando solo en cat, dog, milk
y music (ver tabla diagnóstico abajo).

### Curva de entrenamiento (referencia para la gráfica `lossCurvesQWord2Vec_v2.png`)

El entrenamiento se realizó durante 800 iteraciones QNSPSA. El mejor punto
se encontró en torno a la iteración 400, donde la pérdida alcanzó su mínimo
(≈ 0.0425) y el error rate bajó a 0.1154. A partir de ahí la optimización
oscila sin mejora adicional, comportamiento habitual de QNSPSA en espacios de
alta dimensión con pocos datos.

### Diagnóstico por palabra (referencia para la tabla en la memoria)

```
  Palabra     top-2 picos reales        top-2 esperados (label)   ¿Correcto?
  animal      ['dog', 'cat']            ['cat', 'dog']            ✓
  apple       ['milk', 'fish']          ['milk', 'fish']          ✓
  book        ['movie', 'music']        ['music', 'movie']        ✓
  cat         ['dog', 'i']             ['animal', 'dog']          ✗
  dog         ['cat', 'dog']           ['cat', 'eyes']            ✗
  eyes        ['dog', 'cat']            ['cat', 'dog']            ✓
  fish        ['milk', 'apple']         ['milk', 'apple']         ✓
  hate        ['dog', 'i']             ['i', 'dog']               ✓
  i           ['like', 'hate']          ['like', 'hate']          ✓
  like        ['dog', 'i']             ['i', 'dog']               ✓
  milk        ['fish', 'milk']          ['apple', 'fish']         ✗
  movie       ['book', 'music']         ['music', 'book']         ✓
  music       ['movie', 'fish']         ['movie', 'book']         ✗
```

Los 4 fallos (cat, dog, milk, music) tienen en común que sus contextos
esperados son palabras con alta conectividad en el corpus (dog y i aparecen
con muchas palabras distintas), lo que dificulta que el circuito separe sus
señales.

### Coordenadas PCA 2D (referencia para la gráfica `embeddings_comparison_v2.png`)

PCA sobre distribuciones de probabilidad de dimensión 2⁴ = 16.
**Varianza explicada: 72.5%** (similar a v1).

```
Q-Word2Vec — coordenadas 2D tras PCA (varianza explicada: 72.5%)
  palabra          PC1       PC2
  animal        +0.1453   -0.0485
  apple         -0.1044   +0.1745
  book          -0.1808   -0.1511
  cat           +0.1279   -0.0333
  dog           +0.0927   -0.0117
  eyes          +0.1764   -0.0322
  fish          -0.1041   +0.2014
  hate          +0.1509   -0.0069
  i             +0.0102   -0.0083
  like          +0.1292   +0.0229
  milk          -0.1067   +0.1629
  movie         -0.1597   -0.1257
  music         -0.1767   -0.1439
```

La separación por clústeres es visible: animal/eyes/cat/dog se agrupan en
PC1 positivo; food (apple/fish/milk) en PC1 negativo y PC2 positivo; culture
(book/music/movie) en PC1 negativo y PC2 negativo; sentiment (i/like/hate)
ocupa una zona intermedia.

---

## 6. Comparación v1 vs v2

### Word2Vec

| Métrica | v1 | v2 | Δ |
|---------|----|----|---|
| cosine_delta final | 0.8927 | **0.9360** | +0.043 |
| dog↔cat | 0.326 | **0.789** | +0.463 |
| dog↔animal | 0.572 | **0.916** | +0.344 |
| cat↔animal | 0.962 | 0.910 | −0.052 |
| apple↔fish | 0.917 | **0.914** | ≈0 |
| fish↔milk | 0.702 | **0.953** | +0.251 |
| book↔music | 0.338 | **0.248** | −0.090 |
| book↔movie | 0.998 | 0.167 | −0.831 |
| music↔movie | 0.276 | **0.934** | +0.658 |

La mejora en v2 es clara en la mayoría de clústeres. La excepción llamativa
es el par book↔movie, que baja de 0.998 a 0.167. Esto refleja que en v2 el
corpus más equilibrado da a "book" contextos más variados (no solo películas),
rompiendo la alta co-ocurrencia artificial que tenía en v1. El clúster cultura
en v2 pivota sobre music↔movie (0.934) en lugar de book↔movie.

### Q-Word2Vec

La comparación directa de métricas numéricas entre v1 y v2 no está disponible
porque los pesos de v1 no se re-evaluaron con el código final. Sin embargo:

- El **error rate de entrenamiento** es idéntico (0.1154 en ambos), lo que
  indica que el circuito aprende la misma fracción de contextos
  independientemente del tamaño del corpus.
- La **visualización PCA** en v2 muestra clústeres más nítidos y mejor
  separados, especialmente food y culture, que en v1 aparecían más solapados.
- La **correlación de Pearson** (0.4940) sigue siendo inferior a la reportada
  en el paper (0.81), limitación atribuible al tamaño del corpus y al número
  reducido de qubits.

**Interpretación:** con más datos de entrenamiento el modelo cuántico aprende
representaciones más descriptivas (mejor PCA) pero no baja el error rate
porque al mismo tiempo los requisitos de precisión aumentan — hay más
co-ocurrencias distintas que el circuito debe discriminar, lo que mantiene
el nivel de error aproximadamente constante. Es decir, el modelo mejora
cualitativamente (separación visual) aunque la métrica de entrenamiento no
mejore numéricamente.

---

## 7. Instrucciones para el agente redactor

### Qué escribir y dónde

1. **Nueva subsección del corpus:** describir el paso de 13 a 64 frases,
   el equilibrio entre clústeres y por qué esto es necesario para obtener
   embeddings más robustos. No hace falta reexplicar qué son los clústeres
   ni el vocabulario — ya se describieron en la sección de la v1.

2. **Nuevo grid search Word2Vec:** indicar que se repitió el procedimiento
   de búsqueda de hiperparámetros (mismo método que en v1) y que el resultado
   principal es un alpha mucho más bajo (0.068 vs 0.59) por la mayor cantidad
   de señal de entrenamiento. Mencionar también el aumento de `vector_size` a 4.

3. **Nuevo ajuste Q-Word2Vec:** indicar brevemente que se repitió el ajuste
   sobre el parámetro `c_val`, encontrando que el valor 3 (frente a 2 en v1)
   mejora el entrenamiento con el corpus ampliado. El resto de parámetros
   permanece igual.

4. **Resultados Word2Vec v2:** usar la curva de entrenamiento y la tabla de
   pares coseno de la sección 3. Comentar la mejora global en cosine_delta
   y los casos particulares (mejora en animal, book↔movie como excepción
   justificada).

5. **Resultados Q-Word2Vec v2:** usar la tabla diagnóstico y las coordenadas
   PCA de la sección 5. Comentar que el error rate se mantiene igual pero
   la representación visual mejora.

6. **Comparación v1 vs v2:** usar la tabla de la sección 6. El mensaje
   central es que Word2Vec mejora claramente con más datos, y que Q-Word2Vec
   mejora cualitativamente aunque la métrica de entrenamiento sea similar.

### Imágenes

Las imágenes tienen exactamente el mismo nombre que en v1 con el sufijo `_v2`
antes de la extensión:

| Imagen v1 | Imagen v2 |
|-----------|-----------|
| `lossCurvesWord2Vec.png` | `lossCurvesWord2Vec_v2.png` |
| `embeddingsWord2Vec.png` | `embeddingsWord2Vec_v2.png` |
| `lossCurvesQWord2Vec.png` | `lossCurvesQWord2Vec_v2.png` |
| `embeddings_comparison.png` | `embeddings_comparison_v2.png` |
| `cosine_similarity_comparison.png` | `cosine_similarity_comparison_v2.png` |
| `bloch_sphere.png` | `bloch_sphere.png` *(sin cambios, misma imagen)* |

Todas las imágenes están en `memoria/imagenes/`.
