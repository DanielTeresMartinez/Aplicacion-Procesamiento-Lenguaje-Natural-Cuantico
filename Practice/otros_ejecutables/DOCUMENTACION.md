# Documentación de la práctica: Q-Word2Vec

Implementación del modelo Q-Word2Vec basada en las secciones **3.1 a 3.4** del
artículo *"Q-word2vec: Quantum Natural Language Processing"* (Ohno et al., Quantum
Machine Intelligence, 2025).

Archivos involucrados: `word2Vec.py`, `qWord2Vec.py` y `qvisualization.py`.

---

## Corpus de entrenamiento

**`smallCorpora.txt`** Es el corpus compartido por los dos modelos.

**`smallWordList.txt`** es una lista plana de palabras (una por línea). El paper la
menciona como parte de los datos de entrenamiento ("sentence and word lists") pero no
especifica cómo se usa. En el código se trata como una secuencia de entrenamiento
adicional (no como un vocabulario de referencia), lo que genera pares Skip-Gram según
la proximidad de las palabras en el archivo. Esto crea co-ocurrencias arbitrarias
basadas en el orden en que están escritas las palabras, no en relaciones semánticas
reales.

---

## `word2Vec.py` — Word2Vec clásico (referencia)

Entrena un modelo Word2Vec clásico (Skip-Gram con negative sampling) usando **gensim**.
Su objetivo es generar vectores de palabras semánticamente coherentes sobre el corpus
pequeño que sirvan de **ground truth** para evaluar el modelo cuántico.

El **flujo** es: cargar `smallCorpora.txt` → dividir 80/20 entrenamiento/validación →
entrenar hasta 2000 épocas con parada anticipada (early stopping) basada en
*silhouette score* → guardar los mejores embeddings.

La clase `LossCallback` es el núcleo del control de entrenamiento: al final de cada
época registra el delta de pérdida y, cada `PRINT_EVERY` épocas, calcula el silhouette
score sobre clusters semánticos predefinidos (`_clusters`). Si el score no mejora
durante `PATIENCE` checkpoints consecutivos lanza `EarlyStopping` y el modelo guardado
en ese punto se usa como resultado final.

### Salida: `word2vec_embeddings.txt`

Fichero de texto con una palabra por línea:

    # word  d0  d1  # vector_size=2
    dog  0.12345678  -0.87654321
    cat  ...

Los vectores son de **dimensión 2** para poder visualizarlos directamente. Este fichero
es la **entrada principal** de `qWord2Vec.py`: sin él el modelo cuántico no puede
construir las distancias objetivo ni evaluar la calidad de sus embeddings.

---

## `qWord2Vec.py` — Q-Word2Vec cuántico

Implementa el modelo Q-Word2Vec. La idea central es sustituir las matrices de embedding
de Word2Vec por dos circuitos cuánticos parametrizados `U(θ_u)` y `V(θ_v)` (figura 1 del paper),
entrenarlos para que su salida reproduzca la estructura semántica que Word2Vec ha aprendido, y
medir cuánto se parecen los dos espacios de embeddings.

El código está organizado siguiendo el orden de las secciones del paper. El bloque
`__main__` ejecuta el siguiente orden: datos → §3.4 (profundidad) → §3.1
(circuito) → §3.3 + §4 (entrenamiento) → §3.2 (evaluación).

*Nota*: §X.Y significa que es la sección X.Y del paper en el que me he basado.

La variable `SHOW_VISUALIZATIONS` al inicio del `__main__` controla si se generan las
imágenes al final de la ejecución o no. 

---

### Datos y corpus

`load_corpus` lee `smallCorpora.txt` y devuelve una lista de frases tokenizadas.
`load_word_list` lee `smallWordList.txt` y devuelve todas las palabras como una única
secuencia (una sola "frase"). Al concatenarse al corpus, la ventana deslizante genera
pares Skip-Gram entre palabras adyacentes en la lista, lo cual no representa
co-ocurrencias semánticas reales sino el orden arbitrario del archivo.

`build_vocabulary` toma el corpus completo y asigna un índice entero a cada palabra
única. El vocabulario está limitado a `2^n_qubits` palabras porque el espacio de
Hilbert de `n` qubits tiene exactamente ese número de estados base, y cada palabra
ocupa uno.

`generate_training_data_from_text` genera pares Skip-Gram con ventana deslizante: para
cada palabra recoge las palabras que aparecen en las `window_size` posiciones a su
izquierda y derecha. Con `window_size=1` se obtienen exactamente los dos vecinos
inmediatos, que es lo que el paper llama *two surrounding words*.

`load_word2vec_embeddings` carga `word2vec_embeddings.txt` y devuelve un diccionario
`{palabra: vector}`.

---

### Vectores de etiqueta (Sección 4)

`generate_label_vectors` convierte los pares Skip-Gram en vectores de tamaño
`N = 2^n_qubits`. Para cada palabra objetivo, las dos palabras de contexto más
frecuentes reciben el valor **0.5** y el resto **0**, de modo que el vector suma 1.
Este formato de etiqueta es el que describe la Sección 4 del paper y es lo que el
circuito cuántico tiene que aprender a reproducir.

---

### Estimación de la profundidad del circuito (Sección 3.4)

En lugar de buscar manualmente la profundidad óptima `L`, el paper propone una fórmula
heurística que la estima a partir del tamaño del vocabulario, el número de qubits de
embedding y el número de pares de entrenamiento.

`binary_entropy` calcula `H(p) = -p·log₂(p) - (1-p)·log₂(1-p)`.

`get_error_probability` resuelve la ecuación `ne = (1 − H(p))·2^n` para encontrar el
valor de `p` correspondiente a los parámetros `n` y `ne`. Como `H(p)` no tiene inversa
analítica cerrada, se usa búsqueda binaria en el intervalo `[0, 0.5]`.

`estimate_num_layers` aplica la fórmula de la ec. 6:
`L = ⌈num_data / (3·(n+ne)·Ath)⌉ · p(n,ne)`. El factor 3 corresponde a las tres
rotaciones de Pauli por capa. `Ath` es un umbral empírico (0.02 en los experimentos)
que controla cuántos datos por parámetro se toleran.

---

### Construcción del circuito cuántico (Sección 3.1)

La arquitectura es `|k⟩ → U(θ_u) → reset(qubits ne..n-1) → V(θ_v) → |ψ⟩`, tal como
muestra la Fig. 1 del paper.

`get_entangling_layer` construye la capa `U_enta` con puertas CX en configuración
*circuit-block* (Sim et al. 2019). Esta configuración se eligió en el paper por su
alta expresibilidad y capacidad de entrelazamiento. La puerta CX se prefirió sobre
CZ por resultados preliminares.

`build_parameterized_block` construye un bloque `U(θ)` o `V(θ)` con `L` capas. Cada
capa aplica `RX^⊗n → RY^⊗n → RZ^⊗n → U_enta` (ec. 4). Se usan las tres rotaciones de
Pauli para que las palabras puedan distribuirse por toda la esfera de Bloch en lugar
de quedar restringidas a una circunferencia.

`build_full_qword2vec_circuit` ensambla el circuito completo. Después de `U`, los
qubits `ne..n-1` se reinician a `|0⟩` de modo que solo los `ne` qubits de embedding
llevan información de `U` hacia `V`. Esto actúa como un cuello de botella análogo al
de Word2Vec, forzando a que los qubits de embedding capturen lo esencial.

`prepare_transpiled_circuits` pre-compila los `2^n` circuitos, uno por cada posible
palabra de entrada `|k⟩`. La codificación es: el estado `|k⟩` se prepara aplicando
puertas X en los qubits cuyo bit correspondiente en la representación binaria de `k`
es 1. Pre-transpilar evita repetir este coste en cada evaluación de la función de
pérdida durante el entrenamiento.

---

### Función de pérdida (Sección 3.3)

La pérdida combina dos términos (ec. 5): `Loss = CrossEntropy − C · PearsonR(d_Q, d_W2V)`.

`get_output_probabilities_batched` ejecuta el circuito completo para cada palabra y
devuelve la distribución de probabilidad sobre los `N` estados base:
`prob[k][j] = |⟨j|V·U|k⟩|²`. Esta es la salida que se compara con los vectores de
etiqueta durante el entrenamiento.

`calculate_cross_entropy` calcula `H(y, p) = −Σ_j y_k[j]·log(p_k[j])` promediado
sobre las palabras del vocabulario. Penaliza que la distribución de salida del
circuito no coincida con el vector de etiqueta.

`calculate_custom_loss` calcula la pérdida total. Además de la cross-entropy, añade
el término de correlación de Pearson entre las distancias euclídeas de las
distribuciones cuánticas y las distancias de Word2Vec. Este segundo término actúa como
regularización geométrica: penaliza que el circuito aprenda a predecir contextos
correctamente pero con una geometría incompatible con la de Word2Vec. El parámetro
`C` balancea ambos términos.

`build_target_distances` precalcula el vector de distancias de Word2Vec `d_W2V`, que
contiene las distancias euclídeas entre todos los pares de embeddings, ordenados por
índice de vocabulario. Este vector se pasa al entrenamiento y no cambia durante él.

---

### Error rate (Sección 4)

`calculate_error_rate` mide cuántas predicciones del circuito no coinciden con los
vectores de etiqueta. Para cada palabra `k` compara los dos estados base con mayor
probabilidad en la salida del circuito con los dos picos del vector de etiqueta. El
error rate es el número de discrepancias dividido entre `2 · |muestras|`. Un valor de
0.0 significa que el circuito reproduce exactamente los dos contextos de cada palabra;
el paper usa este criterio como condición de parada anticipada.

---

### Entrenamiento (Sección 3.1)

`train_qword2vec` optimiza `θ_u` y `θ_v` con SGD con momentum usando SPSA para
aproximar el gradiente. El paper especifica `lr = 0.001` y `momentum = 0.9`.

SPSA (Gacon et al. 2021) aproxima el gradiente con solo **dos evaluaciones** de la
función de pérdida por paso:

    δ ∈ {-1, +1}^d   (vector aleatorio Bernoulli)
    ĝ = (L(θ + c·δ) − L(θ − c·δ)) / (2·c·δ)

Esto es mucho más eficiente que el *parameter shift rule*, que requeriría `2d`
evaluaciones (donde `d` es el número de parámetros). Cada pocas épocas se calcula el
error rate; si llega a 0.0 el entrenamiento se detiene anticipadamente.

---

### Evaluación de los embeddings (Sección 3.2)

`get_embeddings_batched` se usa **después** del entrenamiento, no durante él. Para
cada palabra extrae el vector de Bloch `(⟨X⟩, ⟨Y⟩, ⟨Z⟩)` de cada qubit de embedding
calculando la traza parcial del estado para aislar ese qubit. El embedding de cada
palabra tiene dimensión `3·ne` y es el análogo cuántico de los vectores de Word2Vec.

`evaluate_embedding_quality` calcula la correlación de Pearson entre las distancias de
Word2Vec y las distancias de Q-Word2Vec sobre el vocabulario común. Una correlación
alta indica que Q-Word2Vec ha preservado la geometría semántica de Word2Vec. Esta es
la **métrica principal de éxito** del paper.

`save_qword2vec_embeddings` guarda los vectores de Bloch en
`qword2vec_embeddings.txt`.

---

## `qvisualization.py` — Visualizaciones

Contiene todo el código de generación de imágenes, aislado de la lógica de
entrenamiento. `qWord2Vec.py` lo importa y lo llama solo cuando
`SHOW_VISUALIZATIONS = True`.

- `plot_circuit`: genera el diagrama del circuito completo (U → reset → V) en formato
  matplotlib y lo guarda en `qword2vec_circuit.png`.
- `plot_embeddings_2d`: proyecta los vectores de Bloch con PCA a 2D y genera un scatter
  plot en `qword2vec_embeddings_2d.png`.
- `plot_bloch_sphere`: muestra todos los embeddings sobre la esfera de Bloch con
  colores diferenciados por palabra (solo en pantalla, no guarda en disco).

---

## Relación entre los archivos

    smallCorpora.txt  ──┐
    smallWordList.txt ──┤
                        ├──► word2Vec.py ─────────────► word2vec_embeddings.txt
                        │                                          │
                        │                              (ground truth: vectores 2D)
                        │                                          │
                        └──► qWord2Vec.py ◄────────────────────────
                                   │
                                   │  1. Genera label vectors desde los pares Skip-Gram
                                   │  2. Estima L con la fórmula heurística (§3.4)
                                   │  3. Construye circuito U(θ_u) → reset → V(θ_v)
                                   │  4. Entrena: CrossEntropy − C·Pearson(d_Q, d_W2V)
                                   │  5. Extrae embeddings de Bloch y evalúa correlación
                                   │
                                   ├──► qword2vec_embeddings.txt
                                   └──► qvisualization.py (si SHOW_VISUALIZATIONS=True)
                                              ├──► qword2vec_circuit.png
                                              ├──► qword2vec_embeddings_2d.png
                                              └──► Bloch sphere (pantalla)

**`word2Vec.py` debe ejecutarse primero.** Genera `word2vec_embeddings.txt`; si ese
fichero no existe `qWord2Vec.py` lanza `FileNotFoundError` antes de hacer nada.

**Word2Vec es ground truth, no supervisión directa.** Q-Word2Vec no imita los
parámetros de Word2Vec; trata de reproducir la *geometría* de su espacio de
embeddings. La correlación de Pearson final entre los vectores de distancias de
ambos modelos es el número que resume si el experimento ha funcionado.

**La función de pérdida es el puente.** El término `−C·PearsonR` de la ec. 5 usa
las distancias de Word2Vec como señal de regularización geométrica. Sin él el
circuito podría aprender a predecir contextos correctamente pero con palabras
semánticamente similares dispersas sin coherencia en el espacio cuántico.