# Cambios realizados en MyQWord2Vec.py

## Problema de partida

El modelo cuántico Q-Word2Vec producía un **error rate de ~0.7** (70% de predicciones incorrectas) sin convergencia visible durante el entrenamiento. El loss oscilaba aleatoriamente sin tendencia descendente.

---

## Causa raíz 1 — Solo 100 iteraciones para 264 parámetros

**Qué pasaba:** El circuito tiene 11 capas × 4 qubits × 3 rotaciones × 2 (U y V) = **264 parámetros entrenables**. SPSA necesita del orden de miles de iteraciones para ese número de parámetros. Con 100, el optimizador apenas exploraba el espacio de búsqueda.

**Cambio:** `iterations = 100` → `iterations = 2000`

---

## Causa raíz 2 — Perturbación fija y demasiado grande (`c = π/2`)

**Qué pasaba:** Para estimar el gradiente, SPSA perturba todos los parámetros simultáneamente en `±c`. Se estaba usando `c = π/2 ≈ 1.57`, que es la mitad del rango de una rotación cuántica. Esto significa que `theta_plus` y `theta_minus` eran configuraciones del circuito completamente distintas entre sí, y el "gradiente" calculado era prácticamente ruido.

Además, `c` era fijo durante todo el entrenamiento. SPSA estándar (Spall 1998) exige que `c` **decrezca** con las épocas para que el gradiente sea cada vez más preciso conforme nos acercamos al mínimo.

**Cambio:** Se eliminó `shift = π/2` y se sustituyó por la secuencia decreciente estándar:

```
c_k = spsa_c / (k + 1)^gamma
```

Con `spsa_c = 0.2` y `gamma = 1/6`. Así la perturbación empieza en 0.2 y va disminuyendo a lo largo de las 2000 épocas.

---

## Causa raíz 3 — Momentum acumulaba ruido en vez de señal

**Qué pasaba:** Se usaba un optimizador SGD con momentum (`momentum = 0.9`). El momentum es útil cuando los gradientes son consistentes en dirección. Los gradientes SPSA son inherentemente ruidosos (se estiman con solo 2 evaluaciones de la función). Con momentum alto, el acumulador `velocity` acumulaba ese ruido iteración tras iteración, amplificando las oscilaciones en lugar de suavizarlas.

**Cambio:** Se eliminó completamente el momentum. La actualización ahora es:

```
theta_values = theta_values - a_k * grad
```

Donde `a_k` también sigue una secuencia decreciente estándar:

```
a_k = spsa_a / (k + 1 + A)^alpha
```

Con `spsa_a = 0.1`, `A = 100` (estabilizador ≈ 10% de las iteraciones) y `alpha = 0.602`.

---

## Causa raíz 4 — Posible NaN en la correlación de Pearson

**Qué pasaba:** Al inicio del entrenamiento, todos los circuitos cuánticos producen distribuciones de probabilidad muy similares (cercanas a la uniforme). La correlación de Pearson sobre vectores casi constantes devuelve `NaN`. Ese `NaN` se propagaba al gradiente, corrompiendo `theta_values` de forma permanente.

**Cambio:** Protección explícita: si `pearsonr` devuelve `NaN`, se trata como correlación 0 (sin señal, pero sin corrupción).

```python
if np.isnan(correlation):
    correlation = 0.0
```

---

## Resumen de los cambios

| Variable | Antes | Después |
|---|---|---|
| `iterations` | 100 | 2000 |
| Perturbación `c` | `π/2` fijo | `0.2 / (k+1)^(1/6)` decreciente |
| Learning rate `a` | `0.001` fijo + momentum | `0.1 / (k+1+100)^0.602` decreciente |
| Momentum | 0.9 | eliminado |
| Protección NaN Pearson | no existía | `if np.isnan: correlation = 0.0` |
