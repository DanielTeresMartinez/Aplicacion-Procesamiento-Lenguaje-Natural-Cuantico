import itertools
import os
import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_algorithms.optimizers import QNSPSA

from my_tools import (
    load_corpus,
    load_word_list,
    load_word2vec_embeddings,
    build_vocabulary,
    generate_training_data_from_text,
    generate_label_vectors,
    estimate_num_layers,
    build_target_distances,
    calculate_error_rate,
)
from qword2vec import (
    qword2vec_circuit,
    forward_pass,
    calculate_custom_loss,
)

np.random.seed(42)

# ── Configuración fija ────────────────────────────────────────────────────
N_QUBITS = 4
N_EMBEDDING = 2
GS_SHOTS = 256
GS_ITERATIONS = 600
EDUCATED_GUESS = 6
C_VAL = 3  # actualizado de 2 -> 3 respecto a la primera exploración

# Exponentes de los schedules (fijos)
SPSA_GAMMA = 0.101
SPSA_ALPHA = 0.602

# ── Carga de datos ────────────────────────────────────────────────────────
print("Cargando datos...")
corpus = load_corpus("smallCorpora.txt")
if os.path.isfile("smallWordList.txt"):
    corpus = corpus + load_word_list("smallWordList.txt")

w2v_embeddings = load_word2vec_embeddings("word2vec_embeddings.txt")
word_to_id, _ = build_vocabulary(corpus, N_QUBITS)
training_data = generate_training_data_from_text(corpus, word_to_id, window_size=1)
label_vectors = generate_label_vectors(training_data, N_QUBITS)
target_distances = build_target_distances(w2v_embeddings, word_to_id)

sim = AerSimulator()
aer_sampler = AerSampler()

print(f"Datos listos — vocab={len(word_to_id)}, muestras={len(label_vectors)}")


# ── Caché de circuitos por L (evita reconstruir cuando dos Ath dan el mismo L) ──
_circuit_cache = {}


def get_circuit_for_ath(ath):
    n_layers = estimate_num_layers(N_QUBITS, N_EMBEDDING, len(word_to_id), ath)
    if n_layers in _circuit_cache:
        return _circuit_cache[n_layers]

    print(f"  [caché] Construyendo circuito para Ath={ath:.3f} → L={n_layers}...")
    qc, input_p, thetas_pv = qword2vec_circuit(N_QUBITS, N_EMBEDDING, n_layers)

    qc_data_local = []
    for embedded_word in label_vectors:
        bin_word = bin(embedded_word)[2:].rjust(N_QUBITS, "0")
        int_word = [int(bit) for bit in bin_word]
        qc_data_local.append(qc.assign_parameters({input_p: int_word}))

    fidelity_local = QNSPSA.get_fidelity(
        qc_data_local[0].remove_final_measurements(inplace=False).decompose(),
        aer_sampler,
    )

    _circuit_cache[n_layers] = (n_layers, qc_data_local, thetas_pv, fidelity_local)
    return _circuit_cache[n_layers]


# ── Función de pérdida (depende del circuito) ─────────────────────────────
def make_loss_f(qc_data_local, thetas_local):
    def loss_f(param):
        probs = forward_pass(qc_data_local, thetas_local, param, GS_SHOTS, sim)
        return calculate_custom_loss(probs, label_vectors, C_VAL)

    return loss_f


# ── Ejecución de un trial ─────────────────────────────────────────────────
def run_trial(ath, regularization, hessian_delay, spsa_a, spsa_c):
    n_layers, qc_data_local, thetas_local, fidelity_local = get_circuit_for_ath(ath)
    spsa_A_val = GS_ITERATIONS * 0.1  # 10% de iteraciones (Spall 1998)
    loss_f = make_loss_f(qc_data_local, thetas_local)

    best_init, theta_init = np.inf, None
    for _ in range(EDUCATED_GUESS):
        p = np.random.rand(len(thetas_local))
        l = loss_f(p)
        if l < best_init:
            best_init, theta_init = l, p

    best_er = [np.inf]
    last_er = [np.inf]

    def make_lr():
        k = 0
        while True:
            yield spsa_a / (k + 1 + spsa_A_val) ** SPSA_ALPHA
            k += 1

    def make_pert():
        k = 0
        while True:
            yield spsa_c / (k + 1) ** SPSA_GAMMA
            k += 1

    def callback(_nfev, x, fx, _dx, _accept):
        probs = forward_pass(qc_data_local, thetas_local, x, GS_SHOTS, sim)
        er = calculate_error_rate(probs, label_vectors)
        last_er[0] = er
        if er < best_er[0]:
            best_er[0] = er

    def termination_checker(_x, _fx, _nfev, _stepsize, _accept):
        return np.isclose(last_er[0], 0.0, atol=0.1)

    qnspsa = QNSPSA(
        fidelity=fidelity_local,
        maxiter=GS_ITERATIONS,
        blocking=True,
        regularization=regularization,
        hessian_delay=hessian_delay,
        learning_rate=make_lr,
        perturbation=make_pert,
        callback=callback,
        termination_checker=termination_checker,
    )
    qnspsa.minimize(loss_f, theta_init)
    return best_er[0], n_layers


# ── Grid ──────────────────────────────────────────────────────────────────
# Primera exploración con el espacio amplio:
#   ath          en {0.015, 0.020, 0.025, 0.030}
#   regularization en {1e-3, 1e-2, 5e-2, 1e-1}
#   hessian_delay  en {100, 200, 300, 500}
#   spsa_a         en {0.05, 0.15, 0.30, 0.60}
#   spsa_c         en {0.03, 0.07, 0.15, 0.30}
#
# Observaciones tras esa primera pasada:
#  - ath en {0.020, 0.025} concentra los mejores resultados (L en {9-11});
#    valores extremos (0.015 o 0.030) dan circuitos sobredimensionados
#    o con poca capacidad expresiva -> error rate > 0.40.
#  - hessian_delay <= 200 converge a peores minimos; el rango 200-400
#    mostro la mejor convergencia global.
#  - regularization en {1e-2, 5e-2} domino el top-5; los extremos peores.
#  - spsa_a en {0.10-0.25} mejor comportamiento; valores altos (> 0.30)
#    producen actualizaciones inestables y bajos (< 0.10) convergencia lenta.
#  - spsa_c en {0.05-0.10} optimo; valores mayores sobreperturbaban el
#    espacio de parametros y dificultaban la convergencia del Hessiano.
#
param_grid = {
    "ath": [0.018, 0.021, 0.024],
    "hessian_delay": [300, 400, 500],
    "regularization": [8e-3, 1.6e-2, 3e-2],
    "spsa_a": [0.10, 0.16, 0.25],
    "spsa_c": [0.05, 0.07, 0.12],
}

keys = list(param_grid.keys())
combos = list(itertools.product(*param_grid.values()))
total = len(combos)

print(
    f"\nGrid search (segunda exploración): {total} combinaciones × {GS_ITERATIONS} iter/combo\n"
    + "=" * 65
)

results = []
for i, combo in enumerate(combos, 1):
    params = dict(zip(keys, combo))
    label = (
        f"ath={params['ath']:.3f}  "
        f"hd={params['hessian_delay']:>3}  "
        f"reg={params['regularization']:.1e}  "
        f"a={params['spsa_a']:.2f}  "
        f"c={params['spsa_c']:.2f}"
    )
    print(f"[{i:>3}/{total}]  {label}", end="  →  ", flush=True)
    try:
        best_er, n_layers = run_trial(**params)
        print(f"L={n_layers:>2}  best_er={best_er:.4f}")
    except Exception as e:
        best_er, n_layers = np.inf, -1
        print(f"ERROR: {e}")
    results.append({**params, "L": n_layers, "best_er": best_er})

results.sort(key=lambda x: x["best_er"])

# ── Guardar resultados ────────────────────────────────────────────────────
out_file = "grid_search_results_refined.txt"
with open(out_file, "w") as f:
    f.write(
        f"Grid Search QNSPSA (segunda exploración) — {GS_ITERATIONS} iter/combo, {GS_SHOTS} shots\n"
    )
    f.write("=" * 80 + "\n")
    header = f"{'Rank':>4}  {'ath':>5}  {'L':>3}  {'hd':>5}  {'reg':>7}  {'a':>6}  {'c':>6}  {'best_er':>8}\n"
    f.write(header)
    f.write("-" * 65 + "\n")
    for rank, r in enumerate(results, 1):
        f.write(
            f"{rank:>4}  {r['ath']:>5.3f}  {r['L']:>3}  {r['hessian_delay']:>5}  "
            f"{r['regularization']:>7.1e}  {r['spsa_a']:>6.3f}  {r['spsa_c']:>6.3f}  "
            f"{r['best_er']:>8.4f}\n"
        )
    f.write("\n=== MEJOR COMBINACIÓN ===\n")
    for k, v in results[0].items():
        f.write(f"  {k}: {v}\n")

print(f"\n{'=' * 65}")
print("=== MEJOR COMBINACIÓN ===")
for k, v in results[0].items():
    print(f"  {k}: {v}")
print(f"\nResultados completos guardados en {out_file}")
