import itertools
import os
import numpy as np
from scipy.spatial.distance import squareform
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_algorithms.optimizers import QNSPSA

from MyTools import (
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
from MyQWord2VecQNSPSA import qword2vec_circuit, forward_pass, calculate_custom_loss

np.random.seed(42)

# ── Configuración fija ────────────────────────────────────────────────────
N_QUBITS = 4
N_EMBEDDING = 2
N_LAYERS = None
GS_SHOTS = 256  # menos shots para acelerar cada trial
GS_ITERATIONS = 1500  # iteraciones por combinación
EDUCATED_GUESS = 6  # intentos de punto inicial por trial
C_VAL = 3
ATH = 0.02
ERROR_TOL = 0.1

# Hiperparámetros SPSA fijos (solo spsa_A varía con spsa_A_pct)
SPSA_C = 0.2
SPSA_GAMMA = 0.101
SPSA_A = 0.1
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

if N_LAYERS is None:
    N_LAYERS = estimate_num_layers(N_QUBITS, N_EMBEDDING, len(word_to_id), ATH)

qc, input_p, thetas = qword2vec_circuit(N_QUBITS, N_EMBEDDING, N_LAYERS)

qc_data = []
for embedded_word in label_vectors:
    bin_word = bin(embedded_word)[2:].rjust(N_QUBITS, "0")
    int_word = [int(bit) for bit in bin_word]
    qc_data.append(qc.assign_parameters({input_p: int_word}))

sim = AerSimulator()
aer_sampler = AerSampler()

# Precalcula los tres circuitos de fidelidad candidatos
n_circuits = len(qc_data)
fid_indices = {
    "primero": 0,
    "medio": n_circuits // 2,
    "último": n_circuits - 1,
}
fidelities = {
    name: QNSPSA.get_fidelity(
        qc_data[idx].remove_final_measurements(inplace=False).decompose(),
        aer_sampler,
    )
    for name, idx in fid_indices.items()
}

print(
    f"Datos listos — vocab={len(word_to_id)}, muestras={len(label_vectors)}, L={N_LAYERS}"
)


# ── Función de pérdida ────────────────────────────────────────────────────
def loss_f(param):
    probs = forward_pass(qc_data, thetas, param, GS_SHOTS, sim)
    return calculate_custom_loss(probs, target_distances, label_vectors, C_VAL)


# ── Ejecución de un trial ─────────────────────────────────────────────────
def run_trial(regularization, hessian_delay, spsa_A_pct):
    spsa_A_val = GS_ITERATIONS * spsa_A_pct

    # Punto inicial: educated guess
    best_init, theta_init = np.inf, None
    for _ in range(EDUCATED_GUESS):
        p = np.random.rand(len(thetas))
        l = loss_f(p)
        if l < best_init:
            best_init, theta_init = l, p

    best_er = [np.inf]
    last_er = [np.inf]

    def make_lr():
        k = 0
        while True:
            yield SPSA_A / (k + 1 + spsa_A_val) ** SPSA_ALPHA
            k += 1

    def make_pert():
        k = 0
        while True:
            yield SPSA_C / (k + 1) ** SPSA_GAMMA
            k += 1

    def callback(_nfev, x, fx, _dx, _accept):
        probs = forward_pass(qc_data, thetas, x, GS_SHOTS, sim)
        er = calculate_error_rate(probs, label_vectors)
        last_er[0] = er
        if er < best_er[0]:
            best_er[0] = er

    def termination_checker(_x, _fx, _nfev, _stepsize, _accept):
        return np.isclose(last_er[0], 0.0, atol=ERROR_TOL)

    qnspsa = QNSPSA(
        fidelity=fidelities["primero"],
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
    return best_er[0]


# ── Grid ──────────────────────────────────────────────────────────────────
param_grid = {
    "regularization": [5e-4, 3e-3],
    "hessian_delay":  [300, 500, 700],
    "spsa_A_pct":     [0.03, 0.08],
}
# blocking=True fijado (recomendación estándar para landscapes cuánticos ruidosos)
# fidelity_name="primero" fijado (el circuito elegido afecta poco al tensor métrico)

keys = list(param_grid.keys())
combos = list(itertools.product(*param_grid.values()))
total = len(combos)

print(f"\nGrid search: {total} combinaciones × {GS_ITERATIONS} iter/combo\n" + "=" * 60)

results = []
for i, combo in enumerate(combos, 1):
    params = dict(zip(keys, combo))
    label = (
        f"reg={params['regularization']:.0e}  "
        f"hd={params['hessian_delay']:>3}  "
        f"A%={params['spsa_A_pct']:.0%}"
    )
    print(f"[{i:>3}/{total}]  {label}", end="  →  ", flush=True)
    try:
        best_er = run_trial(**params)
        print(f"best_er={best_er:.4f}")
    except Exception as e:
        best_er = np.inf
        print(f"ERROR: {e}")
    results.append({**params, "best_er": best_er})

results.sort(key=lambda x: x["best_er"])

# ── Guardar resultados ────────────────────────────────────────────────────
out_file = "grid_search_results_refined.txt"
with open(out_file, "w") as f:
    f.write(f"Grid Search QNSPSA — {GS_ITERATIONS} iter/combo, {GS_SHOTS} shots\n")
    f.write("=" * 55 + "\n")
    header = f"{'Rank':>4}  {'reg':>6}  {'hd':>5}  {'A_pct':>5}  {'best_er':>8}\n"
    f.write(header)
    f.write("-" * 55 + "\n")
    for rank, r in enumerate(results, 1):
        f.write(
            f"{rank:>4}  {r['regularization']:>6.0e}  {r['hessian_delay']:>5}  "
            f"{r['spsa_A_pct']:>5.0%}  {r['best_er']:>8.4f}\n"
        )
    f.write("\n=== MEJOR COMBINACIÓN ===\n")
    for k, v in results[0].items():
        f.write(f"  {k}: {v}\n")

print(f"\n{'=' * 60}")
print("=== MEJOR COMBINACIÓN ===")
for k, v in results[0].items():
    print(f"  {k}: {v}")
print(f"\nResultados completos guardados en {out_file}")
