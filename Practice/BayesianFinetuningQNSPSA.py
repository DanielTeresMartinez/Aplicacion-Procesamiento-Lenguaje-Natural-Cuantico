"""
Bayesian fine-tuning de QNSPSA con Optuna (TPE sampler).

Punto de partida empírico:
  - Ath 0.2 y 0.3 → error_rate ≈ 0.2  (mejores del grid search)
  - Ambos con hessian_delay = 700
  → Explorar hessian_delay > 700 y distintos valores de regularización.
"""

import os
import numpy as np
import optuna
from optuna.samplers import TPESampler
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
from my_qword2vec_qnspsa import qword2vec_circuit, forward_pass, calculate_custom_loss

np.random.seed(42)

# ── Configuración fija ────────────────────────────────────────────────────
N_QUBITS = 4
N_EMBEDDING = 2
BF_SHOTS = 256
BF_ITERATIONS = 1500
EDUCATED_GUESS = 6
C_VAL = 3

# Hiperparámetros SPSA fijos
SPSA_C = 0.2
SPSA_GAMMA = 0.101
SPSA_A = 0.1
SPSA_ALPHA = 0.602

# Número de trials Bayesianos
N_TRIALS = 12

# ── Espacio de búsqueda ───────────────────────────────────────────────────
# Basado en resultados empíricos del grid search:
#   · ath ∈ [0.015, 0.03]  — vecindad de 0.02 y 0.03
#   · hessian_delay ∈ [200, 800]  — garantiza activación QNSPSA con margen
#     suficiente para mejorar antes de convergencia (~1500 iters máx.)
#   · regularization ∈ [1e-5, 5e-2]  — exploración amplia 3 órdenes magnitud
ATH_LOW, ATH_HIGH = 0.015, 0.03
HD_LOW, HD_HIGH = 200, 800
REG_LOW, REG_HIGH = 1e-5, 5e-2

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

# ── Caché de circuitos por L ──────────────────────────────────────────────
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


# ── Función de pérdida ────────────────────────────────────────────────────
def make_loss_f(qc_data_local, thetas_local):
    def loss_f(param):
        probs = forward_pass(qc_data_local, thetas_local, param, BF_SHOTS, sim)
        return calculate_custom_loss(probs, label_vectors, C_VAL)

    return loss_f


# ── Ejecución de un trial ─────────────────────────────────────────────────
def run_trial(ath, regularization, hessian_delay, n_iter=BF_ITERATIONS):
    n_layers, qc_data_local, thetas_local, fidelity_local = get_circuit_for_ath(ath)
    spsa_A_val = n_iter * 0.1
    loss_f = make_loss_f(qc_data_local, thetas_local)

    best_init, theta_init = np.inf, None
    for _ in range(EDUCATED_GUESS):
        p = np.random.rand(len(thetas_local))
        l = loss_f(p)
        if l < best_init:
            best_init, theta_init = l, p

    best_er = [np.inf]
    best_x = [None]
    last_er = [np.inf]
    loss_hist = []  # (iter, loss, error_rate) de este trial

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
        probs = forward_pass(qc_data_local, thetas_local, x, BF_SHOTS, sim)
        er = calculate_error_rate(probs, label_vectors)
        last_er[0] = er
        loss_hist.append((len(loss_hist) + 1, fx, er))
        if er < best_er[0]:
            best_er[0] = er
            best_x[0] = x.copy()

    def termination_checker(_x, _fx, _nfev, _stepsize, _accept):
        return np.isclose(last_er[0], 0.0, atol=0.1)

    qnspsa = QNSPSA(
        fidelity=fidelity_local,
        maxiter=n_iter,
        blocking=True,
        regularization=regularization,
        hessian_delay=hessian_delay,
        resamplings=5,
        learning_rate=make_lr,
        perturbation=make_pert,
        callback=callback,
        termination_checker=termination_checker,
    )
    qnspsa.minimize(loss_f, theta_init)
    return best_er[0], best_x[0], n_layers, loss_hist


# Pesos del mejor trial global (se actualiza en objective)
_global_best_er = [np.inf]
_global_best_x = [None]
WEIGHTS_FILE = "theta_values_QNSPSA.pkl"


# ── Objetivo Optuna ───────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    import pickle

    ath = trial.suggest_float("ath", ATH_LOW, ATH_HIGH)
    hessian_delay = trial.suggest_int("hessian_delay", HD_LOW, HD_HIGH, step=50)
    regularization = trial.suggest_float("regularization", REG_LOW, REG_HIGH, log=True)

    trial_num = trial.number + 1
    label = f"ath={ath:.3f}  reg={regularization:.2e}  hd={hessian_delay}"
    print(f"\n[Trial {trial_num:>3}/{N_TRIALS}]  {label}", end="  →  ", flush=True)

    try:
        best_er, best_x, n_layers, loss_hist = run_trial(
            ath, regularization, hessian_delay
        )
        trial.set_user_attr("n_layers", n_layers)
        print(f"L={n_layers:>2}  best_er={best_er:.4f}")

        # Guardar historial de pérdida de este trial en su propio fichero
        loss_file = f"loss_history_trial{trial_num}.txt"
        with open(loss_file, "w") as f:
            f.write(
                f"# Trial {trial_num}: ath={ath:.4f}  reg={regularization:.2e}  hd={hessian_delay}  L={n_layers}  best_er={best_er:.4f}\n"
            )
            f.write("iter,loss,error_rate\n")
            for it, fx, er in loss_hist:
                f.write(f"{it},{fx},{er}\n")

        # Guardar pesos si este trial es el mejor global hasta ahora
        if best_er < _global_best_er[0] and best_x is not None:
            _global_best_er[0] = best_er
            _global_best_x[0] = best_x
            with open(WEIGHTS_FILE, "wb") as f:
                pickle.dump(best_x, f)
            print(
                f"  → Nuevos mejores pesos guardados en {WEIGHTS_FILE}  (er={best_er:.4f})"
            )

        return best_er
    except Exception as e:
        print(f"ERROR: {e}")
        return np.inf


# ── Ejecutar optimización Bayesiana ──────────────────────────────────────
print(f"\nBayesian fine-tuning: {N_TRIALS} trials, espacio de búsqueda:")
print(f"  ath          ∈ [{ATH_LOW}, {ATH_HIGH}]  (continuo)")
print(f"  hessian_delay∈ [{HD_LOW}, {HD_HIGH}]  (paso 50)")
print(
    f"  regularization∈ [{REG_LOW:.0e}, {REG_HIGH:.0e}]  (log-uniform, centrado en 5e-4)"
)
print("=" * 65)

sampler = TPESampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler)

# Añadir puntos de arranque conocidos (warm start) para que TPE parta
# de las regiones que ya sabemos que funcionan bien.
study.enqueue_trial({"ath": 0.02, "hessian_delay": 400, "regularization": 5e-4})
study.enqueue_trial({"ath": 0.03, "hessian_delay": 400, "regularization": 5e-4})
study.enqueue_trial({"ath": 0.02, "hessian_delay": 700, "regularization": 1.75e-3})
study.enqueue_trial({"ath": 0.03, "hessian_delay": 700, "regularization": 1.75e-3})

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

# ── Resultados ────────────────────────────────────────────────────────────
trials_sorted = sorted(
    [t for t in study.trials if t.value is not None and t.value < np.inf],
    key=lambda t: t.value,
)

out_file = "bayesian_finetuning_results.txt"
with open(out_file, "w") as f:
    f.write(
        f"Bayesian Fine-Tuning QNSPSA — {BF_ITERATIONS} iter/trial, {BF_SHOTS} shots\n"
    )
    f.write(f"Trials ejecutados: {len(trials_sorted)}/{N_TRIALS}\n")
    f.write("=" * 70 + "\n")
    header = (
        f"{'Rank':>4}  {'ath':>6}  {'L':>3}  {'reg':>8}  {'hd':>5}  {'best_er':>8}\n"
    )
    f.write(header)
    f.write("-" * 60 + "\n")
    for rank, t in enumerate(trials_sorted, 1):
        n_layers = t.user_attrs.get("n_layers", -1)
        f.write(
            f"{rank:>4}  {t.params['ath']:>6.3f}  {n_layers:>3}  "
            f"{t.params['regularization']:>8.2e}  "
            f"{t.params['hessian_delay']:>5}  {t.value:>8.4f}\n"
        )
    f.write("\n=== MEJOR COMBINACIÓN ===\n")
    best = study.best_trial
    f.write(f"  ath:            {best.params['ath']:.4f}\n")
    f.write(f"  hessian_delay:  {best.params['hessian_delay']}\n")
    f.write(f"  regularization: {best.params['regularization']:.4e}\n")
    f.write(f"  best_er:        {best.value:.4f}\n")
    f.write(f"  n_layers:       {best.user_attrs.get('n_layers', '?')}\n")

print(f"\n{'=' * 65}")
print("=== MEJOR COMBINACIÓN ===")
best = study.best_trial
print(f"  ath:            {best.params['ath']:.4f}")
print(f"  hessian_delay:  {best.params['hessian_delay']}")
print(f"  regularization: {best.params['regularization']:.4e}")
print(f"  best_er:        {best.value:.4f}")
print(f"  n_layers:       {best.user_attrs.get('n_layers', '?')}")
print(f"\nResultados completos guardados en {out_file}")
