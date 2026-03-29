import os
import pickle
from MyTools import (
    load_corpus,
    load_word_list,
    load_word2vec_embeddings,
    build_vocabulary,
)
from MyTools import generate_training_data_from_text, generate_label_vectors
from MyTools import estimate_num_layers, build_target_distances
from MyTools import calculate_error_rate, plot_embeddings_comparison


import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import QNSPSA
from qiskit_aer.primitives import SamplerV2 as AerSampler

np.random.seed(42)

# Early stopping
EVAL_EVERY = 100
PATIENCE = 5
MIN_DELTA = 0.005


def qword2vec_circuit(n_qubits, n_embedding, n_layers):

    # Registros para guardar info en los subcircuitos
    qr_embedding = QuantumRegister(n_embedding, name="qEmb")
    qr_remainingU = QuantumRegister(n_qubits - n_embedding, name="qRU")
    qr_remainingV = QuantumRegister(n_qubits - n_embedding, name="qRV")

    cr_output = ClassicalRegister(n_qubits, name="cout")

    # Embedding de entradas
    p_encoding = ParameterVector(name="pEnc", length=n_qubits)

    # Parámetros
    num_params = 3 * n_qubits * n_layers * 2  # Para U y para V
    thetas = ParameterVector(name="Theta", length=num_params)

    # Circuito de codificación de datos
    qc_encoding = QuantumCircuit(qr_embedding, qr_remainingU)
    for i in range(n_qubits):
        qc_encoding.rx(p_encoding[i] * np.pi, i)

    # Circuito de U
    U = QuantumCircuit(qr_embedding, qr_remainingU)
    p_idx = 0
    for l in range(n_layers):
        for q in range(n_qubits):
            U.rx(thetas[p_idx], q)
            U.ry(thetas[p_idx + 1], q)
            U.rz(thetas[p_idx + 2], q)
            p_idx += 3
        for q in range(n_qubits):
            U.cx(q, (q + 1) % n_qubits)

    # Circuito de V
    V = QuantumCircuit(qr_embedding, qr_remainingV)
    for l in range(n_layers):
        for q in range(n_qubits):
            V.rx(thetas[p_idx], q)
            V.ry(thetas[p_idx + 1], q)
            V.rz(thetas[p_idx + 2], q)
            p_idx += 3
        for q in range(n_qubits):
            V.cx(q, (q + 1) % n_qubits)

    # Circuito final
    qc = QuantumCircuit(qr_embedding, qr_remainingU, qr_remainingV, cr_output)
    qc.append(qc_encoding, [*qr_embedding, *qr_remainingU])
    qc.append(U, [*qr_embedding, *qr_remainingU])
    qc.append(V, [*qr_embedding, *qr_remainingV])
    qc.measure([*qr_embedding, *qr_remainingV], cr_output)

    return qc, p_encoding, thetas


def forward_pass(
    qc_data: list[QuantumCircuit],
    param: ParameterVector,
    param_values: np.ndarray,
    n_shots: int,
    sim: AerSimulator,
):
    n_output_bits = qc_data[0].num_clbits

    qc_param = [qc.assign_parameters({param: param_values}) for qc in qc_data]
    tr_qc = transpile(qc_param, sim)
    result = sim.run(tr_qc, shots=n_shots).result()

    prob_array = np.zeros((len(qc_data), 2**n_output_bits))
    for idx, qc in enumerate(tr_qc):
        measurements = result.get_counts(qc)
        for outcome in measurements:
            int_outcome = int(outcome[::-1], 2)
            prob_array[idx, int_outcome] = measurements[outcome] / n_shots

    return prob_array


def calculate_custom_loss(
    prob_distributions, target_distances, label_vectors, C, q_dists=None
):
    if q_dists is None:
        q_dists = pdist(prob_distributions, metric="euclidean")

    assert len(target_distances) == len(q_dists)
    correlation, _ = pearsonr(q_dists, target_distances)
    if np.isnan(correlation):
        correlation = 0.0

    eps, loss, count = 1e-10, 0.0, 0
    for k, label in label_vectors.items():
        loss += -np.sum(label * np.log2(prob_distributions[k] + eps))
        count += 1
    cross_entropy = loss / count if count > 0 else 0.0
    return cross_entropy - (C * correlation)


if __name__ == "__main__":
    SHOW_VISUALIZATIONS = True
    # True  → entrena y guarda los pesos en WEIGHTS_FILE
    # False → carga los pesos desde WEIGHTS_FILE y salta el entrenamiento
    TRAIN = True
    WEIGHTS_FILE = "theta_values_QNSPSA.pkl"
    n_qubits = 4
    n_embedding = 2
    n_layers = None
    n_shots = 2048  # más shots que en Bayesiano para menor varianza

    print(f"--- Q-Word2Vec (n={n_qubits}, ne={n_embedding}) ---")

    # ── Datos y corpus ───────────────────────────────────────────────────────
    try:
        corpus = load_corpus("smallCorpora.txt")
        print("Corpus de frases cargado.")
    except FileNotFoundError:
        print("Error: smallCorpora.txt no encontrado.")
        exit(1)

    if os.path.isfile("smallWordList.txt"):
        corpus = corpus + load_word_list("smallWordList.txt")
    else:
        print("[INFO] smallWordList.txt no encontrado — entrenando solo con frases.")

    w2v_embeddings = load_word2vec_embeddings("word2vec_embeddings.txt")

    word_to_id, id_to_word = build_vocabulary(corpus, n_qubits)
    print(f"Vocabulario: {len(word_to_id)}/{2**n_qubits} palabras")

    # ── §4 · Vectores de etiqueta ────────────────────────────────────────────
    training_data = generate_training_data_from_text(corpus, word_to_id, window_size=1)
    label_vectors = generate_label_vectors(training_data, n_qubits)
    print(f"Muestras de entrenamiento: {len(label_vectors)}")

    # ── §3.4 · Estimación de la profundidad L ────────────────────────────────
    num_data = len(word_to_id)
    ath = 0.030
    if n_layers is None:
        n_layers = estimate_num_layers(n_qubits, n_embedding, num_data, ath)
        print(f"L heurístico = {n_layers}  (pares={num_data}, Ath={ath})")
    else:
        print(f"L fija = {n_layers}")

    # ── §3.2 · Distancias objetivo Word2Vec ──────────────────────────────────
    target_distances = build_target_distances(w2v_embeddings, word_to_id)

    qc, input_p, thetas = qword2vec_circuit(n_qubits, n_embedding, n_layers)

    # Circuitos con el dataset empotrado
    qc_data = []
    for embedded_word in label_vectors:
        bin_word = bin(embedded_word)[2:].rjust(n_qubits, "0")
        int_word = [int(bit) for bit in bin_word]
        qc_data.append(qc.assign_parameters({input_p: int_word}))

    sim = AerSimulator()
    aer_sampler = AerSampler()
    fidelity_circuit = qc_data[0].remove_final_measurements(inplace=False).decompose()
    fidelity = QNSPSA.get_fidelity(fidelity_circuit, aer_sampler)

    dist_matrix = squareform(target_distances)

    iterations = 2500
    c_val = 3
    spsa_c = 0.2
    spsa_gamma = 0.101
    spsa_a = 0.1
    spsa_A = iterations * 0.1
    spsa_alpha = 0.602
    loss_history = []

    def loss_f(param):
        prob_array = forward_pass(qc_data, thetas, param, n_shots, sim)
        return calculate_custom_loss(prob_array, target_distances, label_vectors, c_val)

    if TRAIN:
        # ── Inicialización educada: elegir el mejor entre N arranques aleatorios
        educated_guess = 10
        if educated_guess is not None:
            best_init_loss = np.inf
            for i in range(educated_guess):
                p = np.random.rand(len(thetas))
                c_loss = loss_f(p)
                if c_loss < best_init_loss:
                    best_init_loss, theta_values = c_loss, p
        else:
            theta_values = np.random.rand(len(thetas))

        print(f"====== Fase de entrenamiento ======\n")
        print(
            f"Early stopping: PATIENCE={PATIENCE} evaluaciones "
            f"cada {EVAL_EVERY} iters, MIN_DELTA={MIN_DELTA}\n"
        )

        def make_learning_rate():
            k = 0
            while True:
                yield spsa_a / (k + 1 + spsa_A) ** spsa_alpha
                k += 1

        def make_perturbation():
            k = 0
            while True:
                yield spsa_c / (k + 1) ** spsa_gamma
                k += 1

        loss_file = open("loss_history_QNSPSA.txt", "w")
        loss_file.write("epoch,loss,error_rate\n")

        # Estado compartido entre los dos callbacks
        best_er = [np.inf]
        best_x = [None]
        last_er = [np.inf]
        no_improve_count = [0]
        stop_flag = [False]

        def spsa_callback(_nfev, x, fx, _dx, _accept):
            loss_history.append(fx)
            it = len(loss_history)

            probs = forward_pass(qc_data, thetas, x, n_shots, sim)
            er = calculate_error_rate(probs, label_vectors)
            last_er[0] = er

            # Guardar los mejores pesos siempre que haya mejora real
            if er < best_er[0] - MIN_DELTA:
                best_er[0] = er
                best_x[0] = x.copy()
                with open(WEIGHTS_FILE, "wb") as f:
                    pickle.dump(best_x[0], f)

            loss_file.write(f"{it},{fx},{er}\n")
            loss_file.flush()

            # ── Checkpoint de early stopping cada EVAL_EVERY iteraciones ─────
            if it % EVAL_EVERY == 0 or it == 1:
                if er < best_er[0] + MIN_DELTA:
                    # Dentro del rango de mejora → resetear contador
                    no_improve_count[0] = 0
                    improve_tag = " (new best)" if er <= best_er[0] else ""
                else:
                    # Sin mejora suficiente → incrementar contador
                    no_improve_count[0] += 1
                    improve_tag = ""

                print(
                    f"  Época {it:>4}/{iterations}"
                    f"  |  Pérdida ≈ {fx:+.4f}"
                    f"  |  Error rate ≈ {er:.4f}"
                    f"  |  Mejor ≈ {best_er[0]:.4f}{improve_tag}"
                    f"  (sin_mejora={no_improve_count[0]}/{PATIENCE})"
                )

                if no_improve_count[0] >= PATIENCE:
                    print(
                        f"\n[Early stopping] Sin mejora en {PATIENCE} evaluaciones "
                        f"consecutivas (cada {EVAL_EVERY} iters). "
                        f"Mejor error rate: {best_er[0]:.4f}"
                    )
                    stop_flag[0] = True

        def termination_checker(_x, _fx, _nfev, _stepsize, _accept):
            return stop_flag[0]

        qnspsa = QNSPSA(
            fidelity=fidelity,
            maxiter=iterations,
            blocking=True,
            regularization=5e-4,    # mejor valor encontrado por fine-tuning bayesiano
            hessian_delay=1000,     # mejor valor encontrado por fine-tuning bayesiano
            learning_rate=make_learning_rate,
            perturbation=make_perturbation,
            callback=spsa_callback,
            termination_checker=termination_checker,
        )

        try:
            result = qnspsa.minimize(loss_f, theta_values)
        finally:
            loss_file.close()

        # Recuperar los mejores pesos encontrados durante el entrenamiento
        with open(WEIGHTS_FILE, "rb") as f:
            theta_values = pickle.load(f)
        print(
            f"Mejor error rate encontrado: {best_er[0]:.4f}"
            f" — pesos cargados desde {WEIGHTS_FILE}"
        )

    else:
        print(f"Cargando pesos desde {WEIGHTS_FILE} ...")
        with open(WEIGHTS_FILE, "rb") as f:
            theta_values = pickle.load(f)

    final_probs = forward_pass(qc_data, thetas, theta_values, n_shots, sim)
    error_rate = calculate_error_rate(final_probs, label_vectors)
    q_dists_final = pdist(final_probs, metric="euclidean")
    correlation_final, _ = pearsonr(q_dists_final, target_distances)
    print(f"Error rate final:          {error_rate:.4f}")
    print(f"Correlación de Pearson:    {correlation_final:.4f}  (paper: 0.81)")

    plt.plot(loss_history)
    plt.show()

    plot_embeddings_comparison(final_probs, w2v_embeddings, word_to_id, id_to_word)
