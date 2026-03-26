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
from MyTools import calculate_error_rate


import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import SPSA

np.random.seed(42)


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

    # Codificación de datos
    for i in range(n_qubits):
        qc_encoding.rx(p_encoding[i] * np.pi, i)

    # Circuito de U
    U = QuantumCircuit(qr_embedding, qr_remainingU)
    p_idx = 0
    for l in range(n_layers):
        # Parte variacional
        for q in range(n_qubits):
            U.rx(thetas[p_idx], q)
            U.ry(thetas[p_idx + 1], q)
            U.rz(thetas[p_idx + 2], q)
            p_idx += 3
        # Parte de entrelazamiento
        for q in range(n_qubits):
            U.cx(q, (q + 1) % n_qubits)

    # Circuito de V
    V = QuantumCircuit(qr_embedding, qr_remainingV)
    for l in range(n_layers):
        # Parte variacional
        for q in range(n_qubits):
            V.rx(thetas[p_idx], q)
            V.ry(thetas[p_idx + 1], q)
            V.rz(thetas[p_idx + 2], q)
            p_idx += 3
        # Parte de entrelazamiento
        for q in range(n_qubits):
            V.cx(q, (q + 1) % n_qubits)

    # Circuito final
    qc = QuantumCircuit(qr_embedding, qr_remainingU, qr_remainingV, cr_output)
    qc.append(qc_encoding, [*qr_embedding, *qr_remainingU])
    qc.append(U, [*qr_embedding, *qr_remainingU])
    qc.append(V, [*qr_embedding, *qr_remainingV])

    qc.measure([*qr_embedding, *qr_remainingV], cr_output)

    """
    f= qc.draw('mpl')
    display(f)
    """

    return qc, p_encoding, thetas


def forward_pass(
    qc_data: list[QuantumCircuit],
    param: ParameterVector,
    param_values: np.ndarray,
    n_shots: int,
    sim: AerSimulator,
):

    # Asignación de parámetros a circuitos
    n_output_bits = qc_data[0].num_clbits

    qc_param = []
    for qc in qc_data:
        qc_param.append(qc.assign_parameters({param: param_values}))

    tr_qc = transpile(qc_param, sim)
    result = sim.run(tr_qc, shots=n_shots).result()

    prob_array = np.zeros((len(qc_data), 2**n_output_bits))
    # Convierte resultados en probabilidades
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

    # Entropía cruzada
    eps, loss, count = 1e-10, 0.0, 0

    for k, label in label_vectors.items():
        loss += -np.sum(label * np.log2(prob_distributions[k] + eps))
        count += 1
    cross_entropy = loss / count if count > 0 else 0.0
    return cross_entropy - (C * correlation)


if __name__ == "__main__":
    # False para omitir las visualizaciones y acelerar la ejecución
    SHOW_VISUALIZATIONS = True
    # True  → entrena y guarda los pesos en theta_values.pkl
    # False → carga los pesos desde theta_values.pkl y salta el entrenamiento
    TRAIN = True
    WEIGHTS_FILE = "theta_values.pkl"
    n_qubits = 4
    n_embedding = 2
    n_layers = None
    n_shots = 1024

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

    """
    print('------------------------')
    print('w2v_embeddings')
    print(w2v_embeddings)
    print('.--------------------')
    """

    word_to_id, id_to_word = build_vocabulary(corpus, n_qubits)
    print(f"Vocabulario: {len(word_to_id)}/{2**n_qubits} palabras")

    # ── §4 · Vectores de etiqueta ────────────────────────────────────────────
    # window_size=1 → los dos vecinos inmediatos, coincidiendo con los
    # "two surrounding words" de la Sección 4.
    training_data = generate_training_data_from_text(corpus, word_to_id, window_size=1)
    label_vectors = generate_label_vectors(training_data, n_qubits)
    print(f"Muestras de entrenamiento: {len(label_vectors)}")

    """
    print('--------------------------------')
    print('training: ', training_data)
    print('id2w', id_to_word)
    print('labels', label_vectors)
    print(len(label_vectors[0]))
    print('--------------------------------')
    """

    # ── §3.4 · Estimación de la profundidad L ────────────────────────────────
    num_data = len(word_to_id)
    # Mejor valor encontrado indicado por el paper
    ath = 0.02
    if n_layers is None:
        n_layers = estimate_num_layers(n_qubits, n_embedding, num_data, ath)
        print(f"L heurístico = {n_layers}  (pares={num_data}, Ath={ath})")
    else:
        print(f"L fija= {n_layers}")

    # ── §3.2 · Distancias objetivo Word2Vec ──────────────────────────────────
    target_distances = build_target_distances(w2v_embeddings, word_to_id)

    """
    print('--------------------------------')
    print(len(target_distances))
    print(len(label_vectors))
    print(len(w2v_embeddings))
    print('--------------------------------')
    """

    qc, input_p, thetas = qword2vec_circuit(n_qubits, n_embedding, n_layers)

    # Circuitos con el dataset empotrado
    qc_data = []
    for embedded_word in label_vectors:
        bin_word = bin(embedded_word)[2:].rjust(n_qubits, "0")
        int_word = [int(bit) for bit in bin_word]

        qc_data.append(qc.assign_parameters({input_p: int_word}))

    sim = AerSimulator()

    # Matriz NxN de distancias W2V, es el ground truth fijo
    dist_matrix = squareform(target_distances)

    iterations = 2000
    c_val = 3
    # Hiperparámetros SPSA con secuencias decrecientes (Spall 1998)
    # c_k = spsa_c / (k+1)^gamma  →  perturbación que decrece con las épocas
    # a_k = spsa_a / (k+1+A)^alpha →  learning rate que decrece con las épocas
    spsa_c = 0.2
    spsa_gamma = 1 / 6
    spsa_a = 0.1
    spsa_A = 100  # estabilizador: ≈ 10% de las iteraciones
    spsa_alpha = 0.602
    loss_history = []
    step_show = 100

    def loss_f(param):
        prob_array = forward_pass(qc_data, thetas, param, n_shots, sim)
        return calculate_custom_loss(prob_array, target_distances, label_vectors, c_val)

    if TRAIN:
        ERROR_TOLERANCE = 1e-4
        # Valores de parámetros iniciales (ángulos [-pi, pi])
        # Tratamos de dar valores iniciales "mejores" que el primer aleatorio que se encuentra
        educated_guess = 10
        if educated_guess is not None:
            best_loss = np.inf
            for i in range(educated_guess):
                p = np.random.rand(len(thetas))
                c_loss = loss_f(p)
                if c_loss < best_loss:
                    best_loss, theta_values = c_loss, p
        else:
            theta_values = np.random.rand(len(thetas))

        print(f"====== Fase de entrenamiento ======\n")
        # VERSIÓN BUCLE FOR MANUAL

        # loss_file = open("loss_history.txt", "w")
        # loss_file.write("epoch,loss\n")

        # Implementación seguida de: https://pennylane.ai/qml/demos/tutorial_spsa
        # for it in range(iterations):
        #     c_k = spsa_c / (it + 1) ** spsa_gamma
        #     a_k = spsa_a / (it + 1 + spsa_A) ** spsa_alpha
        #
        #     # Vector de perturbación aleatorio ±1 (distribución de Rademacher)
        #     delta = np.random.choice([-1.0, 1.0], size=len(theta_values))
        #
        #     # Dos evaluaciones de la función de pérdida con parámetros perturbados
        #     loss_plus  = loss_f(theta_values + c_k * delta)
        #     loss_minus = loss_f(theta_values - c_k * delta)
        #
        #     # Estimación del gradiente SPSA:
        #     #    g_k = (L(θ + c_k·Δ) - L(θ - c_k·Δ)) / (2·c_k·Δ)
        #     grad = (loss_plus - loss_minus) / (2 * c_k * delta)
        #
        #     # Actualización de pesos:
        #     #    θ_{k+1} = θ_k - a_k · g_k
        #     theta_values = theta_values - a_k * grad
        #
        #     current_loss = (loss_plus + loss_minus) / 2.0
        #     loss_history.append(current_loss)
        #     loss_file.write(f"{it + 1},{current_loss}\n")
        #     loss_file.flush()
        #     if (it + 1) % step_show == 0 or it == 0:
        #         probs = forward_pass(qc_data, thetas, theta_values, n_shots, sim)
        #         er = calculate_error_rate(probs, label_vectors)
        #         print(f"  Época {it + 1:>4}/{iterations}  |  Pérdida ≈ {current_loss:.4f}  |  Error rate ≈ {er:.4f}")
        #         if np.isclose(er, 0.0, atol=ERROR_TOLERANCE):
        #             print(f"Early stop: error rate ≈ {ERROR_TOLERANCE} alcanzado.")
        #             break
        #
        # loss_file.close()

        # FIN VERSIÓN BUCLE FOR MANUAL

        # VERSIÓN UTILIZANDO LA CLASE SPSA DE QISKIT

        # Secuencias decrecientes como generadores (Spall 1998)
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

        class EarlyStop(Exception):
            pass

        loss_file = open("loss_history_qiskit.txt", "w")
        loss_file.write("epoch,loss,error rate\n")
        last_x = [None]

        def spsa_callback(_nfev, x, fx, _dx, _accept):
            last_x[0] = x.copy()
            loss_history.append(fx)
            it = len(loss_history)
            probs = forward_pass(qc_data, thetas, x, n_shots, sim)
            er = calculate_error_rate(probs, label_vectors, er)
            loss_file.write(f"{it},{fx},{er}\n")
            loss_file.flush()
            if it % step_show == 0 or it == 1:
                print(
                    f"  Época {it:>4}/{iterations}  |  Pérdida ≈ {fx:.4f}  |  Error rate ≈ {er:.4f}"
                )
                if np.isclose(er, 0.0, atol=ERROR_TOLERANCE):
                    raise EarlyStop()

        spsa = SPSA(
            maxiter=iterations,
            blocking=True,
            learning_rate=make_learning_rate,
            perturbation=make_perturbation,
            callback=spsa_callback,
        )

        try:
            result = spsa.minimize(loss_f, theta_values)
            theta_values = result.x
        except EarlyStop:
            theta_values = last_x[0]
            print(f"Early stop: error rate ≈ {ERROR_TOLERANCE} alcanzado.")
        finally:
            loss_file.close()

        with open(WEIGHTS_FILE, "wb") as f:
            pickle.dump(theta_values, f)
        print(f"Pesos guardados en {WEIGHTS_FILE}")

        # FIN DE LA VERSIÓN UTILIZANDO SPSA DE QISKIT

    else:
        print(f"Cargando pesos desde {WEIGHTS_FILE} ...")
        with open(WEIGHTS_FILE, "rb") as f:
            theta_values = pickle.load(f)

    final_probs = forward_pass(qc_data, thetas, theta_values, n_shots, sim)
    error_rate = calculate_error_rate(final_probs, label_vectors)
    print(f"Error rate final: {error_rate:.4f}")

    plt.plot(loss_history)
    plt.show()
