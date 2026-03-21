import os
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
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator


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


def calculate_custom_loss(prob_distributions, target_distances, label_vectors, C):

    q_dists = pdist(prob_distributions, metric="euclidean")

    assert len(target_distances) == len(q_dists)
    correlation, _ = pearsonr(q_dists, target_distances)

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
    target_distances = build_target_distances(w2v_embeddings, word_to_id, n_qubits)

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

    iterations = 100
    c_val = 3
    c = 0.1  # constante de perturbación SPSA
    learning_rate = 0.001
    momentum = 0.9
    velocity = np.zeros(len(thetas))
    loss_history = []
    step_show = 20

    def loss_f(param):
        prob_array = forward_pass(qc_data, thetas, param, n_shots, sim)
        return calculate_custom_loss(prob_array, target_distances, label_vectors, c_val)

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

    for it in range(iterations):

        delta = np.random.choice([-1.0, 1.0], size=len(theta_values))

        theta_plus = theta_values + c * delta
        loss_plus = loss_f(theta_plus)

        theta_minus = theta_values - c * delta
        loss_minus = loss_f(theta_minus)

        grad = (loss_plus - loss_minus) / (2.0 * c * delta)

        velocity = momentum * velocity + learning_rate * grad
        theta_values = theta_values - velocity

        current_loss = (loss_plus + loss_minus) / 2.0
        loss_history.append(current_loss)

        if (it + 1) % step_show == 0 or it == 0:
            print(f"  Época {it + 1:>4}/{iterations}  |  Pérdida ≈ {current_loss:.6f}")

    final_probs = forward_pass(qc_data, thetas, theta_values, n_shots, sim)
    error_rate = calculate_error_rate(final_probs, label_vectors)
    print(f"Error rate final: {error_rate:.4f}")

    plt.plot(loss_history)
    plt.show()
