import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector, Pauli


def scale_data(x_raw):
    for column in range(x_raw.shape[1]):
        mini, maxi = np.min(x_raw[:, column]), np.max(x_raw[:, column])
        x_raw[:, column] = np.pi * (x_raw[:, column] - mini) / (maxi - mini)


def combine_var_skew(data):
    new_col = np.sqrt(data[:, 0] + data[:, 1])
    return np.column_stack((data, new_col))


def build_model_circuit():
    # Ue (data embedding)
    x = ParameterVector("x", 2)
    # Ua (Ansatz)
    θ = ParameterVector("θ", 3)
    qc = QuantumCircuit(1, 1)

    qc.ry(x[1], 0)
    # No se puede empezar por Z, hay que empezar por y o x, recordar esfera Bloch
    qc.rz(x[0], 0)

    qc.rx(θ[0], 0)
    qc.rz(θ[1], 0)
    qc.ry(θ[2], 0)

    qc.measure(0, 0)

    return qc, x, θ


def model(X, θ_values):
    y_hat = []
    sigma_z = Pauli("Z")
    for x_i in X:
        # Asignamos param values to qc
        qc_eval = qc_model.assign_parameters({x: x_i, θ: θ_values})

        # Preguntar por observables
        # get state vector (chatGPT code)
        state = Statevector.from_instruction(
            qc_eval.remove_final_measurements(inplace=False)
        )
        # calculate expectation (chatGPT code)
        exp_z = state.expectation_value(sigma_z).real

        # Apply paper rule to clasify results
        y_hat.append(1 if exp_z >= 0 else 0)
    return np.array(y_hat)


def error(y_true, y_pred):
    return 100 * np.mean(y_true != y_pred)


def error_func(θ_values):
    e = error(y_train, model(x_train, θ_values))
    cost_values.append(e)
    return e


if __name__ == "__main__":
    np.random.seed(42)
    data = np.loadtxt(
        "data_banknote_authentication.txt", delimiter=",", dtype=np.float32
    )
    x_raw = data[:, :-1]
    y = data[:, -1].astype(int)

    scale_data(x_raw)
    data_parsed = combine_var_skew(x_raw)
    X = data_parsed[:, [2, 4]]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    qc_model, x, θ = build_model_circuit()
    simulator = AerSimulator(seed_simulator=42)
    n_shots = 2048
    cost_values = []

    θ_init = np.random.rand(3)
    print("Buscando mejores parámetros...")
    result = minimize(
        fun=error_func, x0=θ_init, method="COBYLA", options={"maxiter": 50}
    )
    θ_best = result.x

    print(f"\nMejores parámetros encontrados: {θ_best}")

    y_pred_test = model(x_test, θ_best)
    test_error = error(y_test, y_pred_test)
    print(f"Error en test: {test_error:.2f}%")

    plt.plot(cost_values)
    plt.xlabel("Iteración")
    plt.ylabel("Error (%)")
    plt.title("Evolución del error")
    plt.show()
