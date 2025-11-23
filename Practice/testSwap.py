from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.visualization import plot_bloch_multivector
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import numpy as np


def format_param(theta):
    # Si es exactamente pi, mostrar "π"
    if np.isclose(theta, np.pi):
        return "π"
    # Si es múltiplo de pi/2
    elif np.isclose(theta, np.pi / 2):
        return "π/2"
    elif np.isclose(theta, np.pi / 4):
        return "π/4"
    # Si es 1/sqrt(2) aproximado
    elif np.isclose(theta, 1 / np.sqrt(2)):
        return "1/√2"
    else:
        return f"{theta:.4g}"


def simulate_efficient(qc, theta_vector, theta_values_list, shots=2048):
    sim = AerSimulator()

    # Crear todos los cirucitos y asignar los valores de los parámetros
    circuits = [
        qc.assign_parameters({theta[i]: vals[i] for i in range(4)})
        for vals in theta_values_list
    ]

    qcts = transpile(circuits, sim)
    results = sim.run(qcts, shots=shots).result()

    # Si aparece más veces 0, significa que hay más similitud entre q1 y q2.
    # Si aparece más veces 1, significa que hay una mayor distancia entre q1 y q2.
    for i, circuit in enumerate(circuits):
        counts = results.get_counts(circuit)
        formatted_theta = [format_param(t) for t in theta_values_list[i]]
        print(f"Parámetros: {formatted_theta}, counts: {counts}")


def simulate_and_show_bloch(qc, theta_vector, theta_values_list):
    sim = AerSimulator(method="statevector")

    for vals in theta_values_list:
        circ = qc.remove_final_measurements(inplace=False)
        circ = circ.assign_parameters(
            {theta_vector[i]: vals[i] for i in range(len(theta_vector))}
        )
        circ.save_statevector()

        sv = sim.run(transpile(circ, sim)).result().get_statevector()

        formatted_theta = [format_param(t) for t in vals]
        print(f"Parámetros: {formatted_theta}")

        plot_bloch_multivector(sv, title="Bloch Spheres Q1 y Q2")
        plt.show()


if __name__ == "__main__":
    show_bloch = True

    qc = QuantumCircuit(3, 1)
    theta = ParameterVector("θ", 4)

    theta_values_list = [
        [0, 0, 0, 0],  # |0>|0>
        [0, 0, np.pi, 0],  # |0>|1>
        [np.pi, 0, 0, 0],  # |1>|0>
        [np.pi, 0, np.pi, 0],  # |1>|1>
        [np.pi / 2, 0, np.pi / 2, 0],  # |+>|+>
    ]

    qc.ry(theta[0], 1)
    qc.rz(theta[1], 1)
    qc.ry(theta[2], 2)
    qc.rz(theta[3], 2)

    qc.h(0)
    qc.cswap(0, 1, 2)
    qc.h(0)
    qc.measure(0, 0)
    qc.draw(output="mpl")
    plt.show()

    if show_bloch:
        simulate_and_show_bloch(qc, theta, theta_values_list)
    else:
        simulate_efficient(qc, theta, theta_values_list)
