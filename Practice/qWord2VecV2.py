from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def get_entangling_layer(num_qubits):
    """
    Builds the entangling layer U_enta using the circuit-block configuration
    (Sim et al. 2019): forward linear chain CX(0,1), CX(1,2), ..., CX(n-2, n-1)
    followed by circular CX(n-1, 0).
    """
    qc = QuantumCircuit(num_qubits)
    if num_qubits > 1:
        qc.cx(num_qubits - 1, 0)
        for i in range(num_qubits - 1, 0, -1):
            qc.cx(i - 1, i)
    return qc


def build_parameterized_block(num_qubits, num_layers, param_name):
    """
    Builds one parameterized block U(theta) or V(theta) of the QWord2Vec circuit.

    Per the paper formula:
        U(theta) = prod_{l=1}^{L} U_enta * R_Z^{otimes n} * R_Y^{otimes n} * R_X^{otimes n}

    Reading right-to-left (matrix multiplication convention) gives circuit order:
        Per layer: R_X^{otimes n} -> R_Y^{otimes n} -> R_Z^{otimes n} -> U_enta

    Parameters
    ----------
    num_qubits : int   -- number of system qubits (n)
    num_layers : int   -- circuit depth L (not counting U_enta depth)
    param_name : str   -- ParameterVector name (e.g. 'theta_u' or 'theta_v')

    Returns
    -------
    qc    : QuantumCircuit with 3 * num_qubits * num_layers parameters
    theta : ParameterVector of the same length
    """
    num_params = 3 * num_qubits * num_layers
    theta = ParameterVector(param_name, num_params)

    qc = QuantumCircuit(num_qubits)
    idx = 0

    for l in range(num_layers):
        for q in range(num_qubits):          # R_X^{otimes n}
            qc.rx(theta[idx], q); idx += 1
        for q in range(num_qubits):          # R_Y^{otimes n}
            qc.ry(theta[idx], q); idx += 1
        for q in range(num_qubits):          # R_Z^{otimes n}
            qc.rz(theta[idx], q); idx += 1

        qc.compose(get_entangling_layer(num_qubits), inplace=True)

        if l < num_layers - 1:
            qc.barrier()

    return qc, theta


def build_full_qword2vec_circuit(n_qubits, n_embedding, num_layers):
    """
    Builds the full QWord2Vec circuit matching Fig. 1 of the paper:

        |k> -- U(theta_u) -- [reset qubits n_e..n-1] -- V(theta_v) -- |psi>

    After U(theta_u), the non-embedding qubits (n_e to n-1) are reset to |0>,
    keeping only the n_e embedding qubits. V(theta_v) then acts on the full
    n-qubit register. The output |psi> is measured on the embedding qubits
    in the computational basis (Z basis).

    Parameters
    ----------
    n_qubits    : int -- total system qubits (n)
    n_embedding : int -- embedding qubits (n_e), 1 <= n_e <= n
    num_layers  : int -- depth L shared by U and V

    Returns
    -------
    qc      : QuantumCircuit
    theta_u : ParameterVector for U (3 * n_qubits * num_layers parameters)
    theta_v : ParameterVector for V (3 * n_qubits * num_layers parameters)
    """
    u_circuit, theta_u = build_parameterized_block(n_qubits, num_layers, "theta_u")
    v_circuit, theta_v = build_parameterized_block(n_qubits, num_layers, "theta_v")

    qc = QuantumCircuit(n_qubits)

    qc.barrier(label="U(theta_u)")
    qc.compose(u_circuit, inplace=True)

    # Reset non-embedding qubits to |0>; embedding qubits retain U output
    qc.barrier(label="reset")
    for q in range(n_embedding, n_qubits):
        qc.reset(q)

    qc.barrier(label="V(theta_v)")
    qc.compose(v_circuit, inplace=True)

    qc.barrier(label="|psi>")

    return qc, theta_u, theta_v




if __name__ == "__main__":
    from viz import show_circuit

    n_qubits = 4
    n_embedding = 2
    num_layers = 2

    qc, theta_u, theta_v = build_full_qword2vec_circuit(n_qubits, n_embedding, num_layers)
    show_circuit(qc, n_qubits, n_embedding, num_layers, save_path="qword2vec_circuit.png")
