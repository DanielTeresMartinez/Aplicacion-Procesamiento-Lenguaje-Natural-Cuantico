import matplotlib.pyplot as plt


def show_circuit(qc, n_qubits, n_embedding, num_layers, save_path=None):
    """
    Draws and displays a QWord2Vec circuit.

    Parameters
    ----------
    qc          : QuantumCircuit -- circuit to draw
    n_qubits    : int            -- total system qubits (n)
    n_embedding : int            -- embedding qubits (n_e)
    num_layers  : int            -- circuit depth L
    save_path   : str | None     -- if given, saves the figure to this path
    """
    fold = max(1, qc.depth() // 2 + 1)
    fig = qc.draw(output="mpl", fold=fold)
    fig.suptitle(
        f"QWord2Vec Circuit  [U(theta_u) -> reset -> V(theta_v)]\n"
        f"n={n_qubits},  ne={n_embedding},  L={num_layers}",
        fontsize=13,
        fontweight="bold",
    )
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Circuit saved to: {save_path}")
    plt.show()
