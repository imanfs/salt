import matplotlib.pyplot as plt
import numpy as np
from utils import h5py_read

# ... (keep all your existing functions) ...


def calculate_conditional_probability(primary, aux, n_bins=10):
    """Calculate the conditional probability P(primary|aux).

    Args:
    primary (np.array): The primary variable (assumed to be discrete).
    aux (np.array): The auxiliary variable.
    n_bins (int): Number of bins for discretizing the auxiliary variable if it's continuous.

    Returns
    -------
    np.array: A 2D array where entry [i,j] is P(primary=i | aux=j)
    """
    if not np.issubdtype(aux.dtype, np.integer):
        aux = np.digitize(aux, bins=np.linspace(np.min(aux), np.max(aux), n_bins))

    joint_counts = np.zeros((np.max(primary) + 1, np.max(aux) + 1))
    for p, a in zip(primary, aux, strict=False):
        joint_counts[p, a] += 1

    # Calculate P(primary, aux)
    joint_prob = joint_counts / np.sum(joint_counts)

    # Calculate P(aux)
    aux_prob = np.sum(joint_prob, axis=0)

    # Calculate P(primary|aux) = P(primary, aux) / P(aux)
    return joint_prob / aux_prob[np.newaxis, :]


def plot_conditional_probability(primary, aux, n_bins=10):
    cond_prob = calculate_conditional_probability(primary, aux, n_bins)

    plt.figure(figsize=(10, 8))
    plt.imshow(cond_prob, aspect="auto", cmap="viridis")
    plt.colorbar(label="Probability")
    plt.xlabel("Auxiliary Variable (binned)")
    plt.ylabel("Primary Variable")
    plt.title("Conditional Probability P(Primary|Auxiliary)")
    plt.show()


def main():
    fname_truth = "/home/xzcappon/phd/datasets/vertexing_120m/output/pp_output_test_ttbar.h5"

    primary_data = h5py_read(fname_truth, "jets", var_name="flavour_label")
    primary_data = np.array(primary_data[:10000])

    aux_label = "flavour"
    aux_truth = h5py_read(fname_truth, "truth_hadrons", var_name=aux_label)

    # Calculate and plot conditional probability
    plot_conditional_probability(
        primary_data, aux_truth[:, 0]
    )  # Using the first column of aux_truth

    # ... (keep your existing code) ...


if __name__ == "__main__":
    main()
