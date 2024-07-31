import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mutual_info_score
from utils import h5py_read


# usually considering main to be discrete bc flavour_label
def calculate_mutual_information(primary, aux, n_bins=10):
    # Discretize the continuous data
    # kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    if not np.issubdtype(aux.dtype, np.integer):
        # aux = kbd.fit_transform(aux.reshape(-1, 1)).ravel()
        aux = np.digitize(aux, bins=np.linspace(np.min(aux), np.max(aux), n_bins))
    # Calculate mutual information
    return mutual_info_score(primary, aux)


def load_hdf5_dataset(file_path, dataset_name):
    with h5py.File(file_path, "r") as f:
        return f[dataset_name][:]


def calculate_pairwise_mi(primary, aux, n_bins=10):
    mi_matrix = np.zeros((primary.shape[1], aux.shape[1]))
    for i in range(primary.shape[1]):
        for j in range(aux.shape[1]):
            mi_matrix[i, j] = calculate_mutual_information(primary[:, i], aux[:, j], n_bins)
    return mi_matrix


def plot_mi(primary, aux, n_bins_values):
    mutual_info_values = []
    for n_bins in n_bins_values:
        mutual_info = calculate_mutual_information(primary, aux, n_bins)
        mutual_info_values.append(mutual_info)

    plt.plot(n_bins_values, mutual_info_values)
    plt.xlabel("Number of Bins")
    plt.ylabel("Mutual Information")
    plt.show()


def main():
    # File paths and dataset names
    # i think only use the test set here
    fname_truth = "/home/xzcappon/phd/datasets/vertexing_120m/output/pp_output_test_ttbar.h5"

    primary_data = h5py_read(fname_truth, "jets", var_name="flavour_label")
    primary_data = np.array(primary_data[:10000])
    # is_light = primary_data == 2
    # is_c = primary_data == 1
    # is_b = primary_data == 0
    aux_label = "pt"

    aux_data = h5py_read(fname_truth, "truth_hadrons", var_name=aux_label)
    aux_data = np.array(aux_data[:10000, 0])
    aux_data = np.nan_to_num(aux_data)

    # had_is_light = df["flavour"] == 0
    # had_is_c = df["flavour"] == 4
    # had_is_b = df["flavour"] == 5

    # Calculate pairwise mutual information between columns
    n_bins = 10_000
    mutual_info = calculate_mutual_information(primary_data, aux_data, n_bins)
    # pairwise_mi = calculate_pairwise_mi(primary_data, aux_data, n_bins)
    print(f"{aux_label} Mutual Information ({n_bins} bins):")
    print(mutual_info)

    n_bins_values = [10, 100, 1000, 10000]
    plot_mi(primary_data, aux_data, n_bins_values)


if __name__ == "__main__":
    main()
