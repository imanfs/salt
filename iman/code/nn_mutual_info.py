import h5py
import numpy as np
from scipy.special import psi
from utils import h5py_read


def load_hdf5_dataset(file_path, dataset_name):
    with h5py.File(file_path, "r") as f:
        return f[dataset_name][:]


def mutual_info_nn(d, c, k=3, base=np.e):
    """Estimates the mutual information between a discrete vector 'd' and a continuous vector 'c'
    using nearest-neighbor statistics.

    Parameters
    ----------
    d: np.array
        Discrete data matrix (each column is a sample)
    c: np.array
        Continuous data matrix (each column is a sample)
    k: int
        Number of nearest neighbors to use (default is 3)
    base: float
        Logarithm base (default is natural logarithm)

    Returns
    -------
    float: Mutual information estimate
    np.array: Volumes
    """
    first_symbol = []
    symbol_IDs = np.zeros(d.shape[1], dtype=int)
    c_split = {}
    cs_indices = {}
    num_d_symbols = 0

    # Bin the continuous data 'c' according to the discrete symbols 'd'
    for c1 in range(d.shape[1]):
        symbol_IDs[c1] = num_d_symbols + 1
        for c2 in range(num_d_symbols):
            if np.array_equal(d[:, c1], d[:, first_symbol[c2]]):
                symbol_IDs[c1] = c2 + 1
                break
        if symbol_IDs[c1] > num_d_symbols:
            num_d_symbols += 1
            first_symbol.append(c1)
            c_split[num_d_symbols] = []
            cs_indices[num_d_symbols] = []
        c_split[symbol_IDs[c1]].append(c[:, c1])
        cs_indices[symbol_IDs[c1]].append(c1)

    # Convert lists to arrays
    for key in c_split:
        c_split[key] = np.array(c_split[key]).T

    # Compute the neighbor statistic for each data pair (c, d) using the binned c_split list
    m_tot = 0
    av_psi_Nd = 0
    V = np.zeros(d.shape[1])
    all_c_distances = np.zeros(c.shape[1])
    psi_ks = 0

    for c_bin in range(1, num_d_symbols + 1):
        one_k = min(k, c_split[c_bin].shape[1] - 1)

        if one_k > 0:
            c_distances = np.zeros(c_split[c_bin].shape[1])
            for pivot in range(c_split[c_bin].shape[1]):
                for cv in range(c_split[c_bin].shape[1]):
                    vector_diff = c_split[c_bin][:, cv] - c_split[c_bin][:, pivot]
                    c_distances[cv] = np.linalg.norm(vector_diff)
                sorted_distances = np.sort(c_distances)
                eps_over_2 = sorted_distances[one_k]  # don't count pivot

                # Count the number of total samples within volume using all samples
                for cv in range(c.shape[1]):
                    vector_diff = c[:, cv] - c_split[c_bin][:, pivot]
                    all_c_distances[cv] = np.linalg.norm(vector_diff)
                m = max(np.sum(all_c_distances <= eps_over_2) - 1, 0)  # don't count pivot

                m_tot += psi(m)
                V[cs_indices[c_bin][pivot]] = (2 * eps_over_2) ** d.shape[0]

        else:
            m_tot += psi(num_d_symbols * 2)

        p_d = c_split[c_bin].shape[1] / d.shape[1]
        av_psi_Nd += p_d * psi(p_d * d.shape[1])
        psi_ks += p_d * psi(max(one_k, 1))

    f = (psi(d.shape[1]) - av_psi_Nd + psi_ks - m_tot / d.shape[1]) / np.log(base)

    return f, V


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
    mi, _ = mutual_info_nn(primary_data, aux_data)
    print(f"Mutual Information: {mi}")


if __name__ == "__main__":
    main()
