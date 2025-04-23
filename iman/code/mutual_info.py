from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mutual_info_score
from utils import h5py_read


def load_hdf5_dataset(file_path, dataset_name):
    with h5py.File(file_path, "r") as f:
        return f[dataset_name][:]


def flavour_class(objects):
    class_probs = objects["object_class_probs"]
    class_probs = np.array(class_probs)
    class_probs_3d = class_probs.view((np.float32, 3))

    # Initialize the new column with zeros
    flavour = 0 * np.ones(class_probs_3d.shape[:2], dtype=int)

    # Assign values based on conditions
    flavour[class_probs_3d[:, :, 0] > 0.5] = 5
    flavour[class_probs_3d[:, :, 1] > 0.5] = 4
    flavour[class_probs_3d[:, :, 2] > 0.5] = -1

    return flavour


def calculate_mutual_information(primary, aux, n_bins=10):
    # kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    if not np.issubdtype(aux.dtype, np.integer):
        # aux = kbd.fit_transform(aux.reshape(-1, 1)).ravel()
        aux = np.digitize(aux, bins=np.linspace(np.min(aux), np.max(aux), n_bins))
    return mutual_info_score(primary, aux)


def calculate_pairwise_mi(primary, aux, n_bins=10):
    mi_matrix = np.zeros((primary.shape[1], aux.shape[1]))
    for i in range(primary.shape[1]):
        for j in range(aux.shape[1]):
            mi_matrix[i, j] = calculate_mutual_information(primary[:, i], aux[:, j], n_bins)
    return mi_matrix


def plot_mi_bins(primary, aux, n_bins_values):
    mutual_info_values = []
    for n_bins in n_bins_values:
        mutual_info = calculate_mutual_information(primary, aux, n_bins)
        mutual_info_values.append(mutual_info)

    plt.plot(n_bins_values, mutual_info_values, marker=".")
    plt.xlabel("Number of Bins")
    plt.ylabel("Mutual Information")
    plt.show()


def plot_mi_objects(primary, aux_truth, aux_preds1, aux_preds2, n_objects=5):
    n_samples = primary.shape[0]

    def calc_mi_aux(aux):
        return [
            calculate_mutual_information(primary, np.nan_to_num(aux[:n_samples, obj]))
            for obj in range(n_objects)
        ]

    mutual_info_t = calc_mi_aux(aux_truth)
    mutual_info_p1 = calc_mi_aux(aux_preds1)
    mutual_info_p2 = calc_mi_aux(aux_preds2)
    timestamp = datetime.now().strftime("%m%d")

    plt.plot(np.arange(1, n_objects + 1), mutual_info_t, marker=".", label="Truth")
    plt.plot(np.arange(1, n_objects + 1), mutual_info_p1, marker=".", label="Default weighting")
    plt.plot(np.arange(1, n_objects + 1), mutual_info_p2, marker=".", label="GLS")
    plt.title("Mutual Information between jet flavour and truth hadron flavour")
    plt.xlabel("Truth hadrons (in order of pT)")
    plt.ylabel("MI")
    plt.xticks(ticks=[1, 2, 3, 4, 5])
    plt.legend()
    plot_dir = "/home/xucabis2/salt/iman/plots/mutual_info"
    plot_name = f"{plot_dir}/mi_GLS_{timestamp}.png"
    print("Saving to ", plot_name)
    plt.savefig(plot_name, transparent=False)


def main():
    fname_truth = "/home/xzcappon/phd/datasets/vertexing_120m/output/pp_output_test_ttbar.h5"

    # usually considering primary task to be discrete bc flavour_label
    primary_data = h5py_read(fname_truth, "jets", var_name="flavour_label")
    primary_data = np.array(primary_data[:30000])
    # is_light, is_c, is_b = primary_data == 2, primary_data == 1, primary_data == 0
    # had_is_light, had_is_c, had_is_b = df["flavour"] == 0, df["flavour"] == 4, df["flavour"] == 5

    aux_label = "flavour"
    aux_truth = h5py_read(fname_truth, "truth_hadrons", var_name=aux_label)

    fname_default = (
        "/home/xucabis2/salt/logs/MaskFormer_default_20240724-T112538/"
        "ckpts/epoch=019-val_loss=0.65355__test_ttbar.h5"
    )
    fname_gls = (
        "/home/xucabis2/salt/logs/MaskFormer_GLS_20240730-T002427/ckpts/"
        "epoch=018-val_loss=0.65171__test_ttbar.h5"
    )
    objs = h5py.File(fname_default, "r")["objects"]
    def_preds = flavour_class(objs)

    objs = h5py.File(fname_gls, "r")["objects"]
    gls_preds = flavour_class(objs)
    plot_mi_objects(primary_data, aux_truth, def_preds, gls_preds, aux_truth.shape[1])


if __name__ == "__main__":
    main()
