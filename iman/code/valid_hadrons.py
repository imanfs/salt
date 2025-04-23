import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from utils import load_file_paths

warnings.filterwarnings("ignore")


def valid_to_csv(label, tpr, fpr, fnr, tnr, file_name="valid.csv"):
    """Append rejection values % increase for u-jets and c-jets to a CSV file."""
    plot_dir = "/home/xucabis2/salt/iman/plots/figs/"
    # Create a DataFrame with the values
    data = {
        "Label": [label],
        "TPR (%)": [round(tpr, 2)],
        "FPR (%)": [round(fpr, 2)],
        "FNR (%)": [round(fnr, 2)],
        "TNR (%)": [round(tnr, 2)],
    }

    df = pd.DataFrame(data)

    file_path = Path(plot_dir) / file_name
    write_header = True

    if Path.exists(file_path):
        with open(file_path) as f:
            first_line = f.readline().strip()
            if first_line:
                write_header = False

    df.to_csv(file_path, mode="a", header=write_header, index=False)
    print(f"Data for label {label} appended to {file_name}")


def correct_to_csv(label, percent, percent_b, percent_c, file_name="hadrons_correct.csv"):
    """Append rejection values % increase for u-jets and c-jets to a CSV file."""
    plot_dir = "/home/xucabis2/salt/iman/plots/figs/csv/"
    # Create a DataFrame with the values
    data = {
        "Label": [label],
        "Correct hadron predictions (%)": [round(percent, 2)],
        "Correct b hadron predictions (%)": [round(percent_b, 2)],
        "Correct c hadron predictions (%)": [round(percent_c, 2)],
    }

    df = pd.DataFrame(data)

    file_path = Path(plot_dir) / file_name
    write_header = True

    if Path.exists(file_path):
        with open(file_path) as f:
            first_line = f.readline().strip()
            if first_line:
                write_header = False

    df.to_csv(file_path, mode="a", header=write_header, index=False)
    print(f"Data for label {label} appended to {file_name}")


def valid(objects, prefix):
    pnull = np.array(objects[prefix + "pnull"])
    return pnull < 0.5


def hadron_flavour(objects, prefix):
    pnull = np.array(objects[prefix + "pnull"])
    pb = np.array(objects[prefix + "pb"])
    pc = np.array(objects[prefix + "pc"])
    print(pb.shape)

    # Initialize the new column with zeros
    flavour = -1 * np.ones(pb.shape, dtype=int)

    # Assign values based on conditions
    flavour[pb > 0.5] = 5
    flavour[pc > 0.5] = 4
    flavour[pnull > 0.5] = -1
    return flavour


def extract_MF_name(path):
    prefix = "MaskFormer_"
    suffix = "_"
    return path.partition(prefix)[2].partition(suffix)[0]


def valid_objects(file_path, fname_truth):
    # extract truth hadrons from test file
    h5truth = h5py.File(fname_truth, "r")
    truth_hadrons = h5truth["truth_hadrons"]
    n_test = 500_000
    # initialise histogram plot
    file_paths_dict = load_file_paths(file_path)
    fnames_preds = file_paths_dict.copy()  # Create a copy of the original dictionary
    # loop through prediction files
    for label, fname_preds in fnames_preds.items():
        # extract regression predictions from prediction file
        h5preds = h5py.File(fname_preds, "r")
        mf_name = extract_MF_name(fname_preds)
        prefix = f"MaskFormer_{mf_name}_"

        n_test = h5preds["tracks"].shape[0]
        objects = h5preds["truth_hadrons"]  # hadrons
        truth = truth_hadrons[:n_test]

        # get valid mask (predict null with p < 0.5) and filter for valid predictions
        tp_mask = valid(objects, prefix) & truth["valid"]
        tn_mask = ~valid(objects, prefix) & ~truth["valid"]
        fp_mask = valid(objects, prefix) & ~truth["valid"]
        fn_mask = ~valid(objects, prefix) & truth["valid"]

        truth_valid = truth["valid"]
        truth_invalid = ~truth["valid"]

        tpr = (np.sum(tp_mask) / np.sum(truth_valid)) * 100
        tnr = (np.sum(tn_mask) / np.sum(truth_invalid)) * 100
        fpr = (np.sum(fp_mask) / np.sum(truth_invalid)) * 100
        fnr = (np.sum(fn_mask) / np.sum(truth_valid)) * 100

        print(f"True Positive Rate: {tpr}")
        print(f"True Negative Rate: {tnr}")
        print(f"False Positive Rate: {fpr}")
        print(f"False Negative Rate: {fnr}")

        valid_to_csv(label, tpr, fpr, fnr, tnr, file_name="hadrons_valid.csv")
        # Calculate the percent of correctly calculated hadrons
        flav_pred = hadron_flavour(objects, prefix)
        flav_truth = truth["flavour"]

        correct_mask = flav_pred == flav_truth
        percent_correct = (np.sum(correct_mask) / flav_truth.size) * 100
        print(f"Percent of correctly classified hadrons: {percent_correct}")

        valid_correct_mask = truth["valid"] & (flav_pred == flav_truth)
        percent_valid_corr = (np.sum(valid_correct_mask) / np.sum(truth["valid"])) * 100
        print(f"Percent of correctly classified valid hadrons: {percent_valid_corr}")

        # Calculate the number of correctly classified hadrons as class 5
        b_mask = truth["valid"] & (flav_pred == 5) & (flav_truth == 5)
        percent_b = (np.sum(b_mask) / np.sum(flav_truth == 5)) * 100
        print(f"Percent of correctly classified b hadrons: {percent_b}")

        # Calculate the number of correctly classified hadrons as class 4
        c_mask = truth["valid"] & (flav_pred == 4) & (flav_truth == 4)
        percent_c = (np.sum(c_mask) / np.sum(flav_truth == 4)) * 100
        print(f"Percent of correctly classified c hadrons: {percent_c}")

        correct_to_csv(
            label, percent_valid_corr, percent_b, percent_c, file_name="hadrons_correct.csv"
        )


fname_truth = "/home/xzcappon/phd/datasets/vertexing_120m/output/pp_output_test_ttbar.h5"

file_path = "/home/xucabis2/salt/iman/file_paths/files_all_ttbar.txt"

valid_file = "/home/xucabis2/salt/iman/plots/figs/csv/hadrons_valid.csv"  # Output CSV file
correct_file = "/home/xucabis2/salt/iman/plots/figs/csv/hadrons_correct.csv"  # Output CSV file

with open(valid_file, "w") as file:
    pass

with open(correct_file, "w") as file:
    pass

valid_objects(file_path, fname_truth)
