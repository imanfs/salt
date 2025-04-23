import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from puma import HistogramPlot
from utils import load_file_paths

warnings.filterwarnings("ignore")


def valid(objects, prefix, n_test):
    pnull = np.array(objects[prefix + "pnull"])[:n_test]
    return pnull < 0.5


def calc_residuals(x_true, x_pred, norm=True):
    residuals = x_pred - x_true
    if norm:
        residuals = residuals / x_true
    # residuals[np.isnan(residuals)] = 0
    residuals[np.isinf(residuals)] = 0
    return residuals


def format_var_units(var_name):
    var_dict = {
        "deta": r"$\Delta\eta$",
        "dphi": r"$\Delta\phi$",
        "pt": r"$p_{T}$",
        "Lxy": r"$L_{xy}$",
    }
    unit_dict = {"pt": "GeV", "Lxy": r"$mm$"}

    var, unit = var_dict.get(var_name, var_name), unit_dict.get(var_name, "GeV")

    return var, unit


def initialise_histogram(var_name, norm, cut_range=None, weighting=None):
    normed = "Normalised" if norm else "Unnormalised"
    tag = (
        "Gradient-based"
        if weighting == "gradbased"
        else "Loss-based"
        if weighting == "lossbased"
        else "PolyLoss"
        if weighting == "poly"
        else "MaskFormer"
    )
    var, unit = format_var_units(var_name)

    cut_tag = (
        f"{cut_range[0]:.0f} < {var} < {cut_range[1]:.0f} {unit}"
        if cut_range is not None and cut_range[1] != np.inf
        else f"{var} > {cut_range[0]:.0f} {unit}"
        if cut_range is not None
        else var
    )

    return HistogramPlot(
        ylabel="Number of hadrons",
        xlabel=f"{normed} {var} residual ({unit})",
        logy=False,
        # bins=60,  # you can also define an integer number for the number of bins
        norm=False,
        atlas_first_tag="Simulation Internal",
        atlas_second_tag=f"{tag} residuals: {cut_tag}",
        leg_fontsize=8,
        figsize=(6, 5),
        n_ratio_panels=1,
    )


def extract_MF_name(path):
    prefix = "MaskFormer_"
    suffix = "_"
    return path.partition(prefix)[2].partition(suffix)[0]


def residuals_to_csv(label, var, res_mean, res_std, file_name="res_norm.csv"):
    """Append rejection values % increase for u-jets and c-jets to a CSV file."""
    plot_dir = "/home/xucabis2/salt/iman/plots/figs/csv/"
    # Create a DataFrame with the values
    rounding = 4 if "unnorm" in file_name else 2
    data = {
        "Weighting": [label],
        "Regression Variable": [var],
        "Residual Mean": [round(res_mean, rounding)],
        "Residual Standard Deviation": [round(res_std, rounding)],
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
    print(f"Data for {label} appended to {file_name}")


def object_residuals(file_path, fname_truth, var_name, cut_range=None, norm=False):
    file_paths_dict = load_file_paths(file_path)
    fnames_preds = file_paths_dict.copy()  # Create a copy of the original dictionary
    # extract truth hadrons from test file
    h5truth = h5py.File(fname_truth, "r")
    truth_hadrons = h5truth["truth_hadrons"]
    n_test = 500_000

    normed = "norm" if norm else "unnorm"

    # loop through prediction files
    for label, fname_preds in fnames_preds.items():
        # extract regression predictions from prediction file
        mf_name = extract_MF_name(fname_preds)
        h5preds = h5py.File(fname_preds, "r")

        objects = h5preds["truth_hadrons"]  # hadrons

        # get truth values and regression predictions for chosen variable (pT, Lxy, etc)
        prefix = f"MaskFormer_{mf_name}_"
        preds = objects[prefix + "regression_" + var_name][:n_test]
        truth = truth_hadrons[var_name][:n_test]
        # get valid mask (predict null with p < 0.5) and filter for valid predictions
        valid_mask = valid(objects, prefix, n_test) & truth_hadrons["valid"][:n_test]
        preds, truth = preds[valid_mask], truth[valid_mask]

        # if no cut is specified, use the full range of the truth values
        cut = cut_range or (np.nanmin(truth), np.nanmax(truth))

        # apply range cut if necessary and calculate residuals
        mask = (truth >= cut[0]) & (truth <= cut[1])
        preds_cut, truth_cut = preds[mask], truth[mask]

        residuals = calc_residuals(truth_cut, preds_cut, norm=norm)
        mean, std = np.mean(residuals), np.std(residuals)

        # save residuals to csv
        residuals_to_csv(label, var_name, mean, std, file_name=f"res_{normed}.csv")


fname_truth = "/home/xzcappon/phd/datasets/vertexing_120m/output/pp_output_test_ttbar.h5"

grad_zprime = "/home/xucabis2/salt/iman/file_paths/files_gradbased_zprime.txt"
loss_zprime = "/home/xucabis2/salt/iman/file_paths/files_lossbased_zprime.txt"
poly_zprime = "/home/xucabis2/salt/iman/file_paths/files_polyloss_zprime.txt"
base_zprime = "/home/xucabis2/salt/iman/file_paths/files_baselines_zprime.txt"

grad_ttbar = "/home/xucabis2/salt/iman/file_paths/files_gradbased_ttbar.txt"
loss_ttbar = "/home/xucabis2/salt/iman/file_paths/files_lossbased_ttbar.txt"
poly_ttbar = "/home/xucabis2/salt/iman/file_paths/files_polyloss_ttbar.txt"
base_ttbar = "/home/xucabis2/salt/iman/file_paths/files_baselines_ttbar.txt"

files_all_ttbar = "/home/xucabis2/salt/iman/file_paths/files_all_ttbar.txt"

# Lxy cuts
cuts = [(0, 0.1), (0.1, 1), (1, 5), (5, np.inf)]

variables = ["pt", "deta", "dphi", "Lxy", "mass"]

# # raw plots with no cuts - sense check
# for variable in variables:
#     object_residuals(fnames_preds, fname_truth, variable, norm=True)
#     object_residuals(fnames_preds, fname_truth, variable, norm=False)

norm_file = "/home/xucabis2/salt/iman/plots/figs/csv/res_norm.csv"
unnorm_file = "/home/xucabis2/salt/iman/plots/figs/csv/res_unnorm.csv"

with open(norm_file, "w") as file:
    pass

with open(unnorm_file, "w") as file:
    pass

files_ttbar = [loss_ttbar, grad_ttbar, poly_ttbar]

file_path = files_all_ttbar

object_residuals(file_path, fname_truth, "dphi", norm=True)
object_residuals(file_path, fname_truth, "dphi", norm=False)

object_residuals(file_path, fname_truth, "deta", norm=True)
object_residuals(file_path, fname_truth, "deta", norm=False)

object_residuals(file_path, fname_truth, "Lxy", (1, np.inf), norm=True)
object_residuals(file_path, fname_truth, "Lxy", (0, 1), norm=False)

# object_residuals(fnames_preds, fname_truth, "mass", plot_range=(-1, 0.5), norm=True)
# object_residuals(fnames_preds, fname_truth, "mass", plot_range=(-750, 500), norm=False)

object_residuals(file_path, fname_truth, "pt", norm=True)

object_residuals(file_path, fname_truth, "mass", norm=True)
object_residuals(file_path, fname_truth, "mass", norm=False)
