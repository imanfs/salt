import warnings

import h5py
import numpy as np
from puma import Histogram, HistogramPlot
from utils import load_file_paths

warnings.filterwarnings("ignore")


def valid(objects, n_test):
    class_probs = np.array(objects["object_class_probs"])[:n_test]
    class_probs_3d = class_probs.view((np.float32, 3))
    return class_probs_3d[:, :, -1] < 0.5


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


def initialise_histogram(var_name, norm, cut_range=None):
    normed = "Normalised" if norm else "Unnormalised"
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
        atlas_second_tag=f"MaskFormer residuals: {cut_tag}",
        leg_fontsize=8,
        figsize=(6, 5),
        n_ratio_panels=1,
    )


def extract_MF_name(path):
    prefix = "MaskFormer_"
    suffix = "_"
    return path.partition(prefix)[2].partition(suffix)[0]


def object_residuals(file_path, fname_truth, var_name, cut_range=None, plot_range=None, norm=False):
    # extract truth hadrons from test file
    h5truth = h5py.File(fname_truth, "r")
    truth_hadrons = h5truth["truth_hadrons"]
    n_test = 500_000

    # initialise histogram plot
    plot_histo = initialise_histogram(var_name, norm, cut_range)

    file_paths_dict = load_file_paths(file_path)
    fnames_preds = file_paths_dict.copy()  # Create a copy of the original dictionary
    # loop through prediction files
    for label, fname_preds in fnames_preds.items():
        # extract regression predictions from prediction file
        h5preds = h5py.File(fname_preds, "r")

        objects = h5preds["objects"]  # hadrons
        reg_preds = objects["regression"]  # truth hadron regression predictions

        # get truth values and regression predictions for chosen variable (pT, Lxy, etc)
        preds = reg_preds["regression_" + var_name][:n_test]
        truth = truth_hadrons[var_name][:n_test]
        # get valid mask (predict null with p < 0.5) and filter for valid predictions
        valid_mask = valid(objects, n_test) & truth_hadrons["valid"][:n_test]
        preds, truth = preds[valid_mask], truth[valid_mask]

        # if no cut is specified, use the full range of the truth values
        cut = cut_range or (np.nanmin(truth), np.nanmax(truth))

        # apply range cut if necessary and calculate residuals
        mask = (truth >= cut[0]) & (truth <= cut[1])
        preds_cut, truth_cut = preds[mask], truth[mask]

        residuals = calc_residuals(truth_cut, preds_cut, norm=norm)
        mean, std = np.mean(residuals), np.std(residuals)

        # if no plot_range is specified, use the full range of residuals
        plot_rng = plot_range or (np.nanmin(residuals), np.nanmax(residuals))
        res_mask = (residuals >= plot_rng[0]) & (residuals <= plot_rng[1])
        residuals = residuals[res_mask]

        # plot naming stuff
        mf_name = extract_MF_name(fname_preds)
        # tag = "GLS" if fnames_preds == fname_gls_old else tag
        name_label = f"{label}: " r"$\mu =$ " f"{mean:.2f}, " r"$\sigma =$ " f"{std:.2f}"
        ref = mf_name == "default"  # reference data will be default weighting scheme

        # create histogram and add to histogram plot
        hist = Histogram(residuals, label=name_label)
        plot_histo.add(hist, reference=ref)

    # more plot naming stuff
    cut_name = (
        "all"
        if cut == (np.nanmin(truth), np.nanmax(truth))
        else f"{cut[0]:.1f}"
        if cut[1] == np.inf
        else f"{cut[0]:.1f}_{cut[1]:.1f}"
    )
    weighting = (
        "lossbased"
        if "lossbased" in file_path
        else "gradbased"
        if "gradbased" in file_path
        else "polyloss"
        if "polyloss" in file_path
        else "undefined"
    )
    normed = "norm" if norm else "unnorm"
    plot_dir = "/home/xucabis2/salt/iman/plots/figs"
    plot_name = f"{plot_dir}/res_{weighting}_{var_name}_{cut_name}_{normed}.png"

    # draw and save plot
    plot_histo.draw()
    plot_histo.savefig(plot_name, transparent=False)
    print("Saving to ", plot_name)


fname_truth = "/home/xzcappon/phd/datasets/vertexing_120m/output/pp_output_test_ttbar.h5"

file_path = "/home/xucabis2/salt/iman/files_gradbased_ttbar.txt"

# Lxy cuts
cuts = [(0, 0.1), (0.1, 1), (1, 5), (5, np.inf)]

variables = ["pt", "deta", "dphi", "Lxy", "mass"]

# # raw plots with no cuts - sense check
# for variable in variables:
#     object_residuals(fnames_preds, fname_truth, variable, norm=True)
#     object_residuals(fnames_preds, fname_truth, variable, norm=False)

object_residuals(file_path, fname_truth, "dphi", plot_range=(-4, 5), norm=True)
object_residuals(file_path, fname_truth, "dphi", plot_range=(-0.1, 0.1), norm=False)

object_residuals(file_path, fname_truth, "deta", plot_range=(-4, 5), norm=True)
object_residuals(file_path, fname_truth, "deta", plot_range=(-0.1, 0.1), norm=False)

object_residuals(file_path, fname_truth, "Lxy", (1, np.inf), plot_range=(-1, 2), norm=True)
object_residuals(file_path, fname_truth, "Lxy", (0, 1), plot_range=(-6, 6), norm=False)

# object_residuals(fnames_preds, fname_truth, "mass", plot_range=(-1, 0.5), norm=True)
# object_residuals(fnames_preds, fname_truth, "mass", plot_range=(-750, 500), norm=False)

object_residuals(file_path, fname_truth, "pt", plot_range=(-2, 4), norm=True)

object_residuals(file_path, fname_truth, "mass", plot_range=(-1, 2.5), norm=True)
object_residuals(file_path, fname_truth, "mass", plot_range=(-4200.0, 4700.0), norm=False)
