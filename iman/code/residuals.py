import warnings
from datetime import datetime

import h5py
import numpy as np
from puma import Histogram, HistogramPlot

warnings.filterwarnings("ignore")


def valid(objects):
    class_probs = np.array(objects["object_class_probs"])
    class_probs_3d = class_probs.view((np.float32, 3))
    return class_probs_3d[:, :, -1] < 0.5


def calc_residuals(x_true, x_pred, norm=True):
    residuals = x_pred - x_true
    if norm:
        residuals = residuals / x_true
    residuals[abs(residuals) > 1e4] = 0
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
        xlabel=f"{normed} {var} residual",
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


def object_residuals(
    fnames_preds, fname_truth, var_name, cut_range=None, plot_range=None, norm=False
):
    # extract truth hadrons from test file
    h5truth = h5py.File(fname_truth, "r")
    truth_hadrons = h5truth["truth_hadrons"]

    # initialise histogram plot
    plot_histo = initialise_histogram(var_name, norm, cut_range)

    # loop through prediction files
    for fname_preds in fnames_preds:
        # extract regression predictions from prediction file
        h5preds = h5py.File(fname_preds, "r")

        n_test = h5preds["tracks"].shape[0]
        objects = h5preds["objects"]  # hadrons
        reg_preds = objects["regression"]  # truth hadron regression predictions

        # get truth values and regression predictions for chosen variable (pT, Lxy, etc)
        preds = reg_preds["regression_" + var_name]
        truth = truth_hadrons[var_name][:n_test]

        # get valid mask (predict null with p < 0.5) and filter for valid predictions
        valid_mask = valid(objects) & truth_hadrons["valid"][:n_test]
        preds, truth = preds[valid_mask], truth[valid_mask]

        # if no cut is specified, use the full range of the truth values
        cut = cut_range or (np.nanmin(truth), np.nanmax(truth))

        # apply range cut if necessary and calculate residuals
        mask = (truth > cut[0]) & (truth < cut[1])
        preds_cut, truth_cut = preds[mask], truth[mask]

        residuals = calc_residuals(truth_cut, preds_cut, norm=norm)
        mean, std = np.mean(residuals), np.std(residuals)

        # if no plot_range is specified, use the full range of residuals
        plot_rng = plot_range or (np.nanmin(residuals), np.nanmax(residuals))
        res_mask = (residuals > plot_rng[0]) & (residuals < plot_rng[1])
        residuals = residuals[res_mask]

        # plot naming stuff
        tag = extract_MF_name(fname_preds)
        label = f"{tag}: " r"$\mu =$ " f"{mean:.2f}, " r"$\sigma =$ " f"{std:.2f}"
        ref = tag == "default"  # reference data will be default weighting scheme

        # create histogram and add to histogram plot
        hist = Histogram(residuals, label=label)
        plot_histo.add(hist, reference=ref)

    # more plot naming stuff
    cut_name = (
        "all"
        if cut == (np.nanmin(truth), np.nanmax(truth))
        else f"{cut[0]:.1f}"
        if cut[1] == np.inf
        else f"{cut[0]:.1f}{cut[1]:.1f}"
    )

    normed = "norm" if norm else "unnorm"
    plot_dir = "/home/xucabis2/salt/iman/plots/Aug1"
    timestamp = datetime.now().strftime("%m%d")
    plot_name = f"{plot_dir}/{normed}/res_{tag}_{var_name}_{cut_name}_{normed}_{timestamp}.png"

    # draw and save plot
    plot_histo.draw()
    print("Saving to ", plot_name)
    plot_histo.savefig(plot_name, transparent=False)


fname_arxiv = (
    "/home/xucabis2/salt/logs/MaskFormer_arxiv_20240724-T114414/"
    "ckpts/epoch=019-val_loss=0.65962__test_ttbar.h5"
)
fname_default = (
    "/home/xucabis2/salt/logs/MaskFormer_default_20240724-T112538/"
    "ckpts/epoch=019-val_loss=0.65355__test_ttbar.h5"
)

fname_equal = (
    "/home/xucabis2/salt/logs/MaskFormer_equal_20240724-T114419/ckpts/"
    "epoch=019-val_loss=0.64428__test_ttbar.h5"
)

fname_gls = (
    "/home/xucabis2/salt/logs/MaskFormer_GLS_20240730-T002427/ckpts/"
    "epoch=018-val_loss=0.65171__test_ttbar.h5"
)

fname_dwa = (
    "/home/xucabis2/salt/logs/MaskFormer_DWA_20240806-T121649/"
    "ckpts/epoch=019-val_loss=0.64425__test_ttbar.h5"
)

fnames_preds = [fname_default, fname_gls, fname_dwa]

fname_truth = "/home/xzcappon/phd/datasets/vertexing_120m/output/pp_output_test_ttbar.h5"

# Lxy cuts
cuts = [(0, 0.1), (0.1, 1), (1, 5), (5, np.inf)]

variables = ["pt", "deta", "dphi", "Lxy", "mass"]

# # raw plots with no cuts - sense check
# for variable in variables:
#     object_residuals(fnames_preds, fname_truth, variable, norm=True)
#     object_residuals(fnames_preds, fname_truth, variable, norm=False)

object_residuals(fnames_preds, fname_truth, "dphi", plot_range=(-4, 5), norm=True)
object_residuals(fnames_preds, fname_truth, "dphi", plot_range=(-0.1, 0.1), norm=False)

object_residuals(fnames_preds, fname_truth, "deta", plot_range=(-4, 5), norm=True)
object_residuals(fnames_preds, fname_truth, "deta", plot_range=(-0.1, 0.1), norm=False)

object_residuals(fnames_preds, fname_truth, "Lxy", (1, np.inf), plot_range=(-1, 2), norm=True)
object_residuals(fnames_preds, fname_truth, "Lxy", (0, 1), plot_range=(-6, 6), norm=False)

object_residuals(fnames_preds, fname_truth, "mass", plot_range=(-1, 0.5), norm=True)
object_residuals(fnames_preds, fname_truth, "mass", plot_range=(-750, 500), norm=False)

object_residuals(fnames_preds, fname_truth, "pt", plot_range=(-2, 4), norm=True)
# object_residuals(fnames_preds, fname_truth, "pt", plot_range=(-0.1, 0.2), norm=False)
