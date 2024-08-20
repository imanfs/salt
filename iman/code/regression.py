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


def initialise_histogram(var_name):
    var, unit = format_var_units(var_name)

    return HistogramPlot(
        ylabel="Number of hadrons",
        xlabel=f"{var} predictions ({unit})",
        logy=False,
        bins=60,  # you can also define an integer number for the number of bins
        norm=False,
        atlas_first_tag="Simulation Internal",
        atlas_second_tag="Mass predictions for 549,218 valid truth hadrons",
        leg_fontsize=8,
        figsize=(6, 5),
        n_ratio_panels=1,
    )


def extract_MF_name(path):
    prefix = "MaskFormer_"
    suffix = "_"
    return path.partition(prefix)[2].partition(suffix)[0]


def object_residuals(fnames_preds, fname_truth, var_name, cut_range=None, plot_range=None):
    # extract truth hadrons from test file
    h5truth = h5py.File(fname_truth, "r")
    truth_hadrons = h5truth["truth_hadrons"]

    # initialise histogram plot
    plot_histo = initialise_histogram(var_name, cut_range)

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
        valid_preds, valid_truth = preds[valid_mask], truth[valid_mask]

        # if no cut is specified, use the full range of the truth values
        cut = cut_range or (np.nanmin(valid_truth), np.nanmax(valid_truth))

        # apply range cut if necessary and calculate residuals
        # if no cut is specified truth_cut remains the same as valid_truth
        mask = (valid_truth >= cut[0]) & (valid_truth <= cut[1])
        preds_cut, truth_cut = valid_preds[mask], valid_truth[mask]

        mean, std = np.mean(preds_cut), np.std(preds_cut)

        # if no plot_range is specified, use the full range of residuals
        plot_rng = plot_range or (np.nanmin(preds_cut), np.nanmax(preds_cut))
        res_mask = (preds_cut >= plot_rng[0]) & (preds_cut <= plot_rng[1])
        preds_cut = preds_cut[res_mask]

        # plot naming stuff
        tag = extract_MF_name(fname_preds)
        # tag = "GLS" if fnames_preds == fname_gls_old else tag
        label = f"{tag}: " r"$\mu =$ " f"{mean:.2f}, " r"$\sigma =$ " f"{std:.2f}"

        # create histogram and add to histogram plot
        hist = Histogram(preds_cut, label=label)
        plot_histo.add(hist, reference=False)

    t_valid_mask = truth_hadrons["valid"][:n_test]
    truth_cut = truth[t_valid_mask]
    truth_label = (
        "truth: "
        r"$\mu =$ "
        f"{np.mean(truth_cut):.2f}, "
        r"$\sigma =$ "
        f"{np.std(truth_cut):.2f}"
    )
    hist = Histogram(truth_cut, label=truth_label, linestyle="--", colour="black", linewidth=1)
    plot_histo.add(hist, reference=True)

    # more plot naming stuff
    cut_name = (
        "all"
        if cut == (np.nanmin(truth), np.nanmax(truth))
        else f"{cut[0]:.1f}"
        if cut[1] == np.inf
        else f"{cut[0]:.1f}{cut[1]:.1f}"
    )

    plot_dir = "/home/xucabis2/salt/iman/plots/regression"
    timestamp = datetime.now().strftime("%m%d")
    plot_name = f"{plot_dir}/reg_{tag}_{var_name}_{cut_name}_{timestamp}.png"

    # draw and save plot
    plot_histo.draw()
    plot_histo.savefig(plot_name, transparent=False)
    print("Saving to ", plot_name)


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


# fnames_preds = [fname_default, fname_gls, fname_dwa]

fname_default = (
    "/home/xucabis2/salt/logs/_old/_pre-b-hadron-weighting/_baselines/"
    "MaskFormer_default_20240724-T112538/ckpts/epoch=019-val_loss=0.65355__test_ttbar.h5"
)


fname_gls = (
    "/home/xucabis2/salt/logs/MaskFormer_GLS_20240809-T191439/"
    "ckpts/epoch=018-val_loss=0.65630__test_ttbar.h5"
)

fname_dwa = (
    "/home/xucabis2/salt/logs/MaskFormer_DWA_20240811-T105042/"
    "ckpts/epoch=019-val_loss=0.64261__test_ttbar.h5"
)

fname_uw = (
    "/home/xucabis2/salt/logs/MaskFormer_UW_20240810-T111330/"
    "ckpts/epoch=019-val_loss=0.29020__test_ttbar.h5"
)

fname_gls_old = (
    "/home/xucabis2/salt/logs/_old/_pre-b-hadron-weighting/"
    "MaskFormer_GLS_20240730-T002427/ckpts/epoch=018-val_loss=0.65171__test_ttbar.h5"
)

fnames_preds = [fname_default, fname_gls_old, fname_dwa, fname_uw]


fname_truth = "/home/xzcappon/phd/datasets/vertexing_120m/output/pp_output_test_ttbar.h5"

# Lxy cuts
cuts = [(0, 0.1), (0.1, 1), (1, 5), (5, np.inf)]

variables = ["pt", "deta", "dphi", "Lxy", "mass"]

# # raw plots with no cuts - sense check
# for variable in variables:
#     object_residuals(fnames_preds, fname_truth, variable, norm=True)
#     object_residuals(fnames_preds, fname_truth, variable, norm=False)


object_residuals(
    fnames_preds, fname_truth, "mass", plot_range=(1000, 7000)
)  # , plot_range=(-750, 500))
