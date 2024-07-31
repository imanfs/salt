import datetime
import warnings

import h5py
import numpy as np
from puma import Histogram, HistogramPlot

warnings.filterwarnings("ignore")


def valid(objects):
    class_probs = objects["object_class_probs"]
    class_probs = np.array(class_probs)
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


def initialise_histogram(var_name, norm):
    normed = "Normalised" if norm else "Unnormalised"
    if var_name == "deta":
        var = r"$\Delta\eta$"
    elif var_name == "dphi":
        var = r"$\Delta\phi$"
    else:
        var = var_name

    return HistogramPlot(
        ylabel=f"{normed} {var} residual",
        xlabel=f"Hadron {var}",
        logy=False,
        # bins=np.linspace(0, 5, 60),  # you can force a binning for the plot here
        # bins=60,  # you can also define an integer number for the number of bins
        # bins_range=bin_range,  # only considered if bins is an integer
        norm=True,
        atlas_first_tag="Simulation Internal",
        figsize=(6, 5),
        n_ratio_panels=1,
    )


# def plot_residuals(plot_histo, hist, var_name, cut_name, tag, norm=False):
#     # if plot_histo.bins_range is None:
#     #     bin_name = "unbinned"
#     # else:
#     #     bin_name = "binned"

#     # Add histograms and plot
#     # n_bins = plot_histo.bins


def extract_MF_name(path):
    prefix = "MaskFormer_"
    # Find the position of 'MaskFormer_'
    start_pos = path.find(prefix)
    if start_pos != -1:
        try:
            start = start_pos + len(prefix)
            end = path.index("_", start)
            return path[start:end]
        except ValueError:
            return ""
    else:
        return ""


def get_timestamp():
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d")


def object_residuals(fnames_preds, fname_truth, var_name, cut_range=None, norm=False):
    # extract truth hadrons from test file
    h5truth = h5py.File(fname_truth, "r")
    truth_hadrons = h5truth["truth_hadrons"]

    # initialise histogram plot
    plot_histo = initialise_histogram(var_name, norm)

    # loop through prediction files
    for fname_preds in fnames_preds:
        # extract regression predictions from prediction file
        h5preds = h5py.File(fname_preds, "r")
        n_test = h5preds["tracks"].shape[0]
        objects = h5preds["objects"]  # hadrons
        reg_preds = objects["regression"]  # truth hadron regression predictions

        # slice truth hadrons file to match number of predicted objects
        truth_hadrons = truth_hadrons[:n_test]

        # get truth values and regression predictions for chosen variable (pT, Lxy, etc)
        preds = reg_preds["regression_" + var_name]
        truth = truth_hadrons[var_name]

        # get valid mask (predict null with p < 0.5) and filter for valid predictions
        valid_mask = valid(objects)
        valid_mask &= truth_hadrons["valid"]
        preds, truth = preds[valid_mask], truth[valid_mask]

        # if no cut is specified, use the full range of the truth values
        if cut_range is None:
            cut = None  # clear previous cut if no cut range chosen
        if cut is None:
            cut = (np.nanmin(truth), np.nanmax(truth))

        # apply range cut if necessary and calculate residuals
        mask = (truth > cut[0]) & (truth < cut[1])
        preds_cut = preds[mask]
        truth_cut = truth[mask]
        residuals = calc_residuals(truth_cut, preds_cut, norm=norm)

        # plot naming formatting
        if cut == (np.nanmin(truth), np.nanmax(truth)):
            cut_name = "all"
        elif cut[1] == np.inf:
            # bin_range = (cut[0],np.nanmax(truth))
            cut_name = f"{cut[0]:.1f}"
        else:
            # bin_range = cut
            cut_name = f"{cut[0]:.1f}_{cut[1]:.1f}"

        # naming stuff for plots (reference plot will be default weighting scheme)
        tag = extract_MF_name(fname_preds)
        label = f"{var_name} ({tag})"
        ref = tag == "default"

        # create histogram and add to histogram plot
        hist = Histogram(residuals, label=label)
        plot_histo.add(hist, reference=ref)

    # more plot naming stuff
    normed = "norm" if norm else "unnorm"
    plot_dir = "/home/xucabis2/salt/iman/plots"
    timestamp = get_timestamp()
    plot_name = f"{plot_dir}/{normed}/residuals_{var_name}_{cut_name}_{normed}_{timestamp}.png"

    # draw and save plot
    plot_histo.draw()
    print("Saving to ", plot_name)
    plot_histo.savefig(plot_name, transparent=False)


logs_dir = "/home/xucabis2/salt/logs"

ckpt_arxiv = "epoch=019-val_loss=0.65962__test_ttbar.h5"
ckpt_default = "epoch=019-val_loss=0.65355__test_ttbar.h5"
ckpt_equal = "epoch=019-val_loss=0.64428__test_ttbar.h5"
ckpt_gls = "epoch=018-val_loss=0.65171__test_ttbar.h5"

fname_arxiv = f"{logs_dir}/MaskFormer_arxiv_20240724-T114414/ckpts/{ckpt_arxiv}"
fname_default = f"{logs_dir}/MaskFormer_default_20240724-T112538/ckpts/{ckpt_default}"
fname_equal = f"{logs_dir}/MaskFormer_equal_20240724-T114419/ckpts/{ckpt_equal}"
fname_gls = f"{logs_dir}/MaskFormer_GLS_20240730-T002427/ckpts/{ckpt_gls}"
fnames_preds = [fname_arxiv, fname_default, fname_equal, fname_gls]

truth_dir = "/home/xzcappon/phd/datasets"
fname_truth = f"{truth_dir}/vertexing_120m/output/pp_output_test_ttbar.h5"

# Lxy cuts
cuts = [(0, 0.1), (0.1, 1), (1, 5), (5, np.inf)]
# for cut in cuts:
#     object_residuals(fnames_preds, fname_truth, "pt", cut, norm=True)
#     object_residuals(fnames_preds, fname_truth, "pt", cut, norm=False)

variables = ["pt", "deta", "dphi", "Lxy", "mass"]
# variables = ["deta"]
for variable in variables:
    object_residuals(fnames_preds, fname_truth, variable, norm=True)
    object_residuals(fnames_preds, fname_truth, variable, norm=False)
