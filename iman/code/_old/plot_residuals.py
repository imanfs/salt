import h5py
import numpy as np
from puma import Histogram, HistogramPlot


def calc_residuals(x_true, x_pred, norm=True):
    residuals = x_pred - x_true
    if norm:
        residuals = residuals / x_true
    # residuals[np.isnan(residuals)] = 0
    residuals[np.isinf(residuals)] = 0
    return residuals


def valid(objects):
    class_probs = objects["object_class_probs"]
    class_probs = np.array(class_probs)

    class_probs_3d = class_probs.view((np.float32, 3))
    # valid_mask = class_probs['pnull'] < 0.5
    return class_probs_3d[:, :, -1] < 0.5


def plot_residuals(fname_preds, fname_truth, var_name, cuts=None, norm=False):
    h5preds, h5truth = h5py.File(fname_preds, "r"), h5py.File(fname_truth, "r")
    n_test = h5preds["tracks"].shape[0]

    objects = h5preds["objects"]  # hadrons
    reg_preds = objects["regression"]  # truth hadron regression predictions
    if "arxiv" in fname_preds:
        tag = "arxiv"
    elif "default" in fname_preds:
        tag = "default"
    elif "equal" in fname_preds:
        tag = "equal"

    # get truth hadrons from truth file
    truth_hadrons = h5truth["truth_hadrons"]
    truth_hadrons = truth_hadrons[:n_test]

    # get valid mask (predict null with p < 0.5)
    valid_mask = valid(objects)
    valid_mask &= truth_hadrons["valid"]

    # get truth values and regression predictions for chosen variable
    preds = reg_preds["regression_" + var_name]
    truth = truth_hadrons[var_name]

    preds, truth = preds[valid_mask], truth[valid_mask]

    if cuts is None:
        cuts = [(np.nanmin(truth), np.nanmax(truth))]
        # bin_range = cuts
    for cut in cuts:
        # apply range cut
        mask = (truth > cut[0]) & (truth < cut[1])
        preds_cut = preds[mask]
        truth_cut = truth[mask]

        # plot naming formatting
        if cut[1] == np.inf:
            # bin_range = (cut[0],np.nanmax(truth))
            cut_name = f"{cut[0]:.1f}"
        elif cuts == [(np.nanmin(truth), np.nanmax(truth))]:
            cut_name = "all"
        else:
            # bin_range = cut
            cut_name = f"{cut[0]:.1f}_{cut[1]:.1f}"

        # initialise histogram plot
        plot_histo = HistogramPlot(
            ylabel=f"Normalised {var_name} residual",
            xlabel=f"Hadron {var_name}",
            logy=False,
            # bins=np.linspace(0, 5, 60),  # you can force a binning for the plot here
            # bins=60,  # you can also define an integer number for the number of bins
            # bins_range=bin_range,  # only considered if bins is an integer
            norm=True,
            atlas_first_tag="Simulation Internal",
            figsize=(6, 5),
            n_ratio_panels=1,
        )
        # if plot_histo.bins_range is None:
        #     bin_name = "unbinned"
        # else:
        #     bin_name = "binned"

        residuals = calc_residuals(truth_cut, preds_cut, norm=norm)
        hist = Histogram(residuals, label=var_name)

        # Add histograms and plot
        # n_bins = plot_histo.bins
        norm_val = "norm" if norm else "unnorm"
        plot_histo.add(hist, reference=True)
        plot_histo.draw()
        plot_dir = "/home/xucabis2/salt/iman/plots"
        plot_name = f"{plot_dir}/{norm_val}/{tag}_residuals_{var_name}_{cut_name}_{norm_val}.png"
        print("Saving to ", plot_name)
        plot_histo.savefig(plot_name, transparent=False)


# fname_preds = "/home/xucabis2/salt/logs/
# MaskFormer_interactive_20240711-T110901/ckpts/
# epoch=019-val_loss=0.65825__test_ttbar.h5"

logs_dir = "/home/xucabis2/salt/logs"
test_dir = "/home/xzcappon/phd/datasets"
ckpt = "epoch=019-val_loss=0.65962__test_ttbar.h5"
fname_preds = f"{logs_dir}/MaskFormer_arxiv_20240724-T114414/ckpts/{ckpt}"
fname_truth = f"{test_dir}/vertexing_120m/output/pp_output_test_ttbar.h5"

logs_dir = "/home/xucabis2/salt/logs"
test_dir = "/home/xzcappon/phd/datasets"
ckpt_arxiv = "epoch=019-val_loss=0.65962__test_ttbar.h5"
ckpt_default = "epoch=019-val_loss=0.65355__test_ttbar.h5"
ckpt_equal = "epoch=019-val_loss=0.64428__test_ttbar.h5"
fname_arxiv = f"{logs_dir}/MaskFormer_arxiv_20240724-T114414/ckpts/{ckpt_arxiv}"
fname_default = f"{logs_dir}/MaskFormer_default_20240724-T112538/ckpts/{ckpt_default}"
fname_equal = f"{logs_dir}/MaskFormer_equal_20240724-T114419/ckpts/{ckpt_equal}"
fname_truth = f"{test_dir}/vertexing_120m/output/pp_output_test_ttbar.h5"

cuts = [(0, 0.1), (0.1, 1), (1, 5), (5, np.inf)]
fnames = [fname_arxiv, fname_default, fname_equal]
for fname in fnames:
    plot_residuals(fname, fname_truth, "deta", norm=False)
