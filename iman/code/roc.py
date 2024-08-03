import h5py
import numpy as np
import pandas as pd
from ftag.hdf5 import H5Reader
from puma import Roc, RocPlot
from puma.metrics import calc_rej


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


def disc_fct(arr: np.ndarray, f_c: float = 0.018) -> np.ndarray:
    # note that here we need to list columns as u,c,b as opposed to how they appear
    # in the test set which is in order b,c,u
    return np.log(arr[2] / (f_c * arr[1] + (1 - f_c) * arr[0]))


fname_default = (
    "/home/xucabis2/salt/logs/MaskFormer_default_20240724-T112538/"
    "ckpts/epoch=019-val_loss=0.65355__test_ttbar.h5"
)

reader = H5Reader(fname_default, batch_size=1_000)
df = pd.DataFrame(
    reader.load(
        {
            "jets": [
                "pt",
                "eta",
                "flavour_label",
                "MaskFormer_default_pb",
                "MaskFormer_default_pc",
                "MaskFormer_default_pu",
                "GN2v00_pb",
                "GN2v00_pc",
                "GN2v00_pu",
            ]
        },
        num_jets=500_000,
    )["jets"]
)

fname_comp = (
    "/home/xucabis2/salt/logs/MaskFormer_GLS_20240730-T002427/"
    "ckpts/epoch=018-val_loss=0.65171__test_ttbar.h5"
)

h5file = h5py.File(fname_comp, "r")
comp_name = extract_MF_name(fname_comp)

reader = H5Reader(fname_comp, batch_size=1_000)
df_comp = pd.DataFrame(
    reader.load(
        {
            "jets": [
                "pt",
                "eta",
                "flavour_label",
                "MaskFormer_default_pb",
                "MaskFormer_default_pc",
                "MaskFormer_default_pu",
            ]
        },
        num_jets=500_000,
    )["jets"]
)

fname_rlw = (
    "/home/xucabis2/salt/logs/MaskFormer_RLW_20240731-T012016/"
    "ckpts/epoch=019-val_loss=0.64431__test_ttbar.h5"
)

h5file = h5py.File(fname_rlw, "r")
rlw_name = extract_MF_name(fname_rlw)

reader = H5Reader(fname_rlw, batch_size=1_000)
df_rlw = pd.DataFrame(
    reader.load(
        {
            "jets": [
                "pt",
                "eta",
                "flavour_label",
                "MaskFormer_RLW_pb",
                "MaskFormer_RLW_pc",
                "MaskFormer_RLW_pu",
            ]
        },
        num_jets=500_000,
    )["jets"]
)

discs_gn2 = np.apply_along_axis(
    disc_fct,
    1,
    df[["GN2v00_pu", "GN2v00_pc", "GN2v00_pb"]].values,
)

discs_mf = np.apply_along_axis(
    disc_fct,
    1,
    df[["MaskFormer_default_pu", "MaskFormer_default_pc", "MaskFormer_default_pb"]].values,
)

discs_mf_comp = np.apply_along_axis(
    disc_fct,
    1,
    df_comp[["MaskFormer_default_pu", "MaskFormer_default_pc", "MaskFormer_default_pb"]].values,
)

discs_mf_rlw = np.apply_along_axis(
    disc_fct,
    1,
    df_rlw[["MaskFormer_RLW_pu", "MaskFormer_RLW_pc", "MaskFormer_RLW_pb"]].values,
)

sig_eff = np.linspace(0.49, 1, 20)
is_light = df["flavour_label"] == 2
is_c = df["flavour_label"] == 1
is_b = df["flavour_label"] == 0
n_jets_light = sum(is_light)
n_jets_c = sum(is_c)

gn2_ujets_rej = calc_rej(discs_gn2[is_b], discs_gn2[is_light], sig_eff)
gn2_cjets_rej = calc_rej(discs_gn2[is_b], discs_gn2[is_c], sig_eff)

mf_ujets_rej = calc_rej(discs_mf[is_b], discs_mf[is_light], sig_eff)
mf_cjets_rej = calc_rej(discs_mf[is_b], discs_mf[is_c], sig_eff)

mf_comp_ujets_rej = calc_rej(discs_mf_comp[is_b], discs_mf_comp[is_light], sig_eff)
mf_comp_cjets_rej = calc_rej(discs_mf_comp[is_b], discs_mf_comp[is_c], sig_eff)

mf_rlw_ujets_rej = calc_rej(discs_mf_rlw[is_b], discs_mf_rlw[is_light], sig_eff)
mf_rlw_cjets_rej = calc_rej(discs_mf_rlw[is_b], discs_mf_rlw[is_c], sig_eff)

plot_roc = RocPlot(
    n_ratio_panels=2,
    ylabel="Background rejection",
    xlabel="$b$-jet efficiency",
    # atlas_second_tag="$\\sqrt{s}=13$ TeV, ttbar jets \ntutorial sample, $f_{c}=0.018$",
    figsize=(6.5, 6),
    y_scale=1.4,
)
# GN2 plots
plot_roc.add_roc(
    Roc(
        sig_eff,
        gn2_ujets_rej,
        n_test=n_jets_light,
        rej_class="ujets",
        signal_class="bjets",
        label="GN2",
    ),
    reference=True,
)
plot_roc.add_roc(
    Roc(
        sig_eff,
        gn2_cjets_rej,
        n_test=n_jets_c,
        rej_class="cjets",
        signal_class="bjets",
        label="GN2",
    ),
    reference=True,
)
# MF baseline plots
plot_roc.add_roc(
    Roc(
        sig_eff,
        mf_ujets_rej,
        n_test=n_jets_light,
        rej_class="ujets",
        signal_class="bjets",
        label="MaskFormer (default)",
    ),
    # reference=True,
)
plot_roc.add_roc(
    Roc(
        sig_eff,
        mf_cjets_rej,
        n_test=n_jets_c,
        rej_class="cjets",
        signal_class="bjets",
        label="MaskFormer (default)",
    ),
    # reference=True,
)
# MF comparison plots
plot_roc.add_roc(
    Roc(
        sig_eff,
        mf_comp_ujets_rej,
        n_test=n_jets_light,
        rej_class="ujets",
        signal_class="bjets",
        label=f"MaskFormer {comp_name}",
    ),
    # reference=True,
)
plot_roc.add_roc(
    Roc(
        sig_eff,
        mf_comp_cjets_rej,
        n_test=n_jets_c,
        rej_class="cjets",
        signal_class="bjets",
        label=f"MaskFormer {comp_name}",
    ),
    # reference=True,
)

# MF RLW plots
plot_roc.add_roc(
    Roc(
        sig_eff,
        mf_rlw_ujets_rej,
        n_test=n_jets_light,
        rej_class="ujets",
        signal_class="bjets",
        label=f"MaskFormer {rlw_name}",
    ),
    # reference=True,
)
plot_roc.add_roc(
    Roc(
        sig_eff,
        mf_rlw_cjets_rej,
        n_test=n_jets_c,
        rej_class="cjets",
        signal_class="bjets",
        label=f"MaskFormer {rlw_name}",
    ),
    # reference=True,
)


plot_dir = "/home/xucabis2/salt/iman/plots/Aug1"
plot_roc.set_ratio_class(1, "ujets")
plot_roc.set_ratio_class(2, "cjets")
plot_roc.draw()
plot_name = f"{plot_dir}/roc_{rlw_name}.png"
print("Saving to ", plot_name)
plot_roc.savefig(plot_name, transparent=False)
