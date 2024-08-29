from datetime import datetime

import numpy as np
import pandas as pd
from ftag.hdf5 import H5Reader
from puma import Roc, RocPlot
from puma.metrics import calc_rej


def extract_MF_name(path, mf_only=False):
    if mf_only:
        return "salt"
    prefix = "MaskFormer_"
    suffix = "_"
    return path.partition(prefix)[2].partition(suffix)[0]


def disc_fct(arr: np.ndarray, f_c: float = 0.018) -> np.ndarray:
    # note that here we need to list columns as u,c,b as opposed to how they appear
    # in the test set which is in order b,c,u
    return np.log(arr[2] / (f_c * arr[1] + (1 - f_c) * arr[0]))


def apply_discs(df, name):
    label = "GN2v00_" if "GN2" in name else f"MaskFormer_{name}"
    return np.apply_along_axis(
        disc_fct,
        1,
        df[[f"{label}_pu", f"{label}_pc", f"{label}_pb"]].values,
    )


def plt_roc(RocPlot, sig_eff, bkg_rej, n_test, rej_class, name, tags="", ref=None):
    label = "GN2" if "GN2" in name else f"MF-{name} " + tags

    RocPlot.add_roc(
        Roc(
            sig_eff,
            bkg_rej,
            n_test=n_test,
            rej_class=rej_class,
            signal_class="bjets",
            label=label,
        ),
        reference=ref if ref is not None else name == "GN2",
        # reference=name == "MaskFormer (default)",
    )


fname_default = "/home/xucabis2/salt/logs/MaskFormer_default_20240828-T113812/ckpts/epoch=029-val_loss=0.64809__test_ttbar.h5"
fname_gls = "/home/xucabis2/salt/logs/MaskFormer_GLS_20240826-T233103/ckpts/epoch=029-val_loss=0.65035__test_ttbar.h5"

fname_rlw = "/home/xucabis2/salt/logs/MaskFormer_RLW_20240826-T234255/ckpts/epoch=029-val_loss=0.64060__test_ttbar.h5"

fname_dwa = "/home/xucabis2/salt/logs/MaskFormer_DWA_20240826-T233005/ckpts/epoch=028-val_loss=0.63885__test_ttbar.h5"

fname_equal = "/home/xucabis2/salt/logs/MaskFormer_equal_20240826-T234233/ckpts/epoch=029-val_loss=0.63967__test_ttbar.h5"

fnames_preds = {
    # fname_default: "",
    fname_equal: "",
    fname_gls: "",
    fname_dwa: "",
}

# fnames_preds = [fname_gls, fname_dwa]

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

sig_eff = np.linspace(0.49, 1, 20)
is_light = df["flavour_label"] == 2
is_c = df["flavour_label"] == 1
is_b = df["flavour_label"] == 0
n_jets_light = sum(is_light)
n_jets_c = sum(is_c)

# discs_gn2 = apply_discs(df, "GN2")
# gn2_ujets_rej = calc_rej(discs_gn2[is_b], discs_gn2[is_light], sig_eff)
# gn2_cjets_rej = calc_rej(discs_gn2[is_b], discs_gn2[is_c], sig_eff)

discs_mf = apply_discs(df, "default")
mf_ujets_rej = calc_rej(discs_mf[is_b], discs_mf[is_light], sig_eff)
mf_cjets_rej = calc_rej(discs_mf[is_b], discs_mf[is_c], sig_eff)


plot_roc = RocPlot(
    n_ratio_panels=2,
    ylabel="Background rejection",
    xlabel="$b$-jet efficiency",
    atlas_second_tag="$\\sqrt{s}=13$ TeV, ttbar jets, $f_{c}=0.018$",
    figsize=(6.5, 6),
    y_scale=1.4,
)

# plt_roc(plot_roc, sig_eff, gn2_ujets_rej, n_jets_light, "ujets", "GN2")
# plt_roc(plot_roc, sig_eff, gn2_cjets_rej, n_jets_c, "cjets", "GN2")

plt_roc(plot_roc, sig_eff, mf_ujets_rej, n_jets_light, "ujets", "default", ref=True)
plt_roc(plot_roc, sig_eff, mf_cjets_rej, n_jets_c, "cjets", "default", ref=True)

for fname in fnames_preds:
    mf_name = extract_MF_name(fname)
    name = mf_name
    # mf_name = "default_" if fname == fname_gls else mf_name + "_"
    # mf_name = "" if fname == fname_original else mf_name
    reader = H5Reader(fname, batch_size=1_000)
    df = pd.DataFrame(
        reader.load(
            {
                "jets": [
                    "pt",
                    "eta",
                    "flavour_label",
                    f"MaskFormer_{mf_name}_pb",
                    f"MaskFormer_{mf_name}_pc",
                    f"MaskFormer_{mf_name}_pu",
                ]
            },
            num_jets=500_000,
        )["jets"]
    )
    discs = apply_discs(df, mf_name)
    ujets_rej = calc_rej(discs[is_b], discs[is_light], sig_eff)
    cjets_rej = calc_rej(discs[is_b], discs[is_c], sig_eff)

    # MF comparison plots
    plt_roc(
        plot_roc, sig_eff, ujets_rej, n_jets_light, "ujets", name, tags=f"{fnames_preds[fname]}"
    )
    plt_roc(plot_roc, sig_eff, cjets_rej, n_jets_c, "cjets", name, tags=f"{fnames_preds[fname]}")

timestamp = datetime.now().strftime("%m%d")
plot_dir = "/home/xucabis2/salt/iman/plots/roc"
plot_roc.set_ratio_class(1, "ujets")
plot_roc.set_ratio_class(2, "cjets")
plot_roc.draw()
plot_name = f"{plot_dir}/roc_no_ref_{mf_name}_{timestamp}.png"
print("Saving to ", plot_name)
plot_roc.savefig(plot_name, transparent=False)
