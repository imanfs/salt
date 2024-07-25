import h5py
import numpy as np
import pandas as pd
from ftag.hdf5 import H5Reader
from puma import Roc, RocPlot
from puma.metrics import calc_rej

logs_dir = "/home/xucabis2/salt/logs"
ckpt = "epoch=019-val_loss=0.64715__test_ttbar.h5"
fname = f"{logs_dir}/MF_16workers_10kbatch_20240710-T103121/ckpts/{ckpt}"
h5file = h5py.File(fname, "r")
print(h5file["jets"].dtype.names)

reader = H5Reader(fname, batch_size=1_000)
df = pd.DataFrame(
    reader.load(
        {
            "jets": [
                "pt",
                "eta",
                "flavour_label",
                "MF_16workers_10kbatch_pb",
                "MF_16workers_10kbatch_pc",
                "MF_16workers_10kbatch_pu",
                "GN2v00_pb",
                "GN2v00_pc",
                "GN2v00_pu",
            ]
        },
        num_jets=1000000,
    )["jets"]
)


def disc_fct(arr: np.ndarray, f_c: float = 0.018) -> np.ndarray:
    # note that here we need to list columns as u,c,b as opposed to how they appear
    # in the test set which is in order b,c,u
    return np.log(arr[2] / (f_c * arr[1] + (1 - f_c) * arr[0]))


discs_gn2 = np.apply_along_axis(
    disc_fct,
    1,
    df[["GN2v00_pu", "GN2v00_pc", "GN2v00_pb"]].values,
)

discs_mf = np.apply_along_axis(
    disc_fct,
    1,
    df[["MF_16workers_10kbatch_pu", "MF_16workers_10kbatch_pc", "MF_16workers_10kbatch_pb"]].values,
)

sig_eff = np.linspace(0.49, 1, 20)
is_light = df["flavour_label"] == 2
is_c = df["flavour_label"] == 1
is_b = df["flavour_label"] == 0
print(df["flavour_label"].unique)
n_jets_light = sum(is_light)
n_jets_c = sum(is_c)

gn2_ujets_rej = calc_rej(discs_gn2[is_b], discs_gn2[is_light], sig_eff)
gn2_cjets_rej = calc_rej(discs_gn2[is_b], discs_gn2[is_c], sig_eff)

mf_ujets_rej = calc_rej(discs_mf[is_b], discs_mf[is_light], sig_eff)
mf_cjets_rej = calc_rej(discs_mf[is_b], discs_mf[is_c], sig_eff)


plot_roc = RocPlot(
    n_ratio_panels=2,
    ylabel="Background rejection",
    xlabel="$b$-jet efficiency",
    atlas_second_tag="$\\sqrt{s}=13$ TeV, ttbar jets \ntutorial sample, $f_{c}=0.018$",
    figsize=(6.5, 6),
    y_scale=1.4,
)
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

plot_roc.add_roc(
    Roc(
        sig_eff,
        mf_ujets_rej,
        n_test=n_jets_light,
        rej_class="ujets",
        signal_class="bjets",
        label="MaskFormer",
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
        label="MaskFormer",
    ),
    # reference=True,
)

plot_roc.set_ratio_class(1, "ujets")
plot_roc.set_ratio_class(2, "cjets")
plot_roc.draw()
plot_roc.savefig("roc.png", transparent=False)
