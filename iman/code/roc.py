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


# sig_eff = np.linspace(0.49, 1, 20)
# is_light = df["flavour_label"] == 2
# is_c = df["flavour_label"] == 1
# is_b = df["flavour_label"] == 0
# n_jets_light = sum(is_light)
# n_jets_c = sum(is_c)

# def disc_fct(arr: np.ndarray, f_c: float = 0.018) -> np.ndarray:
#     # note that here we need to list columns as u,c,b as opposed to how they appear
#     # in the test set which is in order b,c,u
#     return np.log(arr[2] / (f_c * arr[1] + (1 - f_c) * arr[0]))


# def apply_discs(df, name):
#     label = "GN2v00_" if "GN2" in name else f"MaskFormer_{name}"
#     return np.apply_along_axis(
#         disc_fct,
#         1,
#         df[[f"{label}_pu", f"{label}_pc", f"{label}_pb"]].values,
#     )

# def plt_roc(RocPlot, sig_eff, bkg_rej, n_test, rej_class, name, tags="", ref=None):
#     name_label = name.capitalize() if "default" in name else name.upper()
#     label = "GN2" if "GN2" in name else f"MF-{name_label}" + tags

#     RocPlot.add_roc(
#         Roc(
#             sig_eff,
#             bkg_rej,
#             n_test=n_test,
#             rej_class=rej_class,
#             signal_class="bjets",
#             label=label,
#         ),
#         reference=ref if ref is not None else name == "GN2",
#     )


# def get_roc_vars_and_plot(df, name, sig_eff, is_b, is_light, is_c, n_jets_light, n_jets_c):
#     discs_gn2 = apply_discs(df, name)
#     gn2_ujets_rej = calc_rej(discs_gn2[is_b], discs_gn2[is_light], sig_eff)
#     gn2_cjets_rej = calc_rej(discs_gn2[is_b], discs_gn2[is_c], sig_eff)

#     plt_roc(plot_roc, sig_eff, gn2_ujets_rej, n_jets_light, "ujets", name)
#     plt_roc(plot_roc, sig_eff, gn2_cjets_rej, n_jets_c, "cjets", name)


class ROCPlotter:
    def __init__(self, df, sig_eff=None):
        if sig_eff is None:
            sig_eff = np.linspace(0.49, 1, 20)
        self.df = df
        self.sig_eff = sig_eff

        # Calculate labels once and store as attributes
        self.is_light = df["flavour_label"] == 2
        self.is_c = df["flavour_label"] == 1
        self.is_b = df["flavour_label"] == 0

        self.n_jets_light = sum(self.is_light)
        self.n_jets_c = sum(self.is_c)

    def disc_fct(self, arr: np.ndarray, f_c: float = 0.018) -> np.ndarray:
        """Calculate the discriminant function."""
        return np.log(arr[2] / (f_c * arr[1] + (1 - f_c) * arr[0]))

    def apply_discs(self):
        """Apply the discriminant function across the data in df."""
        label = "GN2v00" if "GN2" in self.name else f"MaskFormer_{self.name}"
        return np.apply_along_axis(
            self.disc_fct,
            1,
            self.df[[f"{label}_pu", f"{label}_pc", f"{label}_pb"]].values,
        )

    def plt_roc(self, RocPlot, bkg_rej, n_test, rej_class, tags=""):
        """Plot the ROC curve using the RocPlot object."""
        name_label = self.name.capitalize() if "default" in self.name else self.name.upper()
        label = "GN2" if "GN2" in self.name else f"MF-{name_label}" + tags

        RocPlot.add_roc(
            Roc(
                self.sig_eff,
                bkg_rej,
                n_test=n_test,
                rej_class=rej_class,
                signal_class="bjets",
                label=label,
            ),
            reference=self.ref if self.ref is not None else self.name == "GN2",
        )

    def get_roc_vars_and_plot(self, RocPlot, name, ref=None):
        """Calculate ROC variables and plot them."""
        self.name = name
        self.ref = ref
        discs = self.apply_discs()
        ujets_rej = calc_rej(discs[self.is_b], discs[self.is_light], self.sig_eff)
        cjets_rej = calc_rej(discs[self.is_b], discs[self.is_c], self.sig_eff)

        self.plt_roc(RocPlot, ujets_rej, self.n_jets_light, "ujets")
        self.plt_roc(RocPlot, cjets_rej, self.n_jets_c, "cjets")


fname_default = (
    "/home/xucabis2/salt/logs/_final/MaskFormer_default_20240828-T113812/ckpts/"
    "epoch=029-val_loss=0.64809__test_ttbar.h5"
)
fname_gls = (
    "/home/xucabis2/salt/logs/_final/MaskFormer_GLS_20240902-T084826/ckpts/"
    "epoch=029-val_loss=0.64470__test_ttbar.h5"
)
fname_uw = (
    "/home/xucabis2/salt/logs/_final/MaskFormer_UW_20240901-T003519/ckpts/"
    "epoch=029-val_loss=0.64037__test_ttbar.h5"
)
fname_dwa = (
    "/home/xucabis2/salt/logs/_final/MaskFormer_DWA_20240901-T005936/ckpts/"
    "epoch=029-val_loss=0.63966__test_ttbar.h5"
)
fname_rlw = (
    "/home/xucabis2/salt/logs/_final/MaskFormer_RLW_20240826-T234255/ckpts/"
    "epoch=029-val_loss=0.64060__test_ttbar.h5"
)
fname_stch = (
    "/home/xucabis2/salt/logs/_final/MaskFormer_STCH_20240904-T023541/ckpts/"
    "epoch=029-val_loss=0.64920__test_ttbar.h5"
)

fnames_preds = {
    # fname_default: "",
    fname_stch: "",
    fname_gls: "",
    fname_dwa: "",
    fname_uw: "",
    fname_rlw: "",
}

ref = "MF"

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

plot_roc_gn2 = RocPlot(
    n_ratio_panels=2,
    ylabel="Background rejection",
    xlabel="$b$-jet efficiency",
    atlas_second_tag="$\\sqrt{s}=13$ TeV, ttbar jets, $f_{c}=0.018$",
    figsize=(6.5, 6),
    y_scale=1.4,
)
plot_roc_gn2.set_ratio_class(1, "ujets")
plot_roc_gn2.set_ratio_class(2, "cjets")

plot_roc = RocPlot(
    n_ratio_panels=2,
    ylabel="Background rejection",
    xlabel="$b$-jet efficiency",
    atlas_second_tag="$\\sqrt{s}=13$ TeV, ttbar jets, $f_{c}=0.018$",
    figsize=(6.5, 6),
    y_scale=1.4,
)
plot_roc.set_ratio_class(1, "ujets")
plot_roc.set_ratio_class(2, "cjets")

rocplotter = ROCPlotter(df)

rocplotter.get_roc_vars_and_plot(plot_roc_gn2, name="GN2")
rocplotter.get_roc_vars_and_plot(plot_roc_gn2, "default", ref=False)

rocplotter.get_roc_vars_and_plot(plot_roc, "default", ref=True)

for fname in fnames_preds:
    mf_name = extract_MF_name(fname)
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
    rocplotter = ROCPlotter(df)
    rocplotter.get_roc_vars_and_plot(plot_roc_gn2, mf_name, ref=False)
    rocplotter.get_roc_vars_and_plot(plot_roc, mf_name, ref=False)
    # MF comparison plots

plot_dir = "/home/xucabis2/salt/iman/plots/final"
weighting = "lossbased"

plot_roc_gn2.draw()
plot_name = f"{plot_dir}/roc_{weighting}_GN2ref.png"
print("Saving to ", plot_name)
plot_roc_gn2.savefig(plot_name, transparent=False)


plot_roc.draw()
plot_name = f"{plot_dir}/roc_{weighting}_MFref.png"
print("Saving to ", plot_name)
plot_roc.savefig(plot_name, transparent=False)
