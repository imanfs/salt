import numpy as np
import pandas as pd
from ftag.hdf5 import H5Reader
from puma import Roc, RocPlot
from puma.metrics import calc_rej
from utils import load_file_paths


def extract_MF_name(path, mf_only=False):
    if mf_only:
        return "salt"
    prefix = "MaskFormer_"
    suffix = "_"
    return path.partition(prefix)[2].partition(suffix)[0]


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

    def plt_roc(self, RocPlot, bkg_rej, n_test, rej_class, tags="", colour=None):
        """Plot the ROC curve using the RocPlot object."""
        if self.label is None:
            self.label = self.name.capitalize() if "default" in self.name else self.name.upper()
        label = "GN2" if "GN2" in self.name else f"MF-{self.label}" + tags

        RocPlot.add_roc(
            Roc(
                self.sig_eff,
                bkg_rej,
                n_test=n_test,
                rej_class=rej_class,
                signal_class="bjets",
                label=label,
                colour=colour,
            ),
            reference=self.ref if self.ref is not None else self.name == "GN2",
        )

    def get_roc_vars_and_plot(self, RocPlot, name, label=None, ref=None, colour=None):
        """Calculate ROC variables and plot them."""
        self.name = name
        self.label = label
        self.ref = ref
        discs = self.apply_discs()
        ujets_rej = calc_rej(discs[self.is_b], discs[self.is_light], self.sig_eff)
        cjets_rej = calc_rej(discs[self.is_b], discs[self.is_c], self.sig_eff)

        self.plt_roc(RocPlot, ujets_rej, self.n_jets_light, "ujets", colour=colour)
        self.plt_roc(RocPlot, cjets_rej, self.n_jets_c, "cjets", colour=colour)


# file_path = "/home/xucabis2/salt/iman/files_gradbased_ttbar.txt"
# file_path = "/home/xucabis2/salt/iman/files_lossbased_ttbar.txt"
file_path = "/home/xucabis2/salt/iman/files_polyloss_ttbar.txt"
file_paths_dict = load_file_paths(file_path)
fnames_preds = file_paths_dict.copy()  # Create a copy of the original dictionary
fnames_preds.pop("Default", None)

ref = "MF"

reader = H5Reader(file_paths_dict["Default"], batch_size=1_000)
df_default = pd.DataFrame(
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

# sig_eff = np.linspace(0.59, 0.9, 20)
sig_eff = np.linspace(0.49, 1, 20)
rocplotter_default = ROCPlotter(df_default, sig_eff)

# plot MF default first so they have the same colour
rocplotter_default.get_roc_vars_and_plot(plot_roc_gn2, "default", ref=False)
rocplotter_default.get_roc_vars_and_plot(plot_roc, "default", ref=True)

for label, fname in fnames_preds.items():
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
    rocplotter = ROCPlotter(df, sig_eff)
    # MF comparison plots
    rocplotter.get_roc_vars_and_plot(plot_roc_gn2, mf_name, label=label, ref=False)
    rocplotter.get_roc_vars_and_plot(plot_roc, mf_name, label=label, ref=False)

# plot gn2 last
rocplotter_default.get_roc_vars_and_plot(plot_roc_gn2, name="GN2", ref=True, colour="#28427b")

plot_dir = "/home/xucabis2/salt/iman/plots/figs"
weighting = (
    "lossbased"
    if "lossbased" in file_path
    else "gradbased"
    if "gradbased" in file_path
    else "polyloss"
    if "polyloss" in file_path
    else "undefined"
)

plot_roc_gn2.draw()

range_str = "" if sig_eff[0] == 0.49 and sig_eff[-1] == 1 else f"_{sig_eff[0]}_{sig_eff[-1]}"

plot_name = f"{plot_dir}/roc_{weighting}_GN2ref{range_str}.png"
print("Saving to ", plot_name)
plot_roc_gn2.savefig(plot_name, transparent=False)


plot_roc.draw()
plot_name = f"{plot_dir}/roc_{weighting}_MFref{range_str}.png"
print("Saving to ", plot_name)
plot_roc.savefig(plot_name, transparent=False)
