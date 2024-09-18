import numpy as np
import pandas as pd
from ftag.hdf5 import H5Reader
from puma.metrics import calc_rej
from utils import load_file_paths


def rej_to_csv(label, sig_eff, ujets_rej, cjets_rej, file_name="rejections.csv"):
    """Append rejection values for u-jets and c-jets to a CSV file."""
    # Create a DataFrame with the values
    data = {
        "Label": [label] * len(sig_eff),
        "Signal Efficiency": sig_eff,
        "u-jets Rejection": [round(val, 2) for val in ujets_rej],
        "c-jets Rejection": [round(val, 2) for val in cjets_rej],
    }

    df = pd.DataFrame(data)
    write_header = True
    with open(file_name) as f:
        first_line = f.readline().strip()
        if first_line:
            write_header = False

    df.to_csv(file_name, mode="a", header=write_header, index=False)
    # Append to CSV file (create if doesn't exist)
    print(f"Data for label {label} appended to {file_name}")


def percent_to_csv(label, sig_eff, ujets_percent, cjets_percent, file_name="percent.csv"):
    """Append rejection values % increase for u-jets and c-jets to a CSV file."""
    # Create a DataFrame with the values
    data = {
        "Label": [label] * len(sig_eff),
        "Signal Efficiency": sig_eff,
        "u-jets Rejection (% increase)": [round(val, 2) for val in ujets_percent],
        "c-jets Rejection (% increase)": [round(val, 2) for val in cjets_percent],
    }

    df = pd.DataFrame(data)
    write_header = True

    with open(file_name) as f:
        first_line = f.readline().strip()
        if first_line:
            write_header = False

    df.to_csv(file_name, mode="a", header=write_header, index=False)

    print(f"Data for label {label} appended to {file_name}")


def extract_MF_name(path, mf_only=False):
    if mf_only:
        return "salt"
    prefix = "MaskFormer_"
    suffix = "_"
    return path.partition(prefix)[2].partition(suffix)[0]


def extract_fig_name(path):
    prefix = "files_"
    suffix = "_"
    return path.partition(prefix)[2].partition(suffix)[0]


class RejectionCalc:
    def __init__(self, df, sig_eff=None):
        if sig_eff is None:
            sig_eff = np.array([0.6, 0.7, 0.77, 0.85])
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

    def get_roc_vars(self, name, label=None):
        """Calculate ROC variables and plot them."""
        self.name = name
        self.label = label
        discs = self.apply_discs()
        ujets_rej = calc_rej(discs[self.is_b], discs[self.is_light], self.sig_eff)
        cjets_rej = calc_rej(discs[self.is_b], discs[self.is_c], self.sig_eff)
        return ujets_rej, cjets_rej


# Assuming `roc_calc.get_roc_vars()` returns the rejection values
rej_file = "/home/xucabis2/salt/iman/plots/figs/rejections.csv"  # Output CSV file
percent_file = "/home/xucabis2/salt/iman/plots/figs/percent.csv"  # Output CSV file

with open(rej_file, "w") as file:
    pass

with open(percent_file, "w") as file:
    pass

file_path = "/home/xucabis2/salt/iman/files_all_ttbar.txt"
file_paths_dict = load_file_paths(file_path)
fnames_preds = file_paths_dict.copy()  # Create a copy of the original dictionary
fnames_preds.pop("Default", None)

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

roc_calc = RejectionCalc(df_default)
ujets_rej_def, cjets_rej_def = roc_calc.get_roc_vars("default")
rej_to_csv("Default", roc_calc.sig_eff, ujets_rej_def, cjets_rej_def, file_name=rej_file)

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
    roc_calc = RejectionCalc(df)

    ujets_rej, cjets_rej = roc_calc.get_roc_vars(mf_name)
    rej_to_csv(label, roc_calc.sig_eff, ujets_rej, cjets_rej, file_name=rej_file)

    # Compute percentage increase over the "Default" model
    ujets_rej_percent = ((ujets_rej - ujets_rej_def) / ujets_rej_def) * 100
    cjets_rej_percent = ((cjets_rej - cjets_rej_def) / cjets_rej_def) * 100

    # Append percentage increase to CSV
    percent_to_csv(
        label, roc_calc.sig_eff, ujets_rej_percent, cjets_rej_percent, file_name=percent_file
    )

plot_dir = "/home/xucabis2/salt/iman/plots/figs"
weighting = extract_fig_name(file_path)
