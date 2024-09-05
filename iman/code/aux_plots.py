"""Produce aux task plots from tagger output and labels."""

from __future__ import annotations

import h5py
from puma.hlplots import AuxResults
from puma.utils import logger
from utils import create_and_load_taggers, load_file_paths

file_path = "/home/xucabis2/salt/iman/files_lossbased_ttbar.txt"
file_paths = load_file_paths(file_path)

with h5py.File(file_paths["fname_default"]) as file:
    n_jets = len(file["jets"])

# define jet selections
cuts = [("n_truth_promptLepton", "==", 0), ("pt", ">", 20000), ("pt", "<", 250000)]

# create the AuxResults object
reference = "SV1"
out_dir = "/home/xucabis2/salt/iman/plots/vertexing"
out_dir += "" if reference is not None else "/no_ref"
print(out_dir)
aux_results = AuxResults(sample="ttbar-2", output_dir=out_dir)
aux_results.atlas_second_tag = (
    "$\\sqrt{s}=13$ TeV, dummy jets \n$t\\bar{t}$, $20$ GeV $< p_{T} <250$ GeV"
)

logger.info("Loading taggers.")
for i, fname in enumerate(file_paths.values()):
    ref = reference if i == 0 else None
    create_and_load_taggers(fname, aux_results, cuts, n_jets, ref=ref)


logger.info("Plotting vertexing performance.")
aux_results.plot_var_vtx_perf(vtx_flavours=["bjets", "cjets"], no_vtx_flavours=["ujets"])
aux_results.plot_var_vtx_perf(
    vtx_flavours=["bjets", "cjets"], no_vtx_flavours=["ujets"], incl_vertexing=True
)  # this is to compare old and new models
