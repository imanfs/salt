"""Produce aux task plots from tagger output and labels."""

from __future__ import annotations

import h5py
from puma.hlplots import AuxResults
from puma.utils import logger
from utils import create_and_load_tagger, extract_fig_name, load_file_paths

grad_zprime = "/home/xucabis2/salt/iman/file_paths/files_gradbased_zprime.txt"
loss_zprime = "/home/xucabis2/salt/iman/file_paths/files_lossbased_zprime.txt"
poly_zprime = "/home/xucabis2/salt/iman/file_paths/files_polyloss_zprime.txt"
base_zprime = "/home/xucabis2/salt/iman/file_paths/files_baselines_zprime.txt"

grad_ttbar = "/home/xucabis2/salt/iman/file_paths/files_gradbased_ttbar.txt"
loss_ttbar = "/home/xucabis2/salt/iman/file_paths/files_lossbased_ttbar.txt"
poly_ttbar = "/home/xucabis2/salt/iman/file_paths/files_polyloss_ttbar.txt"
base_ttbar = "/home/xucabis2/salt/iman/file_paths/files_baselines_ttbar.txt"

gn2_zprime = (
    "/home/xucabis2/salt/logs/_final/GN2_20240918-T090931/"
    "ckpts/epoch=028-val_loss=0.63562__test_zprime.h5"
)

files_ttbar = [grad_ttbar, loss_ttbar]
files_zprime = [grad_zprime, loss_zprime]
all_files = files_ttbar + files_zprime
txt_files = files_zprime

for txt_file in txt_files:
    file_paths = load_file_paths(txt_file)

    weighting = extract_fig_name(txt_file)
    weighting_str = "poly" if "poly" in weighting else weighting
    out_dir = f"/home/xucabis2/salt/iman/plots/figs/vertexing/{weighting_str}"

    file_paths = load_file_paths(txt_file)
    fnames = list(file_paths.values())
    labels = list(file_paths.keys())

    # define jet selections
    cuts = [("n_truth_promptLepton", "==", 0), ("pt", ">", 20000), ("pt", "<", 250000)]
    with h5py.File(file_paths["Default"]) as file:
        n_jets = len(file["jets"])

    # define jet selections
    cuts = [("n_truth_promptLepton", "==", 0), ("pt", ">", 20000), ("pt", "<", 250000)]
    sample_name = "zprime" if "zprime" in txt_file else "ttbar"
    print(sample_name, weighting_str)
    sample_tag = (
        "$\\sqrt{s}=13$ TeV, dummy jets \n$t\\bar{t}$, $20$ GeV $< p_{T} <250$ GeV"
        if sample_name == "ttbar"
        else "$\\sqrt{s}=13$ TeV, dummy jets \n$Z'$, $20$ GeV $< p_{T} <250$ GeV"
    )

    # aux_results = AuxResults(sample=sample_name, output_dir=out_dir)
    # aux_results.atlas_second_tag = sample_tag

    # # exclusive vertexing -- no SV1 reference

    # logger.info("Loading taggers.")
    # create_and_load_tagger(fnames[0], labels[0], aux_results, cuts, n_jets, ref=True)
    # create_and_load_tagger(fnames[1], labels[1], aux_results, cuts, n_jets)
    # create_and_load_tagger(fnames[2], labels[2], aux_results, cuts, n_jets)
    # create_and_load_tagger(fnames[3], labels[3], aux_results, cuts, n_jets)
    # create_and_load_tagger(fnames[4], labels[4], aux_results, cuts, n_jets)
    # # create_and_load_tagger(fnames[5], labels[5], aux_results, cuts, n_jets)
    # # create_and_load_tagger(fnames[6], labels[6], aux_results, cuts, n_jets)
    # print(aux_results.taggers.values())
    # logger.info("Plotting exclusive vertexing performance.")
    # aux_results.plot_var_vtx_perf(vtx_flavours=["bjets"], no_vtx_flavours=["ujets"])

    aux_results = AuxResults(sample=sample_name, output_dir=out_dir)
    aux_results.atlas_second_tag = sample_tag

    logger.info("Loading taggers.")
    create_and_load_tagger(fnames[0], labels[0], aux_results, cuts, n_jets, ref=False)
    create_and_load_tagger(fnames[1], labels[1], aux_results, cuts, n_jets)
    create_and_load_tagger(fnames[2], labels[2], aux_results, cuts, n_jets)
    create_and_load_tagger(fnames[3], labels[3], aux_results, cuts, n_jets)
    create_and_load_tagger(fnames[4], labels[4], aux_results, cuts, n_jets)
    create_and_load_tagger(fnames[5], labels[5], aux_results, cuts, n_jets)
    create_and_load_tagger(fnames[6], labels[6], aux_results, cuts, n_jets)
    create_and_load_tagger(gn2_zprime, "SV1", aux_results, cuts, n_jets, ref=True, name="SV1")

    logger.info("Plotting inclusive vertexing performance.")
    aux_results.plot_var_vtx_perf(
        vtx_flavours=["bjets"], no_vtx_flavours=["ujets"], incl_vertexing=True
    )  # this is to compare old and new models
