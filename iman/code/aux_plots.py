"""Produce aux task plots from tagger output and labels."""

from __future__ import annotations

import h5py
from puma.hlplots import AuxResults, Tagger
from puma.utils import logger

fname_arxiv = (
    "/home/xucabis2/salt/logs/MaskFormer_arxiv_20240724-T114414/"
    "ckpts/epoch=019-val_loss=0.65962__test_ttbar.h5"
)
fname_default = (
    "/home/xucabis2/salt/logs/MaskFormer_default_20240724-T112538/"
    "ckpts/epoch=019-val_loss=0.65355__test_ttbar.h5"
)

fname_equal = (
    "/home/xucabis2/salt/logs/MaskFormer_equal_20240724-T114419/ckpts/"
    "epoch=019-val_loss=0.64428__test_ttbar.h5"
)

fname_gls = (
    "/home/xucabis2/salt/logs/MaskFormer_GLS_20240730-T002427/ckpts/"
    "epoch=018-val_loss=0.65171__test_ttbar.h5"
)

file = h5py.File(fname_gls)
# define jet selections
cuts = [("n_truth_promptLepton", "==", 0), ("pt", ">", 20000), ("pt", "<", 250000)]

# define the tagger
MF = Tagger(
    name="MaskFormer_GLS",
    label="MF GLS",
    colour="#4477AA",
    reference=False,
    aux_tasks=["vertexing"],
)
# define the tagger
MF_default = Tagger(
    name="MaskFormer_default",
    label="MF default",
    reference=True,
    aux_tasks=["vertexing"],
)
# create the AuxResults object
aux_results = AuxResults(sample="ttbar-2")

# load tagger from the file object
logger.info("Loading taggers.")
aux_results.load_taggers_from_file(
    [MF],
    fname_gls,
    cuts=cuts,
    num_jets=len(file["jets"]),
)

aux_results.atlas_second_tag = (
    "$\\sqrt{s}=13$ TeV, dummy jets \n$t\\bar{t}$, $20$ GeV $< p_{T} <250$ GeV"
)

# vertexing performance for b-jets
logger.info("Plotting vertexing performance.")
aux_results.plot_var_vtx_perf(vtx_flavours=["bjets"], no_vtx_flavours=["ujets"])
aux_results.plot_var_vtx_perf(
    vtx_flavours=["bjets"], no_vtx_flavours=["ujets"], incl_vertexing=True
)