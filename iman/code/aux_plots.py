"""Produce aux task plots from tagger output and labels."""

from __future__ import annotations

import h5py
from puma.hlplots import AuxResults
from puma.utils import logger
from utils import create_and_load_taggers

fname_arxiv = (
    "/home/xucabis2/salt/logs/MaskFormer_arxiv_20240724-T114414/"
    "ckpts/epoch=019-val_loss=0.65962__test_ttbar.h5"
)
fname_default = (
    "/home/xucabis2/salt/logs/MaskFormer_default_20240811-T102933/"
    "ckpts/epoch=019-val_loss=0.65148__test_ttbar.h5"
)
# fname_default = (
#     "/home/xucabis2/salt/logs/_old/_pre-b-hadron-weighting/_baselines/"
#     "MaskFormer_default_20240724-T112538/ckpts/epoch=019-val_loss=0.65355__test_ttbar.h5"
# )

fname_equal = (
    "/home/xucabis2/salt/logs/MaskFormer_equal_20240724-T114419/ckpts/"
    "epoch=019-val_loss=0.64428__test_ttbar.h5"
)

fname_gls = (
    "/home/xucabis2/salt/logs/MaskFormer_GLS_20240809-T191439/"
    "ckpts/epoch=018-val_loss=0.65630__test_ttbar.h5"
)

fname_dwa = (
    "/home/xucabis2/salt/logs/MaskFormer_DWA_20240811-T105042/"
    "ckpts/epoch=019-val_loss=0.64261__test_ttbar.h5"
)

fname_aligned = (
    "/home/xucabis2/salt/logs/MaskFormer_AlignedMTL_20240821-T131155/"
    "ckpts/epoch=019-val_loss=0.64403__test_ttbar.h5"
)

fname_cagrad = (
    "/home/xucabis2/salt/logs/MaskFormer_CAGrad_20240821-T141325/"
    "ckpts/epoch=019-val_loss=0.64449__test_ttbar.h5"
)

fname_pcgrad = (
    "/home/xucabis2/salt/logs/MaskFormer_PCGrad_20240821-T140219/"
    "ckpts/epoch=019-val_loss=0.64397__test_ttbar.h5"
)

fnames_preds = [fname_default, fname_aligned, fname_cagrad, fname_pcgrad]


# define jet selections
cuts = [("n_truth_promptLepton", "==", 0), ("pt", ">", 20000), ("pt", "<", 250000)]
with h5py.File(fname_default) as file:
    n_jets = len(file["jets"])

# create the AuxResults object
ref = None
out_dir = "/home/xucabis2/salt/iman/plots/vertexing"
out_dir += "" if ref is not None else "/no_ref"
print(out_dir)
aux_results = AuxResults(sample="ttbar-2", output_dir=out_dir)
aux_results.atlas_second_tag = (
    "$\\sqrt{s}=13$ TeV, dummy jets \n$t\\bar{t}$, $20$ GeV $< p_{T} <250$ GeV"
)

logger.info("Loading taggers.")
create_and_load_taggers(fname_default, aux_results, cuts, n_jets, ref=ref)
create_and_load_taggers(fname_aligned, aux_results, cuts, n_jets)
create_and_load_taggers(fname_cagrad, aux_results, cuts, n_jets)
create_and_load_taggers(fname_pcgrad, aux_results, cuts, n_jets)


logger.info("Plotting vertexing performance.")
aux_results.plot_var_vtx_perf(vtx_flavours=["bjets", "cjets"], no_vtx_flavours=["ujets"])
aux_results.plot_var_vtx_perf(
    vtx_flavours=["bjets", "cjets"], no_vtx_flavours=["ujets"], incl_vertexing=True
)  # this is to compare old and new models
