import h5py

logs_dir = "/home/xucabis2/salt/salt/logs"
ckpt = "epoch=019-val_loss=0.64715__test_ttbar.h5"
fname_preds = f"{logs_dir}/MF_16workers_10kbatch_20240710-T103121/ckpts/{ckpt}"
test_dir = "/home/xzcappon/phd/datasets"
fname_truth = f"{test_dir}/vertexing_120m/output/pp_output_test_ttbar.h5"
h5preds = h5py.File(fname_preds, "r")
objects = h5preds["objects"]  # hadrons
reg_preds = objects["regression"]
preds = reg_preds["regression_dphi"]

h5truth = h5py.File(fname_truth, "r")
truth_hadrons = h5truth["truth_hadrons"]
truth_hadrons = truth_hadrons[:500000]

print(truth_hadrons.dtype.names)
print(truth_hadrons["flavour"])
