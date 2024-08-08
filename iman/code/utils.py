import h5py


def h5py_read(h5path, key, dataset=None, var_name=None):
    h5 = h5py.File(h5path, "r")
    # Iterate over items in the file
    if isinstance(h5[key], h5py.Dataset):
        # print(f"'{key}' is a dataset")
        if var_name is not None:
            data = h5[key][var_name]
        else:
            data = {name: h5[key][name] for name in h5[key].dtype.names}

    elif isinstance(h5[key], h5py.Group):
        # print(f"'{key}' is a group")

        if var_name is not None:
            print(h5[key][dataset].dtype.names)
            data = h5[key][dataset][var_name]

        else:
            data = {name: h5[key][dataset][name] for name in h5[key][dataset].dtype.names}

    return data


def extract_MF_name(path):
    prefix = "MaskFormer_"
    suffix = "_"
    return path.partition(prefix)[2].partition(suffix)[0]
