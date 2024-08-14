import h5py
import numpy as np


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


def flavour_class_max(objects):
    class_probs = objects["object_class_probs"]
    class_probs = np.array(class_probs)
    class_probs_3d = class_probs.view((np.float32, 3))

    # Find the index of the maximum value across the 3rd dimension
    max_indices = np.argmax(class_probs_3d, axis=2)

    # Initialize the new column with zeros
    flavour = np.zeros(class_probs_3d.shape[:2], dtype=int)

    # Assign values based on the max index
    flavour[max_indices == 0] = 5
    flavour[max_indices == 1] = 4
    flavour[max_indices == 2] = -1

    return flavour
