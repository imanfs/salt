import h5py
import numpy as np
from puma.hlplots import Tagger


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


def create_taggers(fname, ref=None):
    name = extract_MF_name(fname)
    MF = Tagger(
        name="MaskFormer",
        label=f"MF {name}",
        reference=False,
        aux_tasks=["vertexing"],
        aux_labels=["ftagTruthParentBarcode"],
    )
    if ref is not None:
        ref_tagger = Tagger(
            name=ref,
            label=ref,
            colour="#000000",
            reference=True,
            aux_tasks=["vertexing"],
            aux_labels=["ftagTruthParentBarcode"],
        )
        return MF, ref_tagger
    return MF


def create_and_load_taggers(fname: str, aux_results, cuts, n_jets, ref=None) -> Tagger:
    """Create and load a Tagger object from a file."""
    if ref is not None:
        MF, ref_tagger = create_taggers(fname, ref)
        aux_results.load_taggers_from_file(
            [MF, ref_tagger], fname, aux_key="tracks", cuts=cuts, num_jets=n_jets
        )
    else:
        MF = create_taggers(fname)
        aux_results.load_taggers_from_file(
            [MF], fname, aux_key="tracks", cuts=cuts, num_jets=n_jets
        )


def del_new_tracks(fname: str):
    with h5py.File(fname, "r+") as file:
        if "new_tracks" in file:
            del file["new_tracks"]
            print("Existing 'new_tracks' dataset deleted.")
        else:
            print("Dataset 'new_tracks' does not exist.")
    return fname


def rename_vtxindex(fname: str, force: bool = False):
    # Open the HDF5 file in read/write mode
    with h5py.File(fname, "r+") as file:
        if "new_tracks" in file:
            if force:
                del file["new_tracks"]
                print("Existing 'new_tracks' dataset deleted.")
            else:
                print("Dataset 'new_tracks' already exists. Use 'force=True' to overwrite.")
                return fname

        # Access the existing dataset
        dataset = file["tracks"]

        # Define the new data type with the modified column name
        # Let's say you want to rename column 'MaskFormer_VertexIndex' to the new name
        old_dtype = dataset.dtype
        new_dtype = []
        model_name = extract_MF_name(fname)
        for name in old_dtype.names:
            if name == "MaskFormer_VertexIndex":
                new_dtype.append((f"MaskFormer_{model_name}_VertexIndex", old_dtype[name]))
            else:
                new_dtype.append((name, old_dtype[name]))

        new_dtype = np.dtype(new_dtype)

        # Create a new empty dataset with the new dtype
        new_data = np.zeros(dataset.shape, dtype=new_dtype)

        # Copy the data from the old dataset to the new dataset
        for name in old_dtype.names:
            if name == "MaskFormer_VertexIndex":
                print(dataset[name])
                new_data[f"MaskFormer_{model_name}_VertexIndex"] = dataset[name]
                print(new_data[f"MaskFormer_{model_name}_VertexIndex"])
            else:
                new_data[name] = dataset[name]

        file.create_dataset("new_tracks", data=new_data)
        print("New 'new_tracks' dataset created.")
    return fname
