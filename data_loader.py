"""PLCI-compatible two-view data loading for NEWSTRIDE.

The loader has one deliberate convention: ``data_X[0]`` is the fixed anchor
view and ``data_X[1]`` is the view whose correspondence is corrupted.  The
corruption procedure follows PLCI: a fixed ``RandomState(1234)`` selects the
affected rows and independently permutes their sources.
"""

from pathlib import Path

import numpy as np
import scipy.io as sio
import sklearn.preprocessing as skp
import torch
from torch.utils.data import Dataset


DATASET_VIEWS = {
    # fixed view first, disturbed view second
    "Scene15": ("Scene_15.mat", (1, 0)),
    "Handwritten": ("Handwritten.mat", (0, 1)),
    "CCV20": ("CCV20.mat", (1, 2)),
    "Food101": ("umpc_food101.mat", (0, 1)),
}


def _find_file(data_path, names):
    root = Path(data_path)
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find any of {list(names)} under {root.resolve()}"
    )


def _matlab_cells(value):
    """Flatten MATLAB cell arrays in MATLAB/Fortran order."""
    if not isinstance(value, np.ndarray) or value.dtype != object:
        raise ValueError("Expected a MATLAB cell array")
    return [np.asarray(cell) for cell in np.ravel(value, order="F")]


def _load_views(args):
    dataset = str(args.dataset)
    dataset = {
        "HandWritten": "Handwritten",
        "umpc_food101": "Food101",
        "UMPC-Food101": "Food101",
        "UMPC_Food101": "Food101",
    }.get(dataset, dataset)
    if dataset not in DATASET_VIEWS:
        raise KeyError(f"NEWSTRIDE supports only {sorted(DATASET_VIEWS)}; got {args.dataset!r}")

    default_file, default_indices = DATASET_VIEWS[dataset]
    if dataset == "CCV20":
        path = _find_file(args.data_path, ("CCV20.mat", "CCV.mat"))
    elif dataset == "Food101":
        path = _find_file(args.data_path, ("umpc_food101.mat", "UMPC_Food101.mat"))
    else:
        names = (default_file,)
        if dataset == "Handwritten":
            names = ("Handwritten.mat", "HandWritten.mat")
        path = _find_file(args.data_path, names)

    mat = sio.loadmat(path)

    if dataset == "Food101":
        views = [
            np.asarray(mat["img_features"]),
            np.asarray(mat["txt_features"]),
        ]
        labels = np.asarray(mat["labels"]).reshape(-1)
    else:
        cells = _matlab_cells(mat["X"])
        indices = tuple(getattr(args, "view_indices", default_indices))
        if len(indices) != 2:
            raise ValueError("view_indices must contain exactly two MATLAB cell indices")
        try:
            views = [cells[indices[0]], cells[indices[1]]]
        except IndexError as exc:
            raise IndexError(
                f"view_indices={indices} is invalid for {len(cells)} MATLAB views"
            ) from exc
        label_key = "Y" if "Y" in mat else "y" if "y" in mat else "gt"
        labels = np.asarray(mat[label_key]).reshape(-1)

    data = [np.asarray(view, dtype=np.float32) for view in views]
    if any(view.ndim != 2 for view in data):
        raise ValueError(f"All views must be 2-D; got {[view.shape for view in data]}")
    if data[0].shape[0] != data[1].shape[0]:
        raise ValueError(f"Views have different sample counts: {[view.shape for view in data]}")
    if labels.shape[0] != data[0].shape[0]:
        raise ValueError("Label count does not match the number of samples")

    labels = labels.astype(np.int64, copy=False)
    labels -= labels.min()
    return data, labels, path


def _normalize(data, mode):
    if mode == "standard":
        return [skp.scale(view).astype(np.float32) for view in data]
    if mode == "l2-norm":
        return [skp.normalize(view).astype(np.float32) for view in data]
    if mode == "min-max":
        return [skp.minmax_scale(view).astype(np.float32) for view in data]
    raise ValueError(f"Unknown data_norm={mode!r}")


def _plci_shuffle_view2(data, labels, ratio):
    """Apply PLCI's fixed-seed partial correspondence shuffle to view 2."""
    noisy = [view.copy() for view in data]
    labels_view2 = labels.copy()
    ids_view1 = np.arange(len(labels), dtype=np.int64)
    ids_view2 = ids_view1.copy()
    is_align = np.ones(len(labels), dtype=np.int64)

    if ratio <= 0:
        return noisy, labels_view2, ids_view1, ids_view2, is_align
    if ratio > 1:
        raise ValueError(f"m_ratio must be in [0, 1], got {ratio}")

    rng = np.random.RandomState(1234)
    indices = np.arange(len(labels), dtype=np.int64)
    rng.shuffle(indices)
    indices = indices[: int(ratio * len(indices))]
    is_align[indices] = 0

    source_indices = indices.copy()
    rng.shuffle(source_indices)
    noisy[1][indices] = noisy[1][source_indices]
    labels_view2[indices] = labels_view2[source_indices]
    ids_view2[indices] = ids_view1[source_indices]
    return noisy, labels_view2, ids_view1, ids_view2, is_align


class MultiviewDataset(Dataset):
    def __init__(self, data, label1, label2, id1, id2, is_align):
        self.data = data
        self.label1 = label1
        self.label2 = label2
        self.id1 = id1
        self.id2 = id2
        self.is_align = is_align

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, index):
        views = [torch.from_numpy(view[index].astype(np.float32, copy=False))
                 for view in self.data]
        return (
            index,
            views,
            torch.tensor(self.label1[index], dtype=torch.long),
            torch.tensor(self.label2[index], dtype=torch.long),
            torch.tensor(self.id1[index], dtype=torch.long),
            torch.tensor(self.id2[index], dtype=torch.long),
            torch.tensor(self.is_align[index], dtype=torch.long),
        )


def load_dataset(args):
    clean_data, labels, source_path = _load_views(args)

    # PLCI normalizes before applying the correspondence corruption.  Keep
    # that ordering so standard/min-max normalization has identical semantics.
    clean_data = _normalize(clean_data, args.data_norm)
    noisy_data, noisy_labels, id1, id2, is_align = _plci_shuffle_view2(
        clean_data, labels, args.m_ratio
    )

    expected_dims = [int(dim[0]) for dim in args.encoder_dim]
    actual_dims = [view.shape[1] for view in clean_data]
    if actual_dims != expected_dims:
        raise ValueError(
            f"{args.dataset} feature dimensions disagree with YAML: "
            f"actual={actual_dims}, configured={expected_dims}, source={source_path}"
        )

    clean_ids = np.arange(len(labels), dtype=np.int64)
    clean_align = np.ones(len(labels), dtype=np.int64)
    train_dataset = MultiviewDataset(
        noisy_data, labels, noisy_labels, id1, id2, is_align
    )
    test_dataset = MultiviewDataset(
        [view.copy() for view in clean_data],
        labels,
        labels,
        clean_ids,
        clean_ids,
        clean_align,
    )
    args.n_sample = len(labels)
    args.data_source = str(source_path)
    return train_dataset, test_dataset
