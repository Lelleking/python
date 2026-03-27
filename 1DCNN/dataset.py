import os
import glob
import numpy as np
from scipy.interpolate import interp1d
import torch
from torch.utils.data import Dataset

RESAMPLE_LEN = 1000


def load_all_curves(data_dir="compressed data"):
    """
    Reads all *_compressed.txt files from data_dir.
    Returns:
        curves   : np.ndarray of shape (N, RESAMPLE_LEN), dtype float32
                   Each row is one well, min-max normalised to [0, 1].
        metadata : list of (prefix, well_id) tuples, length N.
    """
    curves = []
    metadata = []

    files = sorted(glob.glob(os.path.join(data_dir, "*_compressed.txt")))
    if not files:
        raise FileNotFoundError(f"No *_compressed.txt files found in '{data_dir}'")

    for filepath in files:
        prefix = os.path.basename(filepath).replace("_compressed.txt", "")

        with open(filepath, "r") as f:
            lines = f.readlines()

        header = lines[0].strip().split("\t")
        well_ids = header[1:]   # skip "Time" column

        rows = []
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            rows.append([float(x) for x in parts])

        if not rows:
            continue

        data = np.array(rows, dtype=np.float32)   # (T, 1 + W)
        n_points = data.shape[0]
        x_orig = np.linspace(0.0, 1.0, n_points)
        x_new = np.linspace(0.0, 1.0, RESAMPLE_LEN)

        for col_idx, well_id in enumerate(well_ids):
            signal = data[:, col_idx + 1]   # skip Time column

            resampled = interp1d(x_orig, signal, kind="cubic")(x_new)

            lo, hi = resampled.min(), resampled.max()
            if hi - lo < 1e-6:
                # flat line — normalise to 0.5
                normalised = np.full(RESAMPLE_LEN, 0.5, dtype=np.float32)
            else:
                normalised = ((resampled - lo) / (hi - lo)).astype(np.float32)

            curves.append(normalised)
            metadata.append((prefix, well_id))

    return np.array(curves, dtype=np.float32), metadata


class FluorescenceDataset(Dataset):
    """
    PyTorch Dataset wrapping the normalised curve array.
    __getitem__ returns a tensor of shape (1, RESAMPLE_LEN).
    """

    def __init__(self, data_dir="compressed data"):
        self.curves, self.metadata = load_all_curves(data_dir)

    def __len__(self):
        return len(self.curves)

    def __getitem__(self, idx):
        # Add channel dimension: (1, L)
        return torch.from_numpy(self.curves[idx]).unsqueeze(0)