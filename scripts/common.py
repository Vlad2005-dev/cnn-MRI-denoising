"""Shared pieces: model, data loading, metrics, paths.

Keeping these in one place means training and evaluation always use the
exact same architecture and the exact same train/val/test split.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parent.parent          # project root, so scripts work from any cwd
DATA_NII = ROOT / "data" / "EPI_Head_ep2d_bold_moco_10.nii.gz"
DATA_NPZ = ROOT / "data" / "mri_denoising_split.npz"
MODELS = ROOT / "trained_models"
PLOTS = ROOT / "plots"
EXAMPLES = ROOT / "denoising_examples"
RESULTS = ROOT / "results"


class Denoiser(nn.Module):
    """Same architecture as v1. skip=False reproduces the 'no_skip' variant."""

    def __init__(self, skip=True):
        super().__init__()
        self.skip = skip
        self.enc1 = nn.Conv2d(1, 16, 3, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dec1 = nn.Conv2d(32, 16, 3, padding=1)
        self.dec2 = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        x1 = torch.relu(self.enc1(x))                     # low-level features, 64x64
        x3 = torch.relu(self.enc2(self.pool(x1)))         # 32x32
        x4 = nn.functional.interpolate(x3, scale_factor=2, mode="bilinear")   # back to 64x64
        x5 = torch.relu(self.dec1(x4))
        x6 = self.dec2(x5 + x1 if self.skip else x5)      # skip connection
        return torch.sigmoid(x6)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_split(name):
    """Return (noisy, clean) float32 tensors of shape (N,1,H,W) for 'train', 'val' or 'test'."""
    D = np.load(DATA_NPZ)
    m = D["split"] == name
    noisy = torch.from_numpy(D["noisy"][m][:, None])
    clean = torch.from_numpy(D["clean"][m][:, None])
    return noisy, clean


def model_name(epochs, skip):
    return f"autoencoder_{epochs}epochs" + ("" if skip else "_no_skip")


def image_metrics(pred, target):
    """Per-image PSNR (dB) and SSIM. Images are in [0, 1], so data_range=1.

    pred, target: arrays of shape (N,H,W) or (N,1,H,W).
    Returns two arrays of length N.
    """
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, *np.shape(target)[-2:])
    target = np.asarray(target, dtype=np.float64).reshape(pred.shape)
    psnr = np.array([peak_signal_noise_ratio(t, p, data_range=1.0) for p, t in zip(pred, target)])
    ssim = np.array([structural_similarity(t, p, data_range=1.0) for p, t in zip(pred, target)])
    return psnr, ssim


@torch.no_grad()
def predict(model, noisy, device, batch_size=256):
    model.eval()
    out = [model(noisy[i:i + batch_size].to(device)).cpu() for i in range(0, len(noisy), batch_size)]
    return torch.cat(out)
