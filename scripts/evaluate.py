"""Final evaluation on the held-out TEST slices.

Compares three things against the clean images, using PSNR and SSIM:
  1. the noisy input (no denoising)        -> the starting point
  2. a Gaussian filter baseline             -> what a classical method gets
  3. the trained CNN                        -> what the network adds

The Gaussian sigma is chosen on the VALIDATION slices, so the baseline is
tuned fairly and the test set is still only used once.

Usage:
    python scripts/evaluate.py                                   # evaluates every model in trained_models/
    python scripts/evaluate.py --models autoencoder_300epochs
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from common import EXAMPLES, MODELS, RESULTS, Denoiser, get_device, image_metrics, load_split, predict

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="*", help="model names (without .pt); default: all in trained_models/")
args = p.parse_args()

names = args.models or sorted(f.stem for f in MODELS.glob("*.pt"))
if not names:
    raise SystemExit("No trained models found. Run scripts/train_autoencoder.py first.")

device = get_device()
Xva, Yva = load_split("val")
Xte, Yte = load_split("test")


def gauss(x, s):
    return np.stack([gaussian_filter(im, s) for im in x[:, 0].numpy()])


# ---- tune Gaussian sigma on validation ----
sigmas = np.round(np.arange(0.3, 2.01, 0.1), 2)
val_psnr = [image_metrics(gauss(Xva, s), Yva.numpy())[0].mean() for s in sigmas]
best_sigma = float(sigmas[int(np.argmax(val_psnr))])
print(f"Gaussian baseline: best sigma on validation = {best_sigma} px")

# ---- evaluate on test ----
preds = {"Noisy input": Xte[:, 0].numpy(), f"Gaussian filter (σ={best_sigma})": gauss(Xte, best_sigma)}
for n in names:
    model = Denoiser(skip="no_skip" not in n).to(device)
    model.load_state_dict(torch.load(MODELS / f"{n}.pt", map_location=device))
    preds[n] = predict(model, Xte, device)[:, 0].numpy()

rows = []
for method, pr in preds.items():
    psnr, ssim = image_metrics(pr, Yte.numpy())
    mse = float(((pr - Yte[:, 0].numpy()) ** 2).mean())
    rows.append({"method": method, "mse": mse,
                 "psnr_mean": float(psnr.mean()), "psnr_std": float(psnr.std()),
                 "ssim_mean": float(ssim.mean()), "ssim_std": float(ssim.std())})

print(f"\nTest set: {len(Xte)} images (held-out z slices)\n")
print(f"{'method':38s} {'MSE':>9s} {'PSNR (dB)':>15s} {'SSIM':>17s}")
for r in rows:
    print(f"{r['method']:38s} {r['mse']:9.5f} {r['psnr_mean']:7.2f} ± {r['psnr_std']:4.2f} "
          f"{r['ssim_mean']:8.4f} ± {r['ssim_std']:.4f}")

with open(RESULTS / "test_metrics.json", "w") as f:
    json.dump({"gaussian_sigma": best_sigma, "n_test_images": len(Xte), "results": rows}, f, indent=2)
print(f"\nsaved {RESULTS / 'test_metrics.json'}")

# ---- example figure: one image from each test slice ----
D = np.load(MODELS.parent / "data" / "mri_denoising_split.npz")
test_z = D["z"][D["split"] == "test"]
idx = [int(np.where(test_z == z)[0][0]) for z in np.unique(test_z)]

cols = list(preds.items()) + [("Clean", Yte[:, 0].numpy())]
fig, axes = plt.subplots(len(idx), len(cols), figsize=(2.4 * len(cols), 3.0 * len(idx)), squeeze=False)
for r, i in enumerate(idx):
    for c, (title, arr) in enumerate(cols):
        ax = axes[r, c]
        ax.imshow(arr[i], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if title != "Clean":
            ps, ss = image_metrics(arr[i:i + 1], Yte[i:i + 1].numpy())
            ax.set_title(f"{title}\n{ps[0]:.1f} dB / SSIM {ss[0]:.3f}", fontsize=7)
        else:
            ax.set_title(f"Clean (z={test_z[i]})", fontsize=7)
plt.tight_layout()
plt.savefig(EXAMPLES / "test_examples.png", dpi=200)
print(f"saved {EXAMPLES / 'test_examples.png'}")

# ---- v1-style 5x3 figure (Noisy / Denoised / Clean) for each model, test images only ----
rng = np.random.default_rng(1)
pick = np.sort(rng.choice(len(Xte), 5, replace=False))
for n in names:
    fig, axes = plt.subplots(5, 3, figsize=(6, 10))
    for r, i in enumerate(pick):
        for c, (title, arr) in enumerate((("Noisy", preds["Noisy input"]), ("Denoised", preds[n]),
                                          ("Clean", Yte[:, 0].numpy()))):
            axes[r, c].imshow(arr[i], cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
            if title == "Clean":
                axes[r, c].set_title(f"Clean (z={test_z[i]})", fontsize=9)
            else:
                ps, ss = image_metrics(arr[i:i + 1], Yte[i:i + 1].numpy())
                axes[r, c].set_title(f"{title}\n{ps[0]:.1f} dB / SSIM {ss[0]:.3f}", fontsize=9)
    fig.suptitle(f"{n} — held-out test slices", fontsize=10)
    plt.tight_layout()
    plt.savefig(EXAMPLES / f"denoising_demo_{n}.png", dpi=200)
    print(f"saved {EXAMPLES / f'denoising_demo_{n}.png'}")
