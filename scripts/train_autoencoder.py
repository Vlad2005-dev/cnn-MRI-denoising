"""Train the denoiser on the training slices, monitor on the validation slices.

Usage:
    python scripts/train_autoencoder.py                 # 300 epochs, with skip connection
    python scripts/train_autoencoder.py --epochs 1000
    python scripts/train_autoencoder.py --no-skip

The checkpoint with the best validation MSE is saved (not just the last epoch).
The test set is never touched here; use evaluate.py for final numbers.
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common import MODELS, PLOTS, Denoiser, get_device, image_metrics, load_split, model_name, predict

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--no-skip", action="store_true", help="train the variant without the skip connection")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

torch.manual_seed(args.seed)
device = get_device()
skip = not args.no_skip
name = model_name(args.epochs, skip)

Xtr, Ytr = load_split("train")
Xva, Yva = load_split("val")
print(f"device: {device} | train: {len(Xtr)} images | val: {len(Xva)} images")

train_dl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=args.batch_size, shuffle=True)

model = Denoiser(skip=skip).to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.MSELoss()

hist = {"train_mse": [], "val_mse": [], "val_psnr": [], "val_ssim": []}
best = float("inf")

for epoch in range(1, args.epochs + 1):
    model.train()
    running = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        running += loss.item() * len(xb)

    pred = predict(model, Xva, device)
    val_mse = loss_fn(pred, Yva).item()
    psnr, ssim = image_metrics(pred.numpy(), Yva.numpy())

    hist["train_mse"].append(running / len(Xtr))
    hist["val_mse"].append(val_mse)
    hist["val_psnr"].append(float(psnr.mean()))
    hist["val_ssim"].append(float(ssim.mean()))

    if val_mse < best:
        best = val_mse
        torch.save(model.state_dict(), MODELS / f"{name}.pt")

    print(f"epoch {epoch:4d} | train MSE {hist['train_mse'][-1]:.5f} | val MSE {val_mse:.5f} "
          f"| val PSNR {psnr.mean():.2f} dB | val SSIM {ssim.mean():.4f}")

best_epoch = int(np.argmin(hist["val_mse"])) + 1
print(f"best val MSE {best:.5f} at epoch {best_epoch} -> saved {MODELS / (name + '.pt')}")
with open(MODELS / f"{name}_history.json", "w") as f:
    json.dump({"args": vars(args), "best_epoch": best_epoch, **hist}, f)

# ---------------- training curves ----------------
ep = np.arange(1, args.epochs + 1)            # start at 1 so the log axis works
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].loglog(ep, hist["train_mse"], label="train")
ax[0].loglog(ep, hist["val_mse"], label="validation")
ax[0].set(xlabel="Epoch", ylabel="MSE", title="MSE (log-log)")
ax[0].legend()
ax[1].plot(ep, hist["val_psnr"])
ax[1].set(xlabel="Epoch", ylabel="PSNR (dB)", title="Validation PSNR")
ax[2].plot(ep, hist["val_ssim"])
ax[2].set(xlabel="Epoch", ylabel="SSIM", title="Validation SSIM")
for a in ax:
    a.axvline(best_epoch, color="grey", ls="--", lw=0.8)
fig.suptitle(f"{name}  (best epoch {best_epoch}, dashed)")
plt.tight_layout()
plt.savefig(PLOTS / f"training_curves_{name}.png", dpi=200)
print(f"saved {PLOTS / f'training_curves_{name}.png'}")
