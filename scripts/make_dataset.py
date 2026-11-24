"""Build the paired clean/noisy dataset with a leakage-free train/val/test split.

Why the split is by slice position (z) and not random:
The phantom is static for the first ~146 timepoints, and the signal varies by
only ~0.5% between timepoints. So slice z at time t is practically the same
image as slice z at any other time t'. In v1, the 1000 images were really
10 distinct slices x 100 noise realisations, and a random 80/20 split put
copies of every validation slice into the training set.

Here every z position belongs to exactly one split. Neighbouring slices are
still ~97% correlated, so the val and test blocks are surrounded by a
one-slice buffer that is not used at all.
"""
import nibabel as nib
import numpy as np

from common import DATA_NII, DATA_NPZ

N_TIME = 100          # don't go above ~146, the phantom starts moving there
NOISE_STD = 0.06

# Slices 3..24 contain the phantom (z>=25 is essentially empty, z<3 is a thin edge).
VAL_Z = [10, 11]
TEST_Z = [17, 18]
BUFFER_Z = [9, 12, 16, 19]                          # unused, separate val/test from train
TRAIN_Z = [z for z in range(3, 25) if z not in VAL_Z + TEST_Z + BUFFER_Z]

V = nib.load(DATA_NII).get_fdata().astype(np.float32)   # (64, 64, 30, 300) = (X, Y, Z, time)
V = V[..., :N_TIME]

# Robust intensity scaling. Percentiles come from the TRAINING slices only,
# so nothing about the val/test slices influences preprocessing.
vmin, vmax = np.percentile(V[:, :, TRAIN_Z, :], (1, 99))
V = np.clip(V, vmin, vmax)
V = (V - vmin) / (vmax - vmin + 1e-6)

clean, split, zs, ts = [], [], [], []
for name, zlist in (("train", TRAIN_Z), ("val", VAL_Z), ("test", TEST_Z)):
    for z in zlist:
        for t in range(N_TIME):
            clean.append(V[:, :, z, t])
            split.append(name)
            zs.append(z)
            ts.append(t)

clean = np.stack(clean).astype(np.float32)          # (N, 64, 64)
rng = np.random.default_rng(0)                      # seeded for reproducibility
noisy = np.clip(clean + NOISE_STD * rng.standard_normal(clean.shape, dtype=np.float32), 0, 1)

np.savez_compressed(DATA_NPZ, clean=clean, noisy=noisy.astype(np.float32),
                    split=np.array(split), z=np.array(zs), t=np.array(ts))

for name, zlist in (("train", TRAIN_Z), ("val", VAL_Z), ("test", TEST_Z)):
    print(f"{name:5s}: z = {zlist}  ->  {len(zlist) * N_TIME} images")
print(f"unused buffer slices: {BUFFER_Z}")
print(f"saved {DATA_NPZ}")
