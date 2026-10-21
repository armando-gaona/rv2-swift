"""RV2-Swift: faster Reader v2 / Reader V3 inference on GPU, same checkpoint.

Drop-in wrapper around villa's `koine_machines.inference.infer`. It takes the same arguments and adds two changes:

1. Larger sliding window (--tile, default 384 px). The network is built from the checkpoint exactly as before, only the
   inference window grows. With Reader v2 (trained at 128) this changes the map; use --tile 0 to keep the checkpoint's
   own window (e.g. Reader V3, trained at 384).
2. Robust normalization on the GPU. Upstream, every DataLoader worker runs normalize_robust (percentiles, median, MAD)
   on the CPU for each patch, and on a GPU machine that becomes the bottleneck. Here workers hand over the raw uint8
   patch and the GPU normalizes the whole batch with the same math (linear quantiles, same fallbacks).
   Reader v2: identical to the CPU path (max abs difference 0.0); Reader V3: 0.002 % of pixels differ by 1/255.
   Without CUDA this part does nothing.

Usage (same positional arguments as infer.py):
    python rv2_swift.py <input.zarr> <reader-v2.pth> <out.tif> --stride 256 --blend-mode hann \
        --batch-size 4 --num-workers 4 --no-compile --direction forward
Extra options:
    --tile N          inference window in px (default 384; 0 = the checkpoint's own window)
    --no-gpu-norm     keep normalization on the CPU workers
"""
import sys

import numpy as np
import torch
import koine_machines.inference.infer as I

_LIMIT = 16_000_000   # torch.quantile refuses larger tensors


def _quantile(x, q):
    if x.numel() <= _LIMIT:
        return torch.quantile(x, q, dim=1)
    return torch.stack([torch.quantile(row, q) for row in x])


def normalize_batch(x):
    """[B, ...] uint8 or float tensor on the GPU -> float32, normalized per sample like vesuvius normalize_robust."""
    shape = x.shape
    f = x.reshape(shape[0], -1).float()
    sd = f.std(dim=1, unbiased=False)
    lo, hi = _quantile(f, 0.01), _quantile(f, 0.99)
    f = torch.maximum(torch.minimum(f, hi[:, None]), lo[:, None])
    med = _quantile(f, 0.5)
    mad = 1.4826 * _quantile((f - med[:, None]).abs(), 0.5)
    fallback = torch.where(sd.abs() >= 1e-6, sd.abs(), (hi - lo).abs() / 2)
    fallback = torch.where(fallback >= 1e-6, fallback, torch.ones_like(fallback))
    scale = torch.where(torch.isfinite(mad) & (mad >= 1e-6), mad, fallback)
    out = (f - med[:, None]) / scale[:, None]
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).reshape(shape)


class NormalizeOnGPU(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        with torch.autocast("cuda", enabled=False):
            x = normalize_batch(x)
        return self.model(x)


def enable_gpu_norm():
    if not torch.cuda.is_available():
        return False
    dataset = I.OmeZarrBlockDataset
    original = dataset.__getitem__

    def getitem(self, index):
        if self.preprocessing != "tifxyz_robust":
            return original(self, index)
        b = self.blocks[index]
        patch = self.reader.read(b.y0, b.x0, self.patch_h, self.patch_w)
        meta = torch.tensor([b.y0, b.x0, b.valid_h, b.valid_w, int(patch.any())], dtype=torch.int64)
        return torch.from_numpy(np.ascontiguousarray(np.moveaxis(patch, -1, 0))).unsqueeze(0), meta

    dataset.__getitem__ = getitem
    build = I.maybe_compile_model
    I.maybe_compile_model = lambda model, **kw: NormalizeOnGPU(build(model, **kw))
    return True


def self_test(n=3, shape=(17, 384, 384)):
    """Max abs difference between vesuvius normalize_robust (CPU) and normalize_batch (GPU). Expected: 0.0"""
    from vesuvius.image_proc.intensity.normalization import normalize_robust
    rng = np.random.default_rng(0)
    err = 0.0
    for i in range(n):
        p = rng.integers(0, 256, shape, dtype=np.uint8) if i else np.full(shape, 7, np.uint8)   # i=0 hits the fallback
        a = normalize_robust(p)
        b = normalize_batch(torch.from_numpy(p)[None].cuda())[0].cpu().numpy()
        err = max(err, float(np.abs(a - b).max()))
    return err


def _pop_option(argv, name, default, cast=str):
    if name in argv:
        i = argv.index(name)
        value = cast(argv[i + 1])
        del argv[i:i + 2]
        return value
    return default


# Patched at import time on purpose: infer.py starts its workers with spawn, which re-imports this module.
_argv = sys.argv[1:]
TILE = _pop_option(_argv, "--tile", 384, int)
GPU_NORM = "--no-gpu-norm" not in _argv
_argv = [a for a in _argv if a != "--no-gpu-norm"]
if TILE:
    I.roi_size_from_config = lambda config: TILE
if GPU_NORM:
    enable_gpu_norm()

if __name__ == "__main__":
    if _argv[:1] == ["--self-test"]:
        print("max abs difference vs normalize_robust:", self_test())
        sys.exit(0)
    sys.exit(I.main(_argv))
