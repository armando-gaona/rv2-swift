<p align="center"><img src="assets/hero.png" alt="Labelled ink, Reader v2 and RV2-Swift side by side on PHerc. 0841" width="100%"></p>

# RV2-Swift

**Faster Reader v2 inference on GPU, with the published checkpoint and no retraining.**

RV2-Swift is a single-file wrapper around villa's ink inference (`koine_machines.inference.infer`). It takes
the same command-line arguments and adds two independent changes:

1. **GPU normalization.** The robust normalization that the DataLoader workers run on the CPU for every patch is
   done on the GPU for the whole batch, with the same math. This works with any checkpoint, including
   [Reader V3](https://huggingface.co/domenicor046/reader-v3), and leaves the ink map practically unchanged.
2. **A larger window for Reader v2 (optional).** Reader v2 is read through a 384 px window instead of the 128 px
   it was trained on. This is where most of the speed comes from with Reader v2, but it is a different way of
   running the model and it changes the map (see [What the larger window changes](#what-the-larger-window-changes)).

| | upstream | RV2-Swift | |
|---|---|---|---|
| **Reader V3**, real segment PHerc. 0139 w043 (28 x 6120 x 8120), NVIDIA L4, GPU normalization only | 65.4 s | **47.9 s** | **1.37x** |
| **Reader v2**, same segment and GPU, GPU normalization + 384 px window | 62.8 s | **32.0 s** | **1.96x** |
| Reader v2, NVIDIA T4 (Kaggle), throughput on large surfaces | 0.52 Mpx/s | **1.38 Mpx/s** | **2.65x** |
| Reader v2, NVIDIA L4 (Colab), throughput on large surfaces | 0.99 Mpx/s | **2.08 Mpx/s** | **2.10x** |
| Reader v2, CPU only (i7-7820HK laptop, one 0841 segment), 384 px window | 190 s | **101 s** | **1.9x** |

Wall times include Python start-up, imports and model loading (about 10 s, the same for both), so the gain
grows with the size of the surface. On small pieces it is modest: the three 0841 segments below (about 5
Mpx each, mostly masked) take 14.3 to 14.9 s upstream and 12.2 to 12.6 s with RV2-Swift on the same L4.

<p align="center"><img src="assets/speed.png" alt="Throughput of Reader v2 and RV2-Swift on T4 and L4" width="85%"></p>

## GPU normalization: same map, less waiting

Upstream, every DataLoader worker runs `normalize_robust` (1st and 99th percentiles, median, MAD) on the CPU for
every patch, about 0.17 s per 17 x 384 x 384 patch. With large tiles the GPU spends much of its time waiting
for that. RV2-Swift lets the workers ship the raw `uint8` patch (4x less data to move) and normalizes the whole
batch on the GPU: linear quantiles as in `np.percentile`, the same MAD scale and the same fallbacks for flat
patches.

* **Reader v2:** the output is bit-identical to the CPU path (max abs difference 0.0 on full segments, at 128 and
  384 px).
* **Reader V3:** on the full w043 segment, 0.002 % of the pixels differ by 1 level out of 255 (float rounding on
  the GPU). For scale, changing the batch size from 8 to 16 in plain upstream V3 changes 0.66 % of the pixels.
  Upstream V3 itself is deterministic run to run.

GPU normalization only pays off when the tiles are large enough to keep the GPU busy. With Reader v2's native
128 px tiles it gives nothing on an L4 (0.91x); at 384 px it does (Reader V3 1.37x above).

## What the larger window changes

Reader v2 was trained on 128 px tiles. Its InstanceNorm layers take their statistics from whatever window
they are given, so reading it through 384 px windows produces a slightly different ink map, not the same one
faster. We checked that map only one way: AUC against the ink labels of the three PHerc. 0841 segments, a scroll
Reader v2 was not trained on (0.5 is chance, 1.0 is perfect).

| segment of PHerc. 0841 | Reader v2 | Reader v2, 384 px window |
|---|---|---|
| auto_grown_20260220174252405 | 0.816 | 0.823 |
| w00 | 0.878 | 0.890 |
| auto_grown_20260220144552896 | 0.813 | 0.826 |

It did not score lower on any of the three. It has **not** been checked for false ink on blank papyrus or on
116 keV scans. If you want a model that reads with 384 px of context, Reader V3 was trained for exactly that
and tested on those cases; use RV2-Swift on it with `--tile 0` to keep only the GPU normalization.

## Install

RV2-Swift runs inside a normal Reader v2 / Reader V3 setup:

```bash
git clone https://github.com/ScrollPrize/villa.git
cd villa && git checkout 3ea17f54a9b3d5fd1aaf73e1d2c8386dbaa9f30e   # merge-ink-pipelines, the version we tested
# install ink-detection as described in villa, then fetch a checkpoint:
#   huggingface.co/domenicor046/reader-v2 (revision 72633d8) or huggingface.co/domenicor046/reader-v3
```

Copy `rv2_swift.py` anywhere and run it with the villa environment active. It needs nothing beyond
what the readers already use (PyTorch, numpy).

## Use

Same positional arguments as `koine_machines.inference.infer`:

```bash
# Reader V3: GPU normalization only, the checkpoint's own 384 px window
python rv2_swift.py surface.zarr reader-v3.pth ink.tif --tile 0 \
    --overlap 0.5 --blend-mode hann --batch-size 8 --num-workers 4 --no-compile --direction forward

# Reader v2: GPU normalization + 384 px window
python rv2_swift.py surface.zarr reader-v2.pth ink.tif \
    --stride 256 --blend-mode hann --batch-size 4 --num-workers 4 --no-compile --direction forward
```

| option | default | meaning |
|---|---|---|
| `--tile N` | 384 | inference window in pixels; `0` keeps the checkpoint's own window |
| `--no-gpu-norm` | off | keep normalization on the CPU workers |
| `--self-test` | | compares GPU and CPU normalization, prints the max difference |

Every other `infer.py` option passes straight through. The same two changes are proposed for villa's
`infer.py` as the flags `--gpu-normalize` and `--roi-size`
([ScrollPrize/villa#2189](https://github.com/ScrollPrize/villa/pull/2189)).

## How we measured

* **Speed, real data:** the public PHerc. 0139 segment `20260112000000-w043_2026011217` (9.362 um surface volume,
  28 x 6120 x 8120), full segment, 4 workers, NVIDIA L4, each configuration run twice back to back.
  Reader V3: overlap 0.5, Hann, batch 8 (67.0 / 63.7 s upstream, 47.7 / 48.1 s with GPU normalization, 1,278
  patches). Reader v2: upstream defaults (128 px, overlap 0.5, batch 32; 62.8 / 62.8 s, 11,006 patches) against
  384 px, stride 256, batch 4 with GPU normalization (32.0 / 32.1 s, 745 patches).
* **Speed, throughput:** synthetic 17-layer surfaces of 2048 and 6144 px; throughput is taken from the difference
  between the two sizes, so start-up is excluded.
* **Quality:** the three labelled 0841 segments, full segment, forward direction, AUC over labelled pixels inside
  the valid mask; upstream Reader v2 with its defaults.
* **Identity:** GPU-normalized output compared pixel by pixel with the CPU-normalized output of the same
  settings, on full segments.

## Limits

* Speed was measured on T4, L4 and a GTX 1070, not on A100.
* Tested with `--no-compile`. `torch.compile` should work but is untested.
* Tested at villa commit `3ea17f5`. The wrapper patches two functions of `infer.py`
  (`roi_size_from_config`, `maybe_compile_model`) and the dataset's `__getitem__`; a large rewrite of `infer.py`
  upstream may need a small update here.

## Credits

Reader v2 and Reader V3 are by Domenico Russo (`domenicor046/reader-v2`, `domenicor046/reader-v3`, MIT), inference
code from [ScrollPrize/villa](https://github.com/ScrollPrize/villa). RV2-Swift only changes how inference is run.
MIT license.
