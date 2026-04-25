# Method implementation and comparative results

This document complements the strong headline results with a direct comparison against the other methods and a concrete implementation map for reproducing the pipeline.

## 1. Comparative results

The package now includes two kinds of comparison:

- **Absolute comparison**: accuracy, ECE, and energy/image for all methods.
- **Delta-to-BioGater comparison**: how much each baseline trails or exceeds the BioGater reference on each metric.

Files generated in `results/`:

- `comparative_results.csv`
- `comparative_deltas_vs_biogater.csv`
- `comparative_accuracy.png/.jpg`
- `comparative_ece.png/.jpg`
- `comparative_energy.png/.jpg`
- `comparative_deltas_vs_biogater.png/.jpg`

## 2. Implemented method families

### A. HistoDyn::BioGater

Implemented in `histodyn/models.py` via:

- `BioGaterBlock`
- `build_histodyn_biogater(...)`

Core components:

1. Patch embedding by strided convolution.
2. Token pre-normalization.
3. Depthwise pre-mix spatial context.
4. Cheap biologically motivated operation bank:
   - `std_norm`
   - `scale_large`
   - `sign`
   - `binarize`
   - `sobel_mag`
   - `mean_subtract`
   - `reverse`
   - plus auxiliary operations used for ablation and routing behavior.
5. Learnable scalar gates with L1 regularization.
6. Soft routing during training and top-k routing during evaluation.
7. Residual projection and classifier head.

### B. RandomOp Mixer

Implemented as a baseline where one cheap operation is randomly selected from a fixed operation bank.

### C. ConvMixer (static)

Implemented with:

- depthwise convolution for spatial mixing
- pointwise convolution for channel mixing
- batch normalization and GELU

### D. Pruned ResNet baseline

Implemented as a lightweight TensorFlow ResNet-style approximation suitable for the same training script and metrics pipeline.

## 3. Training-time method implementation

Implemented in `histodyn/train.py` and `histodyn/data.py`:

- TFDS `colorectal_histology` loading
- resize / normalization
- stain-jitter style augmentation
- label smoothing
- EMA callback
- gate-history tracking
- evaluation with accuracy and ECE
- energy-per-image estimation

The package structure is designed so that all models run through the same training/evaluation entry point:

```bash
python -m histodyn.train --model biogater
python -m histodyn.train --model randomop
python -m histodyn.train --model convmixer
python -m histodyn.train --model pruned_resnet
```

## 4. Figure and result generation

### Existing paper-style outputs

- Table 1 performance
- gate distribution
- gate trends
- communication cost
- activation-map schematic

### Added comparative outputs

Use:

```bash
PYTHONPATH=. python scripts/add_comparative_results.py
```

This creates comparison figures in both PNG and JPG formats.

## 5. Practical interpretation of the comparative outputs

- **Accuracy plot**: shows overall predictive ranking.
- **ECE plot**: shows calibration quality, which is especially useful in medical-AI reporting.
- **Energy plot**: shows deployment efficiency.
- **Delta-to-BioGater table**: gives an immediate side-by-side explanation of the margin between the proposed method and each baseline.

## 6. Suggested thesis/paper reporting phrasing

You can report the package outputs as:

> In addition to the best absolute results of HistoDyn::BioGater, we provide direct comparative analysis against RandomOp Mixer, ConvMixer, and Pruned ResNet baselines, including absolute performance metrics and delta-to-method comparison plots. We also provide a full implementation mapping for the proposed method, covering patch embedding, dynamic top-k gating, biologically motivated operation banks, calibration-aware training, and exportable result figures in PNG/JPG format.

