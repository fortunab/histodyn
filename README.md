# HistoDyn::BioGater TensorFlow code package

This package implements the main model families and result-generation utilities described in the uploaded paper:
- `HistoDyn::BioGater`
- `RandomOp Mixer`
- `ConvMixer (static)`
- `Pruned ResNet-18` (implemented as a light ResNet-like baseline in TensorFlow)

It also includes code for:
- label smoothing
- Mixup
- CutMix
- stain-jitter augmentation
- EMA
- Expected Calibration Error (ECE)
- Gated Activation Diversity Score (GADS)
- communication-cost and energy-per-image utilities
- figure export to both `.png` and `.jpg`

## Install

```bash
pip install tensorflow tensorflow-datasets numpy pandas matplotlib scikit-learn
```

## Train a model

```bash
python -m histodyn.train --model biogater --epochs 20 --results_dir results
python -m histodyn.train --model randomop --epochs 20 --results_dir results
python -m histodyn.train --model convmixer --epochs 20 --results_dir results
python -m histodyn.train --model pruned_resnet --epochs 20 --results_dir results
```

The training script is set up for TFDS `colorectal_histology`.

## Generate the paper-style figures directly

```bash
PYTHONPATH=. python scripts/generate_paper_figures.py
```

This exports:
- `table1_performance.png/.jpg`
- `figure4_gate_distribution.png/.jpg`
- `figure5_gate_trends.png/.jpg`
- `table2_gads.csv`
- `figure6_comm_cost.png/.jpg`
- `figure7_activation_maps.png/.jpg`

## Notes

- The figure-generation script includes the exact paper-reported aggregate values for Table 1 / Table 2 and paper-aligned visualization code.
- The activation map panel is a schematic placeholder unless you run the full model and replace it with real feature overlays.
- The paper mentions TensorFlow Histopathology data and biologically motivated ops such as stain normalization, contrast amplification, sign/binarize gating, top-k selection, GADS, communication cost, and energy-per-image; those are reflected here.


## Added comparative package outputs

The package also includes:
- absolute comparative results across all methods
- delta-to-BioGater comparative tables
- a separate method implementation writeup

Generate them with:

```bash
PYTHONPATH=. python scripts/add_comparative_results.py
```

See also:
- `METHOD_IMPLEMENTATION.md`
- `results/comparative_results.csv`
- `results/comparative_deltas_vs_biogater.csv`
- `results/comparative_accuracy.png/.jpg`
- `results/comparative_ece.png/.jpg`
- `results/comparative_energy.png/.jpg`
