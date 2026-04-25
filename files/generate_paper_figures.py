import os
from pathlib import Path

import numpy as np
import pandas as pd

from histodyn.metrics import gated_activation_diversity_score
from histodyn.plots import (
    plot_activation_maps,
    plot_comm_cost,
    plot_gate_distribution,
    plot_gate_trends,
    plot_table1,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Paper-reported Table 1 and Table 2 values
perf = pd.DataFrame(
    [
        ["RandomOp Mixer", 93.1, 0.058, 0.82],
        ["ConvMixer (static)", 94.6, 0.046, 0.67],
        ["Pruned ResNet-18", 92.8, 0.063, 0.57],
        ["HistoDyn::BioGater", 96.7, 0.021, 0.51],
    ],
    columns=["Model", "Accuracy (%)", "ECE", "Energy/img (J)"],
)
plot_table1(perf, str(RESULTS_DIR / "table1_performance"))
perf.to_csv(RESULTS_DIR / "table1_performance.csv", index=False)

# Figure 4
labels = ["mean_subtract", "abs", "reverse", "sign", "scale_large", "others"]
values = [34.38, 27.35, 23.56, 7.62, 4.49, 2.60]
plot_gate_distribution(labels, values, str(RESULTS_DIR / "figure4_gate_distribution"))

# Figure 5: synthetic but paper-aligned trend generator for visualization
ops = [
    "identity", "softmax_gate", "zeros", "noise_injection", "spatial_dropout", "scale_small",
    "scale_large", "negate", "abs", "square", "sqrt", "sign", "binarize", "tanh_gate",
    "sigmoid_gate", "mean_subtract", "global_context", "reverse", "roll", "slice_first_half",
    "slice_second_half", "linear_skip",
]
epochs = 51
x = np.arange(epochs)
trend = []
for op in ops:
    if op == "mean_subtract":
        y = 0.15 + 0.55 * (1 - np.exp(-x / 10))
    elif op == "abs":
        y = 0.20 + 0.45 * (1 - np.exp(-x / 12))
    elif op == "reverse":
        y = 0.18 + 0.40 * (1 - np.exp(-x / 13))
    elif op in ("sign", "scale_large", "global_context", "binarize"):
        y = 0.12 + 0.20 * (1 - np.exp(-x / 15))
    else:
        y = 0.22 * np.exp(-x / 14)
    trend.append(y)
gate_history = np.stack(trend, axis=1)
gate_history = gate_history / gate_history.max(axis=1, keepdims=True)
plot_gate_trends(gate_history, ops, str(RESULTS_DIR / "figure5_gate_trends"))
np.save(RESULTS_DIR / "figure5_gate_history.npy", gate_history)

# Table 2
pd.DataFrame(
    [
        ["RandomOp Mixer", 2.91],
        ["ConvMixer (static)", 1.87],
        ["HistoDyn::BioGater", 3.22],
    ],
    columns=["Model", "GADS"],
).to_csv(RESULTS_DIR / "table2_gads.csv", index=False)

# Figure 6
plot_comm_cost(["RandomOp", "ConvMixer", "BioGater"], [100, 74, 58], str(RESULTS_DIR / "figure6_comm_cost"))

# Figure 7 placeholder activation overlays
h, w = 150, 150
img = np.ones((h, w, 3), dtype=np.float32)
img[..., 0] = np.linspace(0.85, 1.0, w)[None, :]
img[..., 1] = np.linspace(0.75, 0.95, w)[None, :]
img[..., 2] = np.linspace(0.80, 0.92, w)[None, :]
y, xg = np.mgrid[0:h, 0:w]
act1 = np.exp(-(((xg - 60) ** 2) + ((y - 70) ** 2)) / (2 * 18 ** 2))
act2 = np.exp(-(((xg - 95) ** 2) + ((y - 75) ** 2)) / (2 * 14 ** 2))
plot_activation_maps(img, act1, act2, str(RESULTS_DIR / "figure7_activation_maps"))

print(f"Saved paper-style figures to {RESULTS_DIR.resolve()}")
