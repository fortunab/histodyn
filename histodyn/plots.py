from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, output_base: str, dpi: int = 200):
    fig.savefig(output_base + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_base + ".jpg", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_table1(df: pd.DataFrame, output_base: str):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    _save(fig, output_base)


def plot_gate_distribution(labels: List[str], values: List[float], output_base: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.pie(values, labels=labels, autopct="%1.2f%%")
    ax.set_title("Final gate-weight distribution")
    _save(fig, output_base)


def plot_gate_trends(gate_history: np.ndarray, op_names: List[str], output_base: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = np.arange(gate_history.shape[0])
    for i, name in enumerate(op_names):
        ax.plot(epochs, gate_history[:, i], label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gate value")
    ax.set_title("Temporal trends of operation gating")
    ax.legend(ncol=2, fontsize=8)
    _save(fig, output_base)


def plot_comm_cost(labels: List[str], values: List[float], output_base: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(labels))
    ax.barh(y, values)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Communication cost (%)")
    ax.set_title("Communication cost per step")
    _save(fig, output_base)


def plot_activation_maps(image: np.ndarray, act1: np.ndarray, act2: np.ndarray, output_base: str):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input patch")
    axes[1].imshow(image)
    axes[1].imshow(act1, alpha=0.45)
    axes[1].set_title("scale_large activation")
    axes[2].imshow(image)
    axes[2].imshow(act2, alpha=0.45)
    axes[2].set_title("sign/sobel activation")
    for ax in axes:
        ax.axis("off")
    _save(fig, output_base)
