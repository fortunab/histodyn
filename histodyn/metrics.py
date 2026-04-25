import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float32)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        left, right = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)
        if np.any(mask):
            acc = accuracies[mask].mean()
            conf = confidences[mask].mean()
            ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def gated_activation_diversity_score(gate_history: np.ndarray, eps: float = 1e-12) -> float:
    gate_history = np.asarray(gate_history, dtype=np.float64)
    gate_history = gate_history / (gate_history.sum(axis=1, keepdims=True) + eps)
    p_hat = gate_history.mean(axis=0)
    return float(-np.sum(p_hat * np.log(p_hat + eps)))


def energy_per_image(power_watts: float, step_time_sec: float, batch_size: int, num_devices: int) -> float:
    return float((power_watts * step_time_sec) / (batch_size * num_devices))


def ring_allreduce_comm(alpha: float, beta: float, m_active: float, num_gpus: int) -> float:
    return float(alpha * math.log(num_gpus) + beta * m_active * (num_gpus - 1) / num_gpus)


def collect_predictions(model, dataset) -> Dict[str, np.ndarray]:
    probs = []
    ys = []
    for x, y in dataset:
        p = model.predict(x, verbose=0)
        probs.append(p)
        ys.append(tf.argmax(y, axis=1).numpy())
    return {"y_true": np.concatenate(ys), "y_prob": np.concatenate(probs)}


def save_metrics_json(metrics: Dict, output_path: str):
    Path(output_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
