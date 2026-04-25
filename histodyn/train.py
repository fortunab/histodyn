import argparse
import csv
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from histodyn.data import make_tfds_histology_loaders
from histodyn.metrics import collect_predictions, energy_per_image, expected_calibration_error, save_metrics_json
from histodyn.models import (
    BioGaterBlock,
    ModelConfig,
    build_convmixer_static,
    build_histodyn_biogater,
    build_pruned_resnet18,
    build_randomop_mixer,
)
from histodyn.plots import plot_table1


class EMACallback(tf.keras.callbacks.Callback):
    def __init__(self, momentum=0.999):
        super().__init__()
        self.momentum = momentum
        self.ema_weights = None

    def on_train_begin(self, logs=None):
        self.ema_weights = [w.numpy() for w in self.model.weights]

    def on_train_batch_end(self, batch, logs=None):
        for i, w in enumerate(self.model.weights):
            self.ema_weights[i] = self.momentum * self.ema_weights[i] + (1.0 - self.momentum) * w.numpy()

    def apply_ema(self):
        self.model.set_weights(self.ema_weights)


class GateHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.history = []
        self.op_names = None

    def on_epoch_end(self, epoch, logs=None):
        vals = []
        names = []
        for layer in self.model.layers:
            if isinstance(layer, BioGaterBlock):
                gates = tf.nn.softmax(layer.head_gates).numpy()
                vals.extend(gates.tolist())
                names.extend(layer.op_names)
                break
        self.op_names = names
        self.history.append(vals)


def get_builder(name, cfg):
    if name == "biogater":
        return build_histodyn_biogater(cfg)
    if name == "randomop":
        return build_randomop_mixer(cfg)
    if name == "convmixer":
        return build_convmixer_static(cfg)
    if name == "pruned_resnet":
        return build_pruned_resnet18(cfg)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["biogater", "randomop", "convmixer", "pruned_resnet"], default="biogater")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=150)
    ap.add_argument("--patch_size", type=int, default=10)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--num_blocks", type=int, default=2)
    ap.add_argument("--num_classes", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--results_dir", type=str, default="results")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(42)
    os.makedirs(args.results_dir, exist_ok=True)

    cfg = ModelConfig(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_blocks=args.num_blocks,
        num_classes=args.num_classes,
        top_k=args.top_k,
    )
    train_ds, val_ds, test_ds = make_tfds_histology_loaders(
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
    )
    model = get_builder(args.model, cfg)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")],
    )
    gate_cb = GateHistoryCallback()
    ema_cb = EMACallback(momentum=0.999)
    start = time.perf_counter()
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[gate_cb, ema_cb], verbose=2)
    step_time = (time.perf_counter() - start) / max(1, args.epochs)
    ema_cb.apply_ema()

    preds = collect_predictions(model, test_ds)
    ece = expected_calibration_error(preds["y_true"], preds["y_prob"])
    eval_res = model.evaluate(test_ds, verbose=0)
    metrics = {
        "model": args.model,
        "test_loss": float(eval_res[0]),
        "test_acc": float(eval_res[1]),
        "ece": ece,
        "energy_per_image_j": energy_per_image(power_watts=220.0, step_time_sec=step_time, batch_size=args.batch_size, num_devices=1),
        "config": asdict(cfg),
    }
    save_metrics_json(metrics, os.path.join(args.results_dir, f"metrics_{args.model}.json"))
    if gate_cb.history:
        np.save(os.path.join(args.results_dir, f"gate_history_{args.model}.npy"), np.asarray(gate_cb.history))
        Path(os.path.join(args.results_dir, f"gate_names_{args.model}.txt")).write_text("\n".join(gate_cb.op_names), encoding="utf-8")
    print(metrics)


if __name__ == "__main__":
    main()
