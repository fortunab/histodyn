import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


class STE_Sign(layers.Layer):
    def call(self, x):
        @tf.custom_gradient
        def _ste_sign(u):
            y = tf.sign(u)
            def grad(dy):
                return dy * tf.cast(tf.abs(u) <= 1.0, u.dtype)
            return y, grad
        return _ste_sign(x)


class STE_Binarize(layers.Layer):
    def __init__(self, threshold=0.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def call(self, x):
        t = tf.cast(self.threshold, x.dtype)
        @tf.custom_gradient
        def _ste_bin(u):
            y = tf.cast(u > t, u.dtype)
            def grad(dy):
                return dy * tf.cast(tf.abs(u - t) <= 1.0, u.dtype)
            return y, grad
        return _ste_bin(x)


class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding="valid")
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        x = self.proj(x)
        b = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        c = tf.shape(x)[3]
        x = tf.reshape(x, [b, h * w, c])
        return self.norm(x)


def _tokens_to_grid(x):
    b = tf.shape(x)[0]
    n = tf.shape(x)[1]
    d = tf.shape(x)[2]
    s = tf.cast(tf.math.sqrt(tf.cast(n, tf.float32)), tf.int32)
    return tf.reshape(x, [b, s, s, d]), s


def _grid_to_tokens(x):
    b = tf.shape(x)[0]
    h = tf.shape(x)[1]
    w = tf.shape(x)[2]
    d = tf.shape(x)[3]
    return tf.reshape(x, [b, h * w, d])


class StdNormOp(layers.Layer):
    def call(self, x):
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        std = tf.math.reduce_std(x, axis=1, keepdims=True) + 1e-6
        return (x - mean) / std


class ReverseOp(layers.Layer):
    def call(self, x):
        return tf.reverse(x, axis=[1])


class MeanSubtractOp(layers.Layer):
    def call(self, x):
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        return x - mean


class GlobalContextOp(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.gate = layers.Dense(embed_dim, activation="sigmoid")

    def call(self, x):
        ctx = tf.reduce_mean(x, axis=1, keepdims=True)
        return x * self.gate(ctx)


class SobelMagOp(layers.Layer):
    def call(self, x):
        grid, _ = _tokens_to_grid(x)
        sob = tf.image.sobel_edges(grid)
        mag = tf.sqrt(tf.reduce_sum(tf.square(sob), axis=-1) + 1e-6)
        mag = tf.reduce_mean(mag, axis=-1)
        return _grid_to_tokens(mag)


class RollOp(layers.Layer):
    def __init__(self, shift=1, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.shift = shift
        self.axis = axis

    def call(self, x):
        return tf.roll(x, shift=self.shift, axis=self.axis)


class SliceFirstHalfOp(layers.Layer):
    def call(self, x):
        d = tf.shape(x)[-1]
        first = x[..., : d // 2]
        second = tf.zeros_like(x[..., d // 2 :])
        return tf.concat([first, second], axis=-1)


class SliceSecondHalfOp(layers.Layer):
    def call(self, x):
        d = tf.shape(x)[-1]
        first = tf.zeros_like(x[..., : d // 2])
        second = x[..., d // 2 :]
        return tf.concat([first, second], axis=-1)


class LinearSkipOp(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.proj = layers.Dense(embed_dim)

    def call(self, x):
        return self.proj(x)


class RandomOpMixerBlock(layers.Layer):
    def __init__(self, embed_dim: int, num_ops: int = 6, **kwargs):
        super().__init__(**kwargs)
        ops = cheap_ops_dict(embed_dim)
        self.op_names = list(ops.keys())[:num_ops]
        self.ops = [ops[k] for k in self.op_names]
        self.proj = layers.Dense(embed_dim)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=None):
        idx = tf.random.uniform([], minval=0, maxval=len(self.ops), dtype=tf.int32)
        outs = [op(self.norm(x), training=training) if hasattr(op, 'call') else op(self.norm(x)) for op in self.ops]
        mixed = tf.gather(tf.stack(outs, axis=1), idx, axis=1)
        return x + self.proj(mixed)


class ConvMixerBlock(layers.Layer):
    def __init__(self, embed_dim: int, kernel_size: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.dw = layers.DepthwiseConv2D(kernel_size, padding="same")
        self.pw = layers.Conv2D(embed_dim, kernel_size=1, padding="same")
        self.act = layers.Activation("gelu")
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

    def call(self, x, training=None):
        grid, _ = _tokens_to_grid(x)
        y = self.dw(grid)
        y = self.act(y)
        y = self.bn1(y, training=training)
        y = y + grid
        y = self.pw(y)
        y = self.act(y)
        y = self.bn2(y, training=training)
        return _grid_to_tokens(y)


class BioGaterBlock(layers.Layer):
    def __init__(self, embed_dim: int, top_k: int = 5, l1_lambda: float = 1e-4, drop_path: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.l1_lambda = l1_lambda
        self.drop_path = drop_path
        self.pre_norm = layers.LayerNormalization(epsilon=1e-6)
        self.dw_premix = layers.DepthwiseConv2D(3, padding="same")
        self.post_proj = layers.Dense(embed_dim)
        self.dropout = layers.Dropout(drop_path)
        ops = cheap_ops_dict(embed_dim)
        self.op_names = list(ops.keys())
        self.cheap_heads = [ops[k] for k in self.op_names]
        self.head_gates = self.add_weight(
            name="head_gates",
            shape=(len(self.op_names),),
            initializer=tf.keras.initializers.Constant(0.05),
            trainable=True,
            regularizer=regularizers.L1(l1_lambda),
        )
        self.last_soft_gates = None

    def call(self, x, training=None, return_details=False):
        z = self.pre_norm(x)
        grid, _ = _tokens_to_grid(z)
        z = _grid_to_tokens(self.dw_premix(grid))
        logits = self.head_gates
        soft = tf.nn.softmax(logits)
        if training:
            weights = soft
        else:
            topk = tf.math.top_k(soft, k=min(self.top_k, len(self.op_names)))
            mask = tf.reduce_sum(tf.one_hot(topk.indices, depth=len(self.op_names)), axis=0)
            weights = soft * mask
            weights = weights / (tf.reduce_sum(weights) + 1e-8)
        self.last_soft_gates = soft
        outs = []
        for op in self.cheap_heads:
            try:
                out = op(z, training=training)
            except TypeError:
                out = op(z)
            outs.append(out)
        stacked = tf.stack(outs, axis=1)
        mixed = tf.einsum("k,bknd->bnd", weights, stacked)
        mixed = self.post_proj(mixed)
        mixed = self.dropout(mixed, training=training)
        y = x + mixed
        if return_details:
            return y, {"gates": weights, "op_outputs": stacked}
        return y


def cheap_ops_dict(embed_dim: int) -> Dict[str, layers.Layer]:
    return {
        "identity": layers.Lambda(lambda x: x, name="identity"),
        "softmax_gate": layers.Lambda(lambda x: x * tf.nn.softmax(x, axis=1), name="softmax_gate"),
        "zeros": layers.Lambda(lambda x: tf.zeros_like(x), name="zeros"),
        "noise_injection": layers.Lambda(lambda x: x + tf.random.normal(tf.shape(x), stddev=0.01), name="noise_injection"),
        "spatial_dropout": layers.SpatialDropout1D(0.1, name="spatial_dropout"),
        "scale_small": layers.Lambda(lambda x: 0.5 * x, name="scale_small"),
        "scale_large": layers.Lambda(lambda x: 1.5 * x, name="scale_large"),
        "negate": layers.Lambda(lambda x: -x, name="negate"),
        "abs": layers.Lambda(tf.abs, name="abs"),
        "square": layers.Lambda(tf.square, name="square"),
        "sqrt": layers.Lambda(lambda x: tf.sqrt(tf.abs(x) + 1e-6), name="sqrt"),
        "sign": STE_Sign(name="sign"),
        "binarize": STE_Binarize(name="binarize"),
        "tanh_gate": layers.Lambda(tf.nn.tanh, name="tanh_gate"),
        "sigmoid_gate": layers.Lambda(tf.nn.sigmoid, name="sigmoid_gate"),
        "mean_subtract": MeanSubtractOp(name="mean_subtract"),
        "std_norm": StdNormOp(name="std_norm"),
        "global_context": GlobalContextOp(embed_dim, name="global_context"),
        "reverse": ReverseOp(name="reverse"),
        "roll": RollOp(name="roll"),
        "slice_first_half": SliceFirstHalfOp(name="slice_first_half"),
        "slice_second_half": SliceSecondHalfOp(name="slice_second_half"),
        "linear_skip": LinearSkipOp(embed_dim, name="linear_skip"),
        "sobel_mag": SobelMagOp(name="sobel_mag"),
    }


@dataclass
class ModelConfig:
    img_size: int = 150
    patch_size: int = 10
    embed_dim: int = 128
    num_blocks: int = 2
    num_classes: int = 8
    top_k: int = 5
    drop_path: float = 0.05


def build_histodyn_biogater(cfg: ModelConfig) -> tf.keras.Model:
    inp = layers.Input(shape=(cfg.img_size, cfg.img_size, 3))
    x = PatchEmbedding(cfg.patch_size, cfg.embed_dim, name="patch_embed")(inp)
    for i in range(cfg.num_blocks):
        x = BioGaterBlock(cfg.embed_dim, cfg.top_k, drop_path=cfg.drop_path, name=f"biogater_{i}")(x)
    x_avg = layers.GlobalAveragePooling1D()(x)
    x_max = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([x_avg, x_max])
    x = layers.Dense(cfg.embed_dim, activation="gelu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(cfg.num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="HistoDyn_BioGater")


def build_randomop_mixer(cfg: ModelConfig) -> tf.keras.Model:
    inp = layers.Input(shape=(cfg.img_size, cfg.img_size, 3))
    x = PatchEmbedding(cfg.patch_size, cfg.embed_dim)(inp)
    for i in range(cfg.num_blocks):
        x = RandomOpMixerBlock(cfg.embed_dim, name=f"randomop_{i}")(x)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(cfg.num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="RandomOpMixer")


def build_convmixer_static(cfg: ModelConfig) -> tf.keras.Model:
    inp = layers.Input(shape=(cfg.img_size, cfg.img_size, 3))
    x = PatchEmbedding(cfg.patch_size, cfg.embed_dim)(inp)
    for i in range(cfg.num_blocks):
        x = ConvMixerBlock(cfg.embed_dim, name=f"convmixer_{i}")(x)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(cfg.num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="ConvMixerStatic")


def build_pruned_resnet18(cfg: ModelConfig, pruning_ratio: float = 0.3) -> tf.keras.Model:
    inp = layers.Input(shape=(cfg.img_size, cfg.img_size, 3))
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_tensor=inp,
        pooling="avg",
    )
    x = base.output
    x = layers.Dense(max(64, int(512 * (1.0 - pruning_ratio))), activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(cfg.num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="PrunedResNetLike")
