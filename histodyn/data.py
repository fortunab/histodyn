import os
from typing import Optional, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds


AUTOTUNE = tf.data.AUTOTUNE


def stain_jitter(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.random_brightness(image, 0.08)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.image.random_saturation(image, 0.9, 1.1)
    image = tf.image.random_hue(image, 0.02)
    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_example(example, img_size: int, num_classes: int, training: bool = False):
    image = tf.cast(example["image"], tf.float32) / 255.0
    image = tf.image.resize(image, [img_size, img_size])
    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = stain_jitter(image)
    label = tf.one_hot(tf.cast(example["label"], tf.int32), depth=num_classes)
    return image, label


def mixup_batch(images, labels, alpha=0.2):
    batch_size = tf.shape(images)[0]
    lam = tf.squeeze(tf.random.gamma([1], alpha) / (tf.random.gamma([1], alpha) + tf.random.gamma([1], alpha)))
    idx = tf.random.shuffle(tf.range(batch_size))
    images2 = tf.gather(images, idx)
    labels2 = tf.gather(labels, idx)
    return lam * images + (1.0 - lam) * images2, lam * labels + (1.0 - lam) * labels2


def cutmix_batch(images, labels, alpha=1.0):
    batch_size = tf.shape(images)[0]
    idx = tf.random.shuffle(tf.range(batch_size))
    images2 = tf.gather(images, idx)
    labels2 = tf.gather(labels, idx)
    h = tf.shape(images)[1]
    w = tf.shape(images)[2]
    lam = tf.squeeze(tf.random.uniform([1], 0.3, 0.7))
    cut_rat = tf.sqrt(1.0 - lam)
    cut_w = tf.cast(w * cut_rat, tf.int32)
    cut_h = tf.cast(h * cut_rat, tf.int32)
    cx = tf.random.uniform([], 0, w, dtype=tf.int32)
    cy = tf.random.uniform([], 0, h, dtype=tf.int32)
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, w)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, w)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, h)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, h)
    mask = tf.pad(tf.ones((y2 - y1, x2 - x1, 3), dtype=images.dtype), [[y1, h - y2], [x1, w - x2], [0, 0]])
    mixed = images * (1.0 - mask) + images2 * mask
    lam_adj = 1.0 - (tf.cast((x2 - x1) * (y2 - y1), tf.float32) / tf.cast(w * h, tf.float32))
    labels = lam_adj * labels + (1.0 - lam_adj) * labels2
    return mixed, labels


def apply_batch_augment(images, labels, use_mixup=True, use_cutmix=True):
    r = tf.random.uniform([])
    if use_mixup and r < 0.5:
        return mixup_batch(images, labels)
    if use_cutmix:
        return cutmix_batch(images, labels)
    return images, labels


def make_tfds_histology_loaders(
    dataset_name: str = "colorectal_histology",
    img_size: int = 150,
    batch_size: int = 32,
    num_classes: int = 8,
    use_mixup: bool = True,
    use_cutmix: bool = True,
):
    train = tfds.load(dataset_name, split="train")
    if dataset_name == "colorectal_histology":
        train = train.shuffle(5000, reshuffle_each_iteration=True)
        train_size = 4000
        val_size = 1000
        train_ds = train.take(train_size)
        val_ds = train.skip(train_size).take(val_size)
        test_ds = val_ds
    else:
        raise ValueError(f"Unsupported TFDS dataset: {dataset_name}")

    train_ds = train_ds.map(lambda x: preprocess_example(x, img_size, num_classes, True), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size).map(
        lambda x, y: apply_batch_augment(x, y, use_mixup, use_cutmix),
        num_parallel_calls=AUTOTUNE,
    ).prefetch(AUTOTUNE)

    val_ds = val_ds.map(lambda x: preprocess_example(x, img_size, num_classes, False), num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    test_ds = test_ds.map(lambda x: preprocess_example(x, img_size, num_classes, False), num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds
