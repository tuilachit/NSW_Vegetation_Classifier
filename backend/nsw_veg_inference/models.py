"""Model builders for the two-stage NSW vegetation segmentation pipeline.

The package loads portable weights with `model.load_weights(...)`; it does not
deserialize full `.keras` models, avoiding Lambda-bytecode compatibility issues.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import INPUT_CHANNELS, PATCH_SIZE


def _gn(groups=8, name=None):
    try:
        return layers.GroupNormalization(groups=groups, axis=-1, epsilon=1e-5, name=name)
    except Exception:
        return layers.LayerNormalization(axis=-1, epsilon=1e-5, name=name)


def _conv_gn_relu(x, filters, kernel_size=3, strides=1, dropout=0.0, name_prefix="blk"):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False, name=f"{name_prefix}_conv")(x)
    x = _gn(name=f"{name_prefix}_gn")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name=f"{name_prefix}_drop")(x)
    return x


def _conv_block_gn(x, filters, dropout=0.0, name_prefix="block"):
    x = _conv_gn_relu(x, filters, 3, 1, dropout=dropout, name_prefix=f"{name_prefix}_1")
    x = _conv_gn_relu(x, filters, 3, 1, dropout=0.0, name_prefix=f"{name_prefix}_2")
    return x


def _decoder_block(x, skip, filters, dropout=0.0, name_prefix="dec"):
    x = layers.UpSampling2D(2, interpolation="bilinear", name=f"{name_prefix}_up")(x)
    x = layers.Concatenate(name=f"{name_prefix}_concat")([x, skip])
    return _conv_block_gn(x, filters, dropout=dropout, name_prefix=f"{name_prefix}_conv")


def _build_resnet_encoder():
    backbone = keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_shape=(PATCH_SIZE, PATCH_SIZE, 3),
    )
    feature_names = ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    encoder = keras.Model(backbone.input, [backbone.get_layer(name).output for name in feature_names], name="resnet50_rgb_encoder")
    return encoder


def _build_aux_encoder(input_channels=8):
    inputs = keras.Input(shape=(PATCH_SIZE, PATCH_SIZE, input_channels), name="aux_input")
    x = _conv_block_gn(inputs, 32, dropout=0.0, name_prefix="aux_stem")
    s1 = _conv_gn_relu(x, 64, 3, strides=2, dropout=0.0, name_prefix="aux_s1_down")
    s1 = _conv_block_gn(s1, 64, dropout=0.0, name_prefix="aux_s1")
    s2 = _conv_gn_relu(s1, 128, 3, strides=2, dropout=0.0, name_prefix="aux_s2_down")
    s2 = _conv_block_gn(s2, 128, dropout=0.0, name_prefix="aux_s2")
    s3 = _conv_gn_relu(s2, 256, 3, strides=2, dropout=0.1, name_prefix="aux_s3_down")
    s3 = _conv_block_gn(s3, 256, dropout=0.1, name_prefix="aux_s3")
    s4 = _conv_gn_relu(s3, 512, 3, strides=2, dropout=0.1, name_prefix="aux_s4_down")
    s4 = _conv_block_gn(s4, 512, dropout=0.1, name_prefix="aux_s4")
    bridge = _conv_gn_relu(s4, 1024, 3, strides=2, dropout=0.2, name_prefix="aux_bridge_down")
    bridge = _conv_block_gn(bridge, 1024, dropout=0.2, name_prefix="aux_bridge")
    return keras.Model(inputs, [s1, s2, s3, s4, bridge], name="aux_encoder")


def build_hybrid_unet(num_classes: int, model_name: str):
    """Build the exact hybrid RGB-ResNet50 + auxiliary U-Net architecture."""
    inputs = keras.Input(shape=(PATCH_SIZE, PATCH_SIZE, INPUT_CHANNELS), name="hybrid_input")
    rgb = layers.Lambda(lambda z: z[..., :3], name="rgb_slice")(inputs)
    aux = layers.Lambda(lambda z: z[..., 3:], name="aux_slice")(inputs)

    rgb_encoder = _build_resnet_encoder()
    aux_encoder = _build_aux_encoder(input_channels=8)
    r1, r2, r3, r4, r_bridge = rgb_encoder(rgb)
    a1, a2, a3, a4, a_bridge = aux_encoder(aux)

    s1 = layers.Concatenate(name="fuse_s1")([r1, a1])
    s1 = _conv_block_gn(s1, 64, dropout=0.0, name_prefix="fuse_s1_refine")
    s2 = layers.Concatenate(name="fuse_s2")([r2, a2])
    s2 = _conv_block_gn(s2, 128, dropout=0.0, name_prefix="fuse_s2_refine")
    s3 = layers.Concatenate(name="fuse_s3")([r3, a3])
    s3 = _conv_block_gn(s3, 256, dropout=0.1, name_prefix="fuse_s3_refine")
    s4 = layers.Concatenate(name="fuse_s4")([r4, a4])
    s4 = _conv_block_gn(s4, 512, dropout=0.1, name_prefix="fuse_s4_refine")
    bridge = layers.Concatenate(name="fuse_bridge")([r_bridge, a_bridge])
    bridge = _conv_block_gn(bridge, 1024, dropout=0.2, name_prefix="fuse_bridge_refine")

    d1 = _decoder_block(bridge, s4, 512, dropout=0.2, name_prefix="dec1")
    d2 = _decoder_block(d1, s3, 256, dropout=0.1, name_prefix="dec2")
    d3 = _decoder_block(d2, s2, 128, dropout=0.1, name_prefix="dec3")
    d4 = _decoder_block(d3, s1, 64, dropout=0.0, name_prefix="dec4")
    x = layers.UpSampling2D(2, interpolation="bilinear", name="final_up")(d4)
    x = _conv_block_gn(x, 32, dropout=0.0, name_prefix="final_refine")
    outputs = layers.Conv2D(num_classes, 1, activation="softmax", dtype="float32", name="segmentation_head")(x)
    return keras.Model(inputs, outputs, name=model_name)


def build_binary_model():
    return build_hybrid_unet(num_classes=2, model_name="hybrid_unet_resnet50_binary_veg_other")


def build_group_model():
    return build_hybrid_unet(num_classes=9, model_name="hybrid_unet_resnet50_9group_after_binary")


def load_weighted_models(binary_weights_path: str, group_weights_path: str):
    binary_model = build_binary_model()
    group_model = build_group_model()
    binary_model.load_weights(binary_weights_path)
    group_model.load_weights(group_weights_path)
    return binary_model, group_model
