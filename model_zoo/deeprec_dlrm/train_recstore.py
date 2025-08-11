#!/usr/bin/env python3
import os
import tensorflow as tf
import numpy as np

# Ensure TF logging is concise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Import RecStore TF client and Keras layer
from src.framework.tensorflow.python_client.client import (
    RecstoreClient,
    RecStoreEmbeddingBagLayer,
)


def build_dlrm_with_recstore(num_dense: int, embedding_dim: int):
    # Inputs
    dense_in = tf.keras.Input(shape=(num_dense,), dtype=tf.float32, name="dense")
    # ids: flattened categorical ids; lengths: per-sample counts for bagging
    ids_in = tf.keras.Input(shape=(None,), dtype=tf.int64, ragged=True, name="ids")

    # Convert ragged to flat ids and lengths
    flat_ids = ids_in.values
    row_lengths = ids_in.row_lengths()

    # RecStore EmbeddingBag (mean pool)
    ebc = RecStoreEmbeddingBagLayer(embedding_dim)
    pooled = ebc((flat_ids, row_lengths))  # [B, D]

    # Bottom MLP for dense
    x = dense_in
    for units in [512, 256, 64]:
        x = tf.keras.layers.Dense(units, activation="relu")(x)

    # Interaction (concat + top MLP)
    interacted = tf.keras.layers.Concatenate(axis=-1)([x, pooled])
    for units in [256, 128]:
        interacted = tf.keras.layers.Dense(units, activation="relu")(interacted)
    logits = tf.keras.layers.Dense(1, activation=None)(interacted)
    out = tf.keras.layers.Activation("sigmoid")(logits)

    model = tf.keras.Model(inputs={"dense": dense_in, "ids": ids_in}, outputs=out)
    return model


def main():
    # Initialize RecStore TF ops (loads build/lib/lib_recstore_tf_ops.so)
    _ = RecstoreClient()

    num_dense = 13
    embedding_dim = 128
    model = build_dlrm_with_recstore(num_dense=num_dense, embedding_dim=embedding_dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.AUC(name="auc")])

    # Synthetic data for demo
    batch_size = 64
    steps = 100
    for step in range(steps):
        dense = np.random.randn(batch_size, num_dense).astype(np.float32)
        # Build ragged ids: each sample has 1~3 ids
        lengths = np.random.randint(1, 4, size=(batch_size,), dtype=np.int32)
        ids = np.concatenate([
            np.random.randint(0, 100000, size=(l,), dtype=np.int64) for l in lengths
        ], axis=0)
        ragged_ids = tf.RaggedTensor.from_row_lengths(ids, lengths)
        labels = (np.random.rand(batch_size, 1) > 0.5).astype(np.float32)

        metrics = model.train_on_batch({"dense": dense, "ids": ragged_ids}, labels, return_dict=True)
        if step % 10 == 0:
            print(f"step={step} loss={metrics['loss']:.4f} auc={metrics['auc']:.4f}")

    # Save model (optional)
    model.save("/workspace/model_zoo/deeprec_dlrm/recstore_dlrm_saved_model", overwrite=True)


if __name__ == "__main__":
    main()