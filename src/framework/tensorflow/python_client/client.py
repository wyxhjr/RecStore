import tensorflow as tf
import os
from typing import Optional, Sequence

class RecstoreClient:
    _ops_module = None

    def __init__(self, library_path: Optional[str] = None):
        if RecstoreClient._ops_module:
            return
            
        if library_path is None:
            script_dir = os.path.dirname(__file__)
            default_lib_path = os.path.abspath(
                os.path.join(script_dir, '../../../../build/lib/lib_recstore_tf_ops.so')
            )
            if not os.path.exists(default_lib_path):
                 raise ImportError(
                    f"Could not find Recstore TF library at default path: {default_lib_path}\n"
                    "Please provide the correct path via the 'library_path' argument "
                    "or ensure your project is built correctly."
                )
            library_path = default_lib_path
        
        RecstoreClient._ops_module = tf.load_op_library(library_path)
        print(f"RecstoreClient (TensorFlow) initialized. Loaded library from: {library_path}")

    def emb_read(self, keys: tf.Tensor, embedding_dim: int) -> tf.Tensor:
        if not isinstance(keys, tf.Tensor):
            keys = tf.convert_to_tensor(keys, dtype=tf.uint64)
        if keys.dtype != tf.uint64:
            raise TypeError(f"keys tensor must be of dtype tf.uint64, but got {keys.dtype}")
        return RecstoreClient._ops_module.recstore_emb_read(keys, embedding_dim=embedding_dim)

    def emb_update(self, keys: tf.Tensor, grads: tf.Tensor, embedding_dim: int) -> tf.Operation:
        if not isinstance(keys, tf.Tensor):
            keys = tf.convert_to_tensor(keys, dtype=tf.uint64)
        if not isinstance(grads, tf.Tensor):
            grads = tf.convert_to_tensor(grads, dtype=tf.float32)
        if keys.dtype != tf.uint64:
            raise TypeError(f"keys tensor must be of dtype tf.uint64, but got {keys.dtype}")
        if grads.dtype != tf.float32:
            raise TypeError(f"grads tensor must be of dtype tf.float32, but got {grads.dtype}")
        return RecstoreClient._ops_module.recstore_emb_update(keys, grads, embedding_dim=embedding_dim)

    def emb_write(self, keys: tf.Tensor, values: tf.Tensor, embedding_dim: int) -> tf.Operation:
        if not isinstance(keys, tf.Tensor):
            keys = tf.convert_to_tensor(keys, dtype=tf.uint64)
        if not isinstance(values, tf.Tensor):
            values = tf.convert_to_tensor(values, dtype=tf.float32)
        if keys.dtype != tf.uint64:
            raise TypeError(f"keys tensor must be of dtype tf.uint64, but got {keys.dtype}")
        if values.dtype != tf.float32:
            raise TypeError(f"values tensor must be of dtype tf.float32, but got {values.dtype}")
        return RecstoreClient._ops_module.recstore_emb_write(keys, values, embedding_dim=embedding_dim)


@tf.custom_gradient
def recstore_embedding_lookup(keys: tf.Tensor, embedding_dim: int):
    client = RecstoreClient()
    values = client.emb_read(keys, embedding_dim)
    def grad(dy):
        # dy shape [L, D]; pass-through as grads
        client.emb_update(keys, dy, embedding_dim)
        # No gradient to keys nor embedding_dim
        return tf.zeros_like(keys), None
    return values, grad


class RecStoreEmbeddingBagLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = int(embedding_dim)

    def call(self, inputs: Sequence[tf.Tensor]) -> tf.Tensor:
        # inputs: (values:int64 ids [N], row_splits/lengths:int32 [B]) or just flat ids
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            ids, lengths = inputs
            ids = tf.cast(ids, tf.uint64)
            lengths = tf.cast(lengths, tf.int32)
            # Gather per-sample segments and average
            values = recstore_embedding_lookup(ids, self.embedding_dim)
            # Build segment ids from lengths
            seg_ids = tf.repeat(tf.range(tf.shape(lengths)[0]), lengths)
            pooled = tf.math.segment_mean(values, seg_ids)
            return pooled
        else:
            ids = tf.cast(inputs, tf.uint64)
            values = recstore_embedding_lookup(ids, self.embedding_dim)
            return values

