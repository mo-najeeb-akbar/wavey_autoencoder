import tensorflow as tf
import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from flax.jax_utils import prefetch_to_device
from functools import partial
from pathlib import Path
import glob

def data_to_jax(tensor_dict, dtype=jnp.float32):
    """Convert TensorFlow tensors to JAX arrays"""
    return jax.tree.map(lambda x: jnp.array(x, dtype=dtype), tensor_dict)

def create_input_iter(tfds, dtype=jnp.float32):
    """Create JAX input iterator with device prefetching"""
    d_t_j = partial(data_to_jax, dtype=dtype)
    it = map(d_t_j, tfds)
    return it

class Dataloader:
    def __init__(self, tfrecord_pattern,
                 batch_size=32, shuffle_buffer=1000, enable_augmentation=False,
                 ):
        
        self.tfrecord_pattern = tfrecord_pattern
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.enable_augmentation = enable_augmentation
        
        self.tfrecord_files = sorted(glob.glob(tfrecord_pattern))
        
        print(f"Found {len(self.tfrecord_files)} TFRecord files")
    
    def _parse_example(self, example_proto):
        """Parse a single TFRecord example"""
        
        feature_description = {
            'features': tf.io.FixedLenFeature([], tf.string),
        }
        
        # Parse the example
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        
        # Deserialize the arrays
        img = tf.io.decode_image(parsed['features'])
        img = tf.image.rgb_to_grayscale(img)
        img = tf.cast(img, tf.float32) / 255.
        # mean = tf.reduce_mean(img, axis=[0, 1], keepdims=True)
        # variance = tf.reduce_mean(tf.square(img - mean), axis=[0, 1], keepdims=True)
        # img = (img - mean) / tf.sqrt(variance + 1e-8)

        return {
            'features': img,
        }
    
    
    def get_dataset(self, split='train', train_fraction=1.0, shuffle=True):
        """Get a tf.data.Dataset that yields individual crops with infinite repetition and proper epochs"""
        
        # Create dataset from TFRecord files
        dataset = tf.data.TFRecordDataset(self.tfrecord_files)
        
        # Parse examples
        dataset = dataset.map(self._parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.repeat()

        if shuffle:
            dataset = dataset.shuffle(self.shuffle_buffer)
        
        # Shuffle individual crops within the epoch
        if shuffle:
            # Use a larger buffer to mix crops from different images well
            crop_shuffle_buffer = 4800 
            dataset = dataset.shuffle(crop_shuffle_buffer)
            
        
        # Batch the individual crop samples
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        
        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_jax_iterator(self, split='train', dtype=jnp.float32, **kwargs):
        """Get JAX iterator using the standard pattern"""
        tfds = self.get_dataset(**kwargs)
        return create_input_iter(tfds, dtype=dtype)
    
