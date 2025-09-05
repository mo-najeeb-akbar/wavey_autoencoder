import tensorflow as tf
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
import glob

def data_to_jax(tensor_dict, dtype=jnp.float32):
    """Convert TensorFlow tensors to JAX arrays"""
    return jax.tree.map(lambda x: jnp.array(x, dtype=dtype), tensor_dict)

def create_input_iter(tfds, dtype=jnp.float32):
    """Create JAX input iterator"""
    d_t_j = partial(data_to_jax, dtype=dtype)
    return map(d_t_j, tfds)

class Dataloader:
    def __init__(self, tfrecord_pattern, batch_size=32, enable_augmentation=False, 
                 precomputed_global_mean=None, precomputed_global_std=None):
        
        self.batch_size = batch_size
        self.enable_augmentation = enable_augmentation
        
        # Find TFRecord files
        self.tfrecord_files = sorted(glob.glob(tfrecord_pattern))
        if not self.tfrecord_files:
            raise ValueError(f"No TFRecord files found: {tfrecord_pattern}")
        
        print(f"Found {len(self.tfrecord_files)} TFRecord files")
        
        # Count samples and compute/set global statistics
        self.total_samples = self._count_samples()
        print(f"Total samples: {self.total_samples}")
        print(f"Batches per epoch: {self.total_samples // batch_size}")
        
        if precomputed_global_mean is not None and precomputed_global_std is not None:
            self.global_mean = precomputed_global_mean
            self.global_std = precomputed_global_std
            print(f"Using precomputed stats: mean={self.global_mean:.1f}, std={self.global_std:.1f}")
        else:
            print("Computing global statistics...")
            self.global_mean, self.global_std = self._compute_global_stats()
            print(f"Computed stats: mean={self.global_mean:.1f}, std={self.global_std:.1f}")
    
    def _count_samples(self):
        """Count total samples across all shards"""
        total = 0
        for tfrecord_file in self.tfrecord_files:
            total += sum(1 for _ in tf.data.TFRecordDataset([tfrecord_file]))
        return total
    
    def _compute_global_stats(self):
        """Compute global mean and std across entire dataset"""
        dataset = tf.data.TFRecordDataset(self.tfrecord_files)
        
        def parse_intensity_only(example_proto):
            feature_description = {
                'intensity': tf.io.FixedLenFeature([], tf.string),
                'height': tf.io.FixedLenFeature([], tf.int64),
                'width': tf.io.FixedLenFeature([], tf.int64),
            }
            parsed = tf.io.parse_single_example(example_proto, feature_description)
            intensity = tf.io.parse_tensor(parsed['intensity'], out_type=tf.float32)
            intensity = tf.reshape(intensity, [parsed['height'], parsed['width']])
            intensity = tf.where(tf.math.is_nan(intensity), 0.0, intensity)
            return intensity
        
        dataset = dataset.map(parse_intensity_only).batch(100)
        
        # Welford's algorithm for stable computation
        count = 0
        mean = 0.0
        m2 = 0.0
        
        for batch in dataset:
            batch_flat = tf.reshape(batch, [-1]).numpy()
            for value in batch_flat:
                count += 1
                delta = value - mean
                mean += delta / count
                delta2 = value - mean
                m2 += delta * delta2
        
        variance = m2 / (count - 1) if count > 1 else 0.0
        return float(mean), float(np.sqrt(variance))
    
    def _parse_example(self, example_proto):
        """Parse TFRecord example"""
        feature_description = {
            'intensity': tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'crop_id': tf.io.FixedLenFeature([], tf.int64),
            'crop_row': tf.io.FixedLenFeature([], tf.int64),
            'crop_col': tf.io.FixedLenFeature([], tf.int64),
        }
        
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        
        # Parse intensity
        intensity = tf.io.parse_tensor(parsed['intensity'], out_type=tf.float32)
        intensity = tf.reshape(intensity, [parsed['height'], parsed['width']])
        intensity = tf.where(tf.math.is_nan(intensity), 0.0, intensity)
        
        # Apply augmentations
        if self.enable_augmentation:
            intensity = self._augment(intensity)
        
        # Global standardization
        intensity = (intensity - self.global_mean) / self.global_std
        
        return {
            'features': tf.expand_dims(intensity, axis=-1),  # [height, width]
            'crop_id': parsed['crop_id'],
            'crop_position': tf.stack([parsed['crop_row'], parsed['crop_col']]),
        }
    
    def _augment(self, intensity):
        """Apply augmentations"""
        # Add channel dimension for TF image ops
        intensity = tf.expand_dims(intensity, axis=-1)
        
        # Random flips and rotations
        intensity = tf.image.random_flip_left_right(intensity)
        intensity = tf.image.random_flip_up_down(intensity)
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        intensity = tf.image.rot90(intensity, k=k)
        
        # Brightness and contrast
        intensity = tf.image.random_brightness(intensity, max_delta=0.1)
        intensity = tf.image.random_contrast(intensity, lower=0.9, upper=1.1)
        
        # Remove channel dimension
        return tf.squeeze(intensity, axis=-1)
    
    def get_dataset(self, shuffle=True):
        """Get tf.data.Dataset"""
        dataset = tf.data.TFRecordDataset(self.tfrecord_files)
        dataset = dataset.map(self._parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.repeat()
        
        if shuffle:
            dataset = dataset.shuffle(4 * self.batch_size)
        
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def get_jax_iterator(self, **kwargs):
        """Get JAX iterator"""
        tfds = self.get_dataset(**kwargs)
        return create_input_iter(tfds)
    
    def get_batches_per_epoch(self):
        """Number of batches per epoch"""
        return self.total_samples // self.batch_size
