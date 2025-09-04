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
    def __init__(self, tfrecord_pattern, crop_size=256, crops_per_image=16, 
                 batch_size=32, shuffle_buffer=1000, enable_augmentation=False,
                 nan_fill_value=0.0):
        
        self.tfrecord_pattern = tfrecord_pattern
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.enable_augmentation = enable_augmentation
        self.nan_fill_value = nan_fill_value
        
        # Calculate grid layout for crops
        self.grid_size = int(np.sqrt(crops_per_image))
        assert self.grid_size * self.grid_size == crops_per_image, \
            f"crops_per_image ({crops_per_image}) must be a perfect square"
        
        # Pre-calculate all crop coordinates as tensors for vectorized ops
        self._calculate_crop_coordinates()
        
        # Find all matching TFRecord files
        self.tfrecord_files = sorted(glob.glob(tfrecord_pattern))
        if not self.tfrecord_files:
            raise ValueError(f"No TFRecord files found matching pattern: {tfrecord_pattern}")
        
        print(f"Found {len(self.tfrecord_files)} TFRecord files")
        print(f"Will generate {crops_per_image} crops of {crop_size}x{crop_size} per image")
    
    def _calculate_crop_coordinates(self):
        """Pre-calculate crop coordinates as TensorFlow constants for vectorized extraction"""
        # For 1024x1024 image and grid of crops
        image_size = 1024
        step_size = (image_size - self.crop_size) // (self.grid_size - 1) if self.grid_size > 1 else 0
        
        # Create coordinate arrays
        starts_i = []
        starts_j = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                start_i = min(i * step_size, image_size - self.crop_size)
                start_j = min(j * step_size, image_size - self.crop_size)
                starts_i.append(start_i)
                starts_j.append(start_j)
        
        # Convert to TensorFlow constants for efficient vectorized ops
        self.crop_starts_i = tf.constant(starts_i, dtype=tf.int32)
        self.crop_starts_j = tf.constant(starts_j, dtype=tf.int32)
        
        # Pre-compute coordinate grids for tf.gather_nd
        self._precompute_gather_indices()
    
    def _precompute_gather_indices(self):
        """Precompute indices for vectorized crop extraction using tf.gather_nd"""
        # Create indices for all crops at once
        # Shape will be [crops_per_image, crop_size, crop_size, 2]
        all_indices = []
        
        for crop_idx in range(self.crops_per_image):
            start_i = self.crop_starts_i[crop_idx]
            start_j = self.crop_starts_j[crop_idx]
            
            # Create meshgrid for this crop
            i_coords = tf.range(start_i, start_i + self.crop_size)
            j_coords = tf.range(start_j, start_j + self.crop_size)
            i_grid, j_grid = tf.meshgrid(i_coords, j_coords, indexing='ij')
            
            # Stack coordinates for gather_nd
            indices = tf.stack([i_grid, j_grid], axis=-1)  # [crop_size, crop_size, 2]
            all_indices.append(indices)
        
        # Stack all crop indices: [crops_per_image, crop_size, crop_size, 2]
        self.gather_indices = tf.stack(all_indices)
    
    def _parse_example(self, example_proto):
        """Parse a single TFRecord example"""
        
        # Define the feature description
        feature_description = {
            # Raw arrays
            'intensity': tf.io.FixedLenFeature([], tf.string),
            'quality': tf.io.FixedLenFeature([], tf.string),
            
            # Array metadata
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            
            # Physical metadata
            'pixel_size_nm': tf.io.FixedLenFeature([], tf.float32),
            'magnification': tf.io.FixedLenFeature([], tf.float32),
            
            # Instrument metadata  
            'light_level_pct': tf.io.FixedLenFeature([], tf.float32),
            'numerical_aperture': tf.io.FixedLenFeature([], tf.float32),
        }
        
        # Parse the example
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        
        # Deserialize the arrays
        intensity = tf.io.parse_tensor(parsed['intensity'], out_type=tf.float32)
        quality = tf.io.parse_tensor(parsed['quality'], out_type=tf.float32)
        
        # Reshape to known dimensions
        intensity = tf.reshape(intensity, [parsed['height'], parsed['width']])
        quality = tf.reshape(quality, [parsed['height'], parsed['width']])
        
        return {
            'intensity': intensity,
            'quality': quality,
            'pixel_size_nm': parsed['pixel_size_nm'],
            'magnification': parsed['magnification'],
            'light_level_pct': parsed['light_level_pct'],
            'numerical_aperture': parsed['numerical_aperture'],
        }
    
    def _extract_all_crops_vectorized(self, data):
        """Vectorized extraction of all crops from a single image"""
        intensity = data['intensity']
        
        # Extract all crops using vectorized gather_nd
        # Use tf.map_fn to apply gather_nd for each crop's indices
        def extract_single_crop_indices(indices):
            return tf.gather_nd(intensity, indices)
        
        # Extract all crops at once: [crops_per_image, crop_size, crop_size]
        all_crops = tf.map_fn(
            extract_single_crop_indices,
            self.gather_indices,
            fn_output_signature=tf.TensorSpec([self.crop_size, self.crop_size], tf.float32),
            parallel_iterations=self.crops_per_image
        )
        
        # Handle NaNs vectorized
        all_crops = tf.where(tf.math.is_nan(all_crops), self.nan_fill_value, all_crops)
        
        # Standardize all crops at once
        global_mean = 50481.640625
        global_std = 16498.2578125
        all_crops = (all_crops - global_mean) / global_std
        
        # Apply augmentation if enabled (vectorized)
        if self.enable_augmentation:
            all_crops = self._apply_augmentation_vectorized(all_crops)
        
        # Add channel dimension: [crops_per_image, crop_size, crop_size, 1]
        all_crops = tf.expand_dims(all_crops, axis=-1)
        
        return all_crops
    
    def _apply_augmentation_vectorized(self, all_crops):
        """Apply augmentations to all crops simultaneously"""
        if not self.enable_augmentation:
            return all_crops
        
        # Add channel dimension for tf.image ops: [crops_per_image, crop_size, crop_size, 1]
        all_crops = tf.expand_dims(all_crops, axis=-1)
        
        # Apply augmentations to each crop
        def augment_single_crop(crop):
            # Random horizontal/vertical flips
            crop = tf.image.random_flip_left_right(crop)
            crop = tf.image.random_flip_up_down(crop)
            
            # Random 90-degree rotations
            k = tf.random.uniform([], 0, 4, dtype=tf.int32)
            crop = tf.image.rot90(crop, k=k)
            
            # Small random brightness adjustment
            crop = tf.image.random_brightness(crop, max_delta=0.1)
            
            # Small random contrast adjustment
            crop = tf.image.random_contrast(crop, lower=0.9, upper=1.1)
            
            return crop
        
        # Apply to all crops: [crops_per_image, crop_size, crop_size, 1]
        all_crops = tf.map_fn(
            augment_single_crop,
            all_crops,
            fn_output_signature=tf.TensorSpec([self.crop_size, self.crop_size, 1], tf.float32),
            parallel_iterations=self.crops_per_image
        )
        
        # Remove channel dimension for noise addition
        all_crops = tf.squeeze(all_crops, axis=-1)
        
        # Add gaussian noise vectorized
        noise = tf.random.normal(tf.shape(all_crops), mean=0.0, stddev=0.02)
        all_crops = all_crops + noise
        
        return all_crops
    
    def get_dataset(self, split='train', train_fraction=1.0, shuffle=True):
        """Get a tf.data.Dataset that yields individual crops with infinite repetition and proper epochs"""
        
        def create_epoch_dataset():
            """Create a single epoch of data"""
            # Create dataset from TFRecord files
            dataset = tf.data.TFRecordDataset(self.tfrecord_files)
            
            # Parse examples
            dataset = dataset.map(self._parse_example, num_parallel_calls=tf.data.AUTOTUNE)
            
            # Vectorized crop extraction function
            def create_crop_samples_vectorized(data):
                """Create all crop samples from a single image using vectorized operations"""
                # Extract all crops at once
                all_crops = self._extract_all_crops_vectorized(data)
                
                # Create metadata for all crops efficiently
                batch_size = self.crops_per_image
                
                return tf.data.Dataset.from_tensor_slices({
                    'features': all_crops,
                    'metadata': {
                        'pixel_size_nm': tf.broadcast_to(data['pixel_size_nm'], [batch_size]),
                        'magnification': tf.broadcast_to(data['magnification'], [batch_size]),
                        'light_level_pct': tf.broadcast_to(data['light_level_pct'], [batch_size]),
                        'numerical_aperture': tf.broadcast_to(data['numerical_aperture'], [batch_size]),
                        'crop_idx': tf.range(batch_size, dtype=tf.int32)
                    }
                })
            
            # Shuffle images first (different order each epoch)
            if shuffle:
                dataset = dataset.shuffle(self.shuffle_buffer)
            
            # Extract all crops using interleave
            dataset = dataset.interleave(
                create_crop_samples_vectorized,
                cycle_length=tf.data.AUTOTUNE,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False if shuffle else True
            )
            
            # Shuffle individual crops within the epoch
            if shuffle:
                # Use a larger buffer to mix crops from different images well
                crop_shuffle_buffer = min(self.shuffle_buffer * self.crops_per_image, 10000)
                dataset = dataset.shuffle(crop_shuffle_buffer)
            
            return dataset
        
        # Create infinitely repeating dataset where each epoch contains all crops exactly once
        dataset = tf.data.Dataset.range(1).repeat().flat_map(lambda _: create_epoch_dataset())
        
        # Batch the individual crop samples
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        
        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_jax_iterator(self, split='train', dtype=jnp.float32, **kwargs):
        """Get JAX iterator using the standard pattern"""
        tfds = self.get_dataset(**kwargs)
        return create_input_iter(tfds, dtype=dtype)
    
