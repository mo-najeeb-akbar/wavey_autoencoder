import tensorflow as tf
import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from flax.jax_utils import prefetch_to_device
from functools import partial
from pathlib import Path
import glob
import json

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
                 batch_size=32, 
                 shuffle_buffer=1000, 
                 enable_augmentation=False,
                 normalize_method='standardize',  # 'standardize', 'minmax', or 'none'
                 target_dtype=jnp.float32):
        
        self.tfrecord_pattern = tfrecord_pattern
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.enable_augmentation = enable_augmentation
        self.normalize_method = normalize_method
        self.target_dtype = target_dtype
        
        # Find TFRecord files
        self.tfrecord_files = sorted(glob.glob(tfrecord_pattern))
        
        if not self.tfrecord_files:
            raise ValueError(f"No TFRecord files found matching pattern: {tfrecord_pattern}")
        
        print(f"Found {len(self.tfrecord_files)} TFRecord files")
        
        # Load metadata if available
        self.metadata = self._load_metadata()
        if self.metadata:
            print(f"Dataset metadata: {self.metadata['total_output_samples']} samples")
            if self.metadata['cropping']['enabled']:
                crop_info = self.metadata['cropping']
                print(f"Crop size: {crop_info['crop_size']}x{crop_info['crop_size']}, "
                      f"overlap: {crop_info['overlap']}px, "
                      f"~{crop_info['crops_per_input']} crops per input image")
    
    def _load_metadata(self):
        """Load dataset metadata if available"""
        # Try to find metadata file based on TFRecord pattern
        pattern_path = Path(self.tfrecord_pattern)
        possible_metadata_files = [
            pattern_path.parent / f"{pattern_path.stem}_metadata.json",
            pattern_path.parent / "dataset_metadata.json",
        ]
        
        for metadata_file in possible_metadata_files:
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load metadata from {metadata_file}: {e}")
                    
        return None
    
    def _parse_example(self, example_proto):
        """Parse a single TFRecord example"""
        
        # Define the feature description matching what we stored
        feature_description = {
            # Raw data
            'intensity': tf.io.FixedLenFeature([], tf.string),
            'quality': tf.io.FixedLenFeature([], tf.string),
            
            # Array metadata
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            
            # Physical metadata
            'pixel_size_nm': tf.io.FixedLenFeature([], tf.float32),
            'magnification': tf.io.FixedLenFeature([], tf.float32),
            'field_of_view_um': tf.io.FixedLenFeature([], tf.float32),
            
            # Crop metadata
            'is_crop': tf.io.FixedLenFeature([], tf.int64),
            'crop_id': tf.io.FixedLenFeature([], tf.int64),
            'crop_row': tf.io.FixedLenFeature([], tf.int64),
            'crop_col': tf.io.FixedLenFeature([], tf.int64),
            'crop_size': tf.io.FixedLenFeature([], tf.int64),
            'total_crops_from_source': tf.io.FixedLenFeature([], tf.int64),
            
            # Instrument metadata
            'light_level_pct': tf.io.FixedLenFeature([], tf.float32),
            'numerical_aperture': tf.io.FixedLenFeature([], tf.float32),
            'wavelength_nm': tf.io.FixedLenFeature([], tf.float32),
            'frame_count': tf.io.FixedLenFeature([], tf.int64),
        }
        
        # Parse the example
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        
        # Deserialize the intensity and quality arrays
        intensity = tf.io.parse_tensor(parsed['intensity'], out_type=tf.float32)
        quality = tf.io.parse_tensor(parsed['quality'], out_type=tf.float32)
        
        # Set shapes (TensorFlow needs this for batching)
        height = parsed['height']
        width = parsed['width']
        intensity = tf.reshape(intensity, [height, width])
        quality = tf.reshape(quality, [height, width])
        
        # Handle NaN values (replace with zeros for now, or you could use masking)
        intensity = tf.where(tf.math.is_nan(intensity), 0.0, intensity)
        quality = tf.where(tf.math.is_nan(quality), 0.0, quality)
        
        # Stack intensity and quality to create a 2-channel image
        # Shape: [height, width, 2]
        features = tf.stack([intensity, quality], axis=-1)
        
        # Normalize based on the specified method
        if self.normalize_method == 'standardize':
            # Per-channel standardization
            mean = tf.reduce_mean(features, axis=[0, 1], keepdims=True)
            variance = tf.reduce_mean(tf.square(features - mean), axis=[0, 1], keepdims=True)
            features = (features - mean) / tf.sqrt(variance + 1e-8)
        elif self.normalize_method == 'minmax':
            # Per-channel min-max normalization to [0, 1]
            min_val = tf.reduce_min(features, axis=[0, 1], keepdims=True)
            max_val = tf.reduce_max(features, axis=[0, 1], keepdims=True)
            features = (features - min_val) / (max_val - min_val + 1e-8)
        # 'none' means no normalization
        
        # Apply augmentations if enabled
        if self.enable_augmentation:
            features = self._apply_augmentations(features)
        
        # Convert all numeric values to the target dtype
        def convert_dtype(x):
            if x.dtype.is_floating:
                return tf.cast(x, tf.float32)  # Keep as float32 for TensorFlow
            else:
                return x  # Keep integers as-is
        
        return {
            # Main data
            'features': features,  # [height, width, 2] - intensity and quality
            
            # Physical metadata (useful for downstream tasks)
            'pixel_size_nm': convert_dtype(parsed['pixel_size_nm']),
            'magnification': convert_dtype(parsed['magnification']),
            'field_of_view_um': convert_dtype(parsed['field_of_view_um']),
            
            # Crop metadata (useful for tracking and reconstruction)
            'is_crop': parsed['is_crop'],
            'crop_id': parsed['crop_id'],
            'crop_position': tf.stack([parsed['crop_row'], parsed['crop_col']]),  # [2] - row, col
            'crop_size': parsed['crop_size'],
            'total_crops_from_source': parsed['total_crops_from_source'],
            
            # Instrument metadata
            'light_level_pct': convert_dtype(parsed['light_level_pct']),
            'numerical_aperture': convert_dtype(parsed['numerical_aperture']),
            'wavelength_nm': convert_dtype(parsed['wavelength_nm']),
            'frame_count': parsed['frame_count'],
            
            # Derived metadata
            'height': parsed['height'],
            'width': parsed['width'],
        }
    
    def _apply_augmentations(self, features):
        """Apply data augmentations to the features"""
        # Random horizontal flip
        if tf.random.uniform([]) > 0.5:
            features = tf.image.flip_left_right(features)
        
        # Random vertical flip
        if tf.random.uniform([]) > 0.5:
            features = tf.image.flip_up_down(features)
        
        # Random 90-degree rotations
        k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        features = tf.image.rot90(features, k)
        
        # Small random brightness/contrast adjustments for intensity channel only
        if tf.random.uniform([]) > 0.5:
            intensity = features[:, :, 0:1]
            quality = features[:, :, 1:2]
            
            # Random brightness adjustment
            intensity = tf.image.random_brightness(intensity, max_delta=0.1)
            
            # Random contrast adjustment
            intensity = tf.image.random_contrast(intensity, lower=0.9, upper=1.1)
            
            features = tf.concat([intensity, quality], axis=-1)
        
        return features
    
    def get_dataset(self, split='train', train_fraction=1.0, shuffle=True):
        """Get a tf.data.Dataset that yields individual crops with infinite repetition"""
        
        # Create dataset from TFRecord files
        dataset = tf.data.TFRecordDataset(self.tfrecord_files)
        
        # Parse examples
        dataset = dataset.map(self._parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Optional: filter by split if you want to implement train/val splits
        # You could use crop_id or sample hash for deterministic splitting
        if split == 'train' and train_fraction < 1.0:
            # Simple deterministic split based on crop_id
            dataset = dataset.filter(
                lambda x: tf.math.floormod(x['crop_id'], 10) < int(train_fraction * 10)
            )
        elif split == 'val' and train_fraction < 1.0:
            dataset = dataset.filter(
                lambda x: tf.math.floormod(x['crop_id'], 10) >= int(train_fraction * 10)
            )
        
        # Repeat indefinitely
        dataset = dataset.repeat()
        
        # Shuffle if requested
        if shuffle:
            # Use a larger buffer to mix crops from different images well
            crop_shuffle_buffer = max(self.shuffle_buffer, 4 * self.batch_size)
            dataset = dataset.shuffle(crop_shuffle_buffer)
        
        # Batch the individual crop samples
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        
        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_jax_iterator(self, split='train', dtype=None, **kwargs):
        """Get JAX iterator using the standard pattern"""
        if dtype is None:
            dtype = self.target_dtype
            
        tfds = self.get_dataset(split=split, **kwargs)
        return create_input_iter(tfds, dtype=dtype)
    
    def get_sample_batch(self, split='train', **kwargs):
        """Get a single batch for inspection"""
        dataset = self.get_dataset(split=split, shuffle=False, **kwargs)
        return next(iter(dataset))
    
    def print_batch_info(self, batch):
        """Print information about a batch (useful for debugging)"""
        print(f"Batch shapes:")
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {type(value)}")
        
        # Print some statistics
        features = batch['features']
        print(f"\nFeature statistics:")
        print(f"  Min: {tf.reduce_min(features):.4f}")
        print(f"  Max: {tf.reduce_max(features):.4f}")
        print(f"  Mean: {tf.reduce_mean(features):.4f}")
        print(f"  Std: {tf.math.reduce_std(features):.4f}")
        
        # Print crop info
        is_crop = batch['is_crop'][0]
        if is_crop:
            crop_size = batch['crop_size'][0]
            crop_pos = batch['crop_position'][0]
            print(f"\nCrop info (first sample):")
            print(f"  Crop size: {crop_size}x{crop_size}")
            print(f"  Position: ({crop_pos[0]}, {crop_pos[1]})")
            print(f"  From {batch['total_crops_from_source'][0]} total crops")


# Example usage and testing functions
def test_dataloader(tfrecord_pattern, batch_size=8):
    """Test the dataloader with a small batch"""
    print(f"Testing dataloader with pattern: {tfrecord_pattern}")
    
    # Create dataloader
    dataloader = CroppedDataloader(
        tfrecord_pattern=tfrecord_pattern,
        batch_size=batch_size,
        shuffle_buffer=100,
        enable_augmentation=True,
        normalize_method='global_standardize',
        nan_fill_value=0.0
    )
    
    # Test TensorFlow dataset
    print("\n--- Testing TensorFlow dataset ---")
    tf_batch = dataloader.get_sample_batch()
    dataloader.print_batch_info(tf_batch)
    
    # Test JAX iterator
    print("\n--- Testing JAX iterator ---")
    jax_iter = dataloader.get_jax_iterator(shuffle=False)
    jax_batch = next(jax_iter)
    
    print(f"JAX batch shapes:")
    for key, value in jax_batch.items():
        print(f"  {key}: {value.shape} ({value.dtype})")
    
    print(f"\nJAX feature statistics (intensity only):")
    features = jax_batch['features']
    print(f"  Min: {jnp.min(features):.4f}")
    print(f"  Max: {jnp.max(features):.4f}")
    print(f"  Mean: {jnp.mean(features):.4f}")
    print(f"  Std: {jnp.std(features):.4f}")
    
    # Print dataset info
    print(f"\nDataset info:")
    print(f"  Total samples: {dataloader.get_sample_count()}")
    print(f"  Batches per epoch: {dataloader.get_batches_per_epoch()}")
    
    return dataloader, jax_batch

