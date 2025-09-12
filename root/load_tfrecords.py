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
                 precomputed_global_mean=None, precomputed_global_std=None,
                 # Augmentation parameters
                 brightness_range=0.1,
                 contrast_range=(0.9, 1.1),
                 rotation_range=15.0,
                 shift_range=0.1,
                 zoom_range=(0.9, 1.1),
                 flip_horizontal=True,
                 flip_vertical=True,
                 noise_std=0.01):
        
        self.batch_size = batch_size
        self.enable_augmentation = enable_augmentation
        
        # Augmentation parameters
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.zoom_range = zoom_range
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.noise_std = noise_std
        
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
            print(f"Using precomputed stats (post-percentile normalization): mean={self.global_mean:.4f}, std={self.global_std:.4f}")
        else:
            print("Computing global statistics after percentile normalization...")
            self.global_mean, self.global_std = self._compute_global_stats()
            print(f"Computed stats (post-percentile normalization): mean={self.global_mean:.4f}, std={self.global_std:.4f}")
        
        if enable_augmentation:
            print("Data augmentation enabled with:")
            print(f"  - Brightness range: ±{brightness_range}")
            print(f"  - Contrast range: {contrast_range}")
            print(f"  - Rotation range: ±{rotation_range}°")
            print(f"  - Shift range: ±{shift_range * 100}%")
            print(f"  - Zoom range: {zoom_range}")
            print(f"  - Horizontal flip: {flip_horizontal}")
            print(f"  - Vertical flip: {flip_vertical}")
            print(f"  - Noise std: {noise_std}")
    
    def _percentile_normalization(self, img, lower_percentile=2.0, upper_percentile=98.0):
        """Apply percentile-based contrast normalization to avoid blowout from dark backgrounds"""
        # Flatten image for percentile calculation
        img_flat = tf.reshape(img, [-1])
        
        # Calculate percentiles
        # Use tfp.stats.percentile if available, otherwise approximate
        try:
            import tensorflow_probability as tfp
            p_low = tfp.stats.percentile(img_flat, lower_percentile)
            p_high = tfp.stats.percentile(img_flat, upper_percentile)
        except ImportError:
            # Fallback: use tf.nn.top_k to approximate percentiles
            sorted_vals, _ = tf.nn.top_k(img_flat, k=tf.size(img_flat))
            sorted_vals = tf.reverse(sorted_vals, axis=[0])  # Sort ascending
            
            n_pixels = tf.size(img_flat)
            low_idx = tf.cast(tf.cast(n_pixels, tf.float32) * lower_percentile / 100.0, tf.int32)
            high_idx = tf.cast(tf.cast(n_pixels, tf.float32) * upper_percentile / 100.0, tf.int32)
            
            low_idx = tf.clip_by_value(low_idx, 0, n_pixels - 1)
            high_idx = tf.clip_by_value(high_idx, 0, n_pixels - 1)
            
            p_low = sorted_vals[low_idx]
            p_high = sorted_vals[high_idx]
        
        # Avoid division by zero
        p_range = p_high - p_low
        p_range = tf.maximum(p_range, 1e-8)
        
        # Normalize to [0, 1] based on percentiles
        img_norm = (img - p_low) / p_range
        img_norm = tf.clip_by_value(img_norm, 0.0, 1.0)
        
        return img_norm
    
    def _count_samples(self):
        """Count total samples across all shards"""
        total = 0
        for tfrecord_file in self.tfrecord_files:
            total += sum(1 for _ in tf.data.TFRecordDataset([tfrecord_file]))
        return total
    
    def _compute_global_stats(self):
        """Compute global mean and std across entire dataset after percentile normalization"""
        dataset = tf.data.TFRecordDataset(self.tfrecord_files)
        
        def parse_features_only(example_proto):
            feature_description = {
                'features': tf.io.FixedLenFeature([], tf.string),
            }
            parsed = tf.io.parse_single_example(example_proto, feature_description)
            img = tf.io.decode_image(parsed['features'])
            img = tf.image.rgb_to_grayscale(img)
            img = tf.cast(img, tf.float32) / 255.0
            img = tf.squeeze(img, axis=-1)  # Remove channel dimension
            
            # Apply percentile normalization
            img = self._percentile_normalization(img)
            
            return img
        
        dataset = dataset.map(parse_features_only).batch(100)
        
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
    
    def _rotate_image(self, img, angle):
        """Rotate image by given angle (in radians)"""
        # Get image dimensions
        shape = tf.shape(img)
        height, width = shape[0], shape[1]
        
        # Create rotation matrix
        cos_angle = tf.cos(angle)
        sin_angle = tf.sin(angle)
        
        # Center coordinates
        cx = tf.cast(width, tf.float32) / 2.0
        cy = tf.cast(height, tf.float32) / 2.0
        
        # Rotation transform (translate to origin, rotate, translate back)
        transform = [cos_angle, -sin_angle, (-cos_angle + sin_angle) * cx + cx,
                    sin_angle, cos_angle, (-sin_angle - cos_angle) * cy + cy,
                    0.0, 0.0]
        
        return tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(img, 0),
            transforms=tf.expand_dims(transform, 0),
            output_shape=tf.shape(img)[:2],
            fill_mode='REFLECT',
            interpolation='BILINEAR',
            fill_value=0.0
        )[0]
    
    def _random_shift(self, img):
        """Apply random translation to the image"""
        shape = tf.shape(img)
        height, width = tf.cast(shape[0], tf.float32), tf.cast(shape[1], tf.float32)
        
        # Calculate shift amounts
        max_shift_h = height * self.shift_range
        max_shift_w = width * self.shift_range
        
        shift_h = tf.random.uniform([], -max_shift_h, max_shift_h)
        shift_w = tf.random.uniform([], -max_shift_w, max_shift_w)
        
        # Create translation transform
        transform = [1.0, 0.0, -shift_w,
                    0.0, 1.0, -shift_h,
                    0.0, 0.0]
        
        return tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(img, 0),
            transforms=tf.expand_dims(transform, 0),
            output_shape=tf.shape(img)[:2],
            fill_mode='REFLECT',
            interpolation='BILINEAR',
            fill_value=0.0
        )[0]
    
    def _random_zoom(self, img):
        """Apply random zoom to the image"""
        zoom_factor = tf.random.uniform([], self.zoom_range[0], self.zoom_range[1])
        
        shape = tf.shape(img)
        height, width = tf.cast(shape[0], tf.float32), tf.cast(shape[1], tf.float32)
        
        # Center coordinates
        cx = width / 2.0
        cy = height / 2.0
        
        # Zoom transform (scale around center)
        inv_zoom = 1.0 / zoom_factor
        transform = [inv_zoom, 0.0, (1.0 - inv_zoom) * cx,
                    0.0, inv_zoom, (1.0 - inv_zoom) * cy,
                    0.0, 0.0]
        
        return tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(img, 0),
            transforms=tf.expand_dims(transform, 0),
            output_shape=tf.shape(img)[:2],
            fill_mode='REFLECT',
            interpolation='BILINEAR',
            fill_value=0.0
        )[0]
    
    def _augment(self, img):
        """Apply comprehensive augmentations"""
        # Add channel dimension for TF image ops
        img = tf.expand_dims(img, axis=-1)
        
        # Random flips
        if self.flip_horizontal:
            img = tf.image.random_flip_left_right(img)
        if self.flip_vertical:
            img = tf.image.random_flip_up_down(img)
        
        # Random 90-degree rotations
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        img = tf.image.rot90(img, k=k)
        
        # Remove channel dimension for geometric transforms
        img = tf.squeeze(img, axis=-1)
        
        # Random rotation
        if self.rotation_range > 0:
            angle = tf.random.uniform([], 
                                    -self.rotation_range * np.pi / 180, 
                                    self.rotation_range * np.pi / 180)
            img = self._rotate_image(img, angle)
        
        # Random translation (shift)
        if self.shift_range > 0:
            img = self._random_shift(img)
        
        # Random zoom
        if self.zoom_range != (1.0, 1.0):
            img = self._random_zoom(img)
        
        # Add channel dimension back for brightness/contrast
        img = tf.expand_dims(img, axis=-1)
        
        # Brightness adjustment
        if self.brightness_range > 0:
            img = tf.image.random_brightness(img, max_delta=self.brightness_range)
        
        # Contrast adjustment
        if self.contrast_range != (1.0, 1.0):
            img = tf.image.random_contrast(img, 
                                         lower=self.contrast_range[0], 
                                         upper=self.contrast_range[1])
        
        # Remove channel dimension
        img = tf.squeeze(img, axis=-1)
        
        # Add random noise
        if self.noise_std > 0:
            noise = tf.random.normal(tf.shape(img), stddev=self.noise_std)
            img = img + noise
        
        # Clip values to valid range
        img = tf.clip_by_value(img, 0.0, 1.0)
        
        return img
    
    def _parse_example(self, example_proto):
        """Parse TFRecord example"""
        feature_description = {
            'features': tf.io.FixedLenFeature([], tf.string),
        }
        
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        
        # Decode image and convert to grayscale
        img = tf.io.decode_image(parsed['features'])
        img = tf.image.rgb_to_grayscale(img)
        img = tf.cast(img, tf.float32) / 255.0
        
        # Remove channel dimension for processing
        img = tf.squeeze(img, axis=-1)
        
        # Apply percentile normalization FIRST
        img = self._percentile_normalization(img)
        
        # Apply augmentations (after percentile normalization)
        if self.enable_augmentation:
            img = self._augment(img)
        
        # Global standardization (subtract mean and divide by std)
        img = (img - self.global_mean) / self.global_std
        
        return {
            'features': tf.expand_dims(img, axis=-1),  # [height, width, 1]
        }
    
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