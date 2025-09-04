import pandas as pd
import numpy as np
import jax.numpy as jnp
import sys

df = pd.read_parquet(sys.argv[1])

# Create the 200 vectors of length 43788
result_vectors = df.pivot(index='ID', columns='gene_model', values='value')

# Convert to list of numpy arrays
vectors_list = [result_vectors.iloc[i].values for i in range(len(result_vectors))]

# Convert categorical values (1-8) to indices (0-7) for one-hot encoding
vectors_as_indices = [(vector - 1).astype(int) for vector in vectors_list]

# Create one-hot encoded targets
def to_one_hot(indices, num_classes=8):
   return jnp.eye(num_classes)[indices]

# Prepare JAX dataset
inputs = jnp.array(vectors_as_indices)  # Shape: (200, 43788) - flat vectors with indices 0-7
targets = jnp.array([to_one_hot(vec) for vec in vectors_as_indices])  # Shape: (200, 43788, 8) - one-hot

print(f"Input shape: {inputs.shape}")  # (200, 43788)
print(f"Target shape: {targets.shape}")  # (200, 43788, 8)
print(f"Input dtype: {inputs.dtype}")
print(f"Target dtype: {targets.dtype}")

print(f"\nSample input (first 10 values): {inputs[0][:10]}")
print(f"Sample target (first gene, all classes): {targets[0][0]}")

# Verify one-hot encoding worked
print(f"Input value: {inputs[0][0]}, One-hot: {targets[0][0]}")
