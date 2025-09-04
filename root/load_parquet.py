import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax

def load_genetic_data(filepath, gene_parents=None, k=2000, selection_method='balanced'):
    """
    Load and process genetic data from parquet file with gene selection
    
    Args:
        filepath: Path to parquet file
        gene_parents: Array of parent categories for each gene (0-7), shape (num_genes,)
        k: Number of genes to select
        selection_method: 'balanced', 'variance', 'cv', or 'all'
    """
    import pandas as pd
    import numpy as np
    import jax.numpy as jnp
    
    df = pd.read_parquet(filepath)
    
    # Create the full vectors first
    result_vectors = df.pivot(index='ID', columns='gene_model', values='value')
    
    # Convert to numpy for easier manipulation
    full_data = result_vectors.values  # Shape: (num_samples, num_genes)
    
    print(f"Original data shape: {full_data.shape}")
    
    # Gene selection based on method
    if selection_method == 'all':
        selected_indices = np.arange(full_data.shape[1])
        selected_data = full_data
    elif selection_method == 'variance':
        # Simple variance-based selection
        gene_variances = np.var(full_data, axis=0)
        selected_indices = np.argsort(gene_variances)[-k:][::-1]
        selected_data = full_data[:, selected_indices]
    elif selection_method == 'cv':
        # Coefficient of variation
        gene_means = np.mean(full_data, axis=0)
        gene_stds = np.std(full_data, axis=0)
        cv = gene_stds / (gene_means + 1e-8)
        selected_indices = np.argsort(cv)[-k:][::-1]
        selected_data = full_data[:, selected_indices]
    elif selection_method == 'balanced' and gene_parents is not None:
        # Balanced selection across parent categories
        genes_per_parent = k // 8
        selected_indices = []
        
        for parent_id in range(8):
            parent_mask = (gene_parents == parent_id)
            if not np.any(parent_mask):
                continue
                
            parent_genes = full_data[:, parent_mask]
            
            # Use coefficient of variation within this parent
            means = np.mean(parent_genes, axis=0)
            stds = np.std(parent_genes, axis=0)
            cv = stds / (means + 1e-8)
            
            # Get top genes from this parent
            top_indices_in_parent = np.argsort(cv)[-genes_per_parent:]
            parent_gene_indices = np.where(parent_mask)[0]
            selected_indices.extend(parent_gene_indices[top_indices_in_parent])
        
        selected_indices = np.array(selected_indices)
        selected_data = full_data[:, selected_indices]
    else:
        raise ValueError("For balanced selection, gene_parents must be provided")
    
    print(f"Selected {len(selected_indices)} genes using {selection_method} method")
    print(f"Reduced data shape: {selected_data.shape}")
    
    # Convert categorical values (1-8) to indices (0-7) for one-hot encoding
    vectors_as_indices = (selected_data - 1).astype(int)
    
    # Create one-hot encoded targets
    def to_one_hot(indices, num_classes=8):
        return jnp.eye(num_classes)[indices]
    
    # Prepare JAX dataset
    inputs = jnp.array(vectors_as_indices)  # Shape: (num_samples, selected_genes)
    targets = jnp.array([to_one_hot(vec) for vec in vectors_as_indices])  # Shape: (num_samples, selected_genes, 8)
    
    print(f"Final data - Input shape: {inputs.shape}, Target shape: {targets.shape}")
    
    # Return selected indices too, in case you need them later
    return inputs, targets, selected_indices

def create_random_batches(inputs, targets, batch_size, key, drop_remainder=True):
    """Create randomized batches for one epoch"""
    num_samples = inputs.shape[0]
    
    if drop_remainder:
        num_batches = num_samples // batch_size
        total_samples = num_batches * batch_size
    else:
        num_batches = (num_samples + batch_size - 1) // batch_size
        total_samples = num_samples
    
    # Shuffle indices
    indices = jax.random.permutation(key, num_samples)
    
    if drop_remainder:
        indices = indices[:total_samples]
    
    # Create batches
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        batch_indices = indices[start_idx:end_idx]
        
        batch_inputs = inputs[batch_indices]
        batch_targets = targets[batch_indices]
        batches.append((batch_inputs, batch_targets))
    
    return batches

