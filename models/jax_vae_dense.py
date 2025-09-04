from dataclasses import field
import jax
import jax.numpy as jnp
import flax.linen as nn


class Encoder(nn.Module):
    latent_dim: int 

    @nn.compact
    def __call__(self, x):
        h = x.astype(jnp.float32)

        # Encoder with residual connections
        for i, dim in enumerate([128, 128, 128]):
            h = nn.Dense(dim)(h)
            h = nn.LayerNorm()(h)
            h = nn.gelu(h)
        # Latent parameters
        mu = nn.Dense(self.latent_dim)(h)
        logvar = nn.Dense(self.latent_dim)(h)

        return mu, logvar

class Decoder(nn.Module):
    num_genes: int
    num_classes: int

    @nn.compact
    def __call__(self, h):
        for dim in [128, 128, 128]:
            h = nn.Dense(dim)(h)
            h = nn.LayerNorm()(h)
            h = nn.gelu(h)

        logits = nn.Dense(self.num_genes * self.num_classes)(h)
        logits = logits.reshape(-1, self.num_genes, self.num_classes)

        return logits

class VAE(nn.Module):
    latent_dim: int = 128
    num_classes: int = 8
    num_genes: int = 1
    
    def setup(self, ):
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.num_genes, self.num_classes)

    def reparameterize(self, mu, logvar, key):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, mu.shape)
        return mu + eps * std
    
    def __call__(self, x, key):
        # Encode
        mu, logvar = self.encoder(x)
        # Reparameterize
        z = self.reparameterize(mu, logvar, key)
        # Decode
        logits = self.decoder(z)
        
        return logits, mu, logvar
