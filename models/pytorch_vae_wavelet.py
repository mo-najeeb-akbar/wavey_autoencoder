import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, filters, block_name=None):
        super().__init__()
        self.filters = filters
        self.block_name = block_name or "res"
        
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=filters)
        
    def forward(self, x):
        skip = x
        
        # First conv
        out = self.conv1(x)
        out = self.gn1(out)
        out = F.silu(out)
        
        # Second conv
        out = self.conv2(out)
        out = self.gn2(out)

        return F.silu(out + skip)

class Encoder(nn.Module):
    def __init__(self, latent_dim, features, input_size=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.features = features
        
        # Calculate the size after 4 downsampling operations (stride=2 each)
        # input_size -> input_size/2 -> input_size/4 -> input_size/8 -> input_size/16
        self.final_spatial_size = input_size // (2**4)
        self.flattened_size = self.final_spatial_size * self.final_spatial_size * features
        
        # Downsampling layers - use padding=1 to match JAX SAME padding for 3x3 kernels
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(4 if i == 0 else features, features, 3, stride=2, padding=1, bias=False)
            for i in range(4)
        ])
        # self.bn_layers = nn.ModuleList([
        #     nn.BatchNorm2d(features, momentum=0.1, eps=1e-5) for _ in range(4)
        # ])
        self.gn_layers = nn.ModuleList([
            nn.GroupNorm(num_groups=8, num_channels=features) for _ in range(4)
        ])
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(features) for i in range(4)
        ])
        
        # Dense layers - using calculated flattened size
        self.dense_0 = nn.Linear(self.flattened_size, 128, bias=False)
        self.ln_0 = nn.LayerNorm(128)
        # self.bn_0 = nn.BatchNorm1d(128, momentum=0.1, eps=1e-5)
        
        # VAE outputs
        self.dense_mu = nn.Linear(128, latent_dim)
        self.dense_logvar = nn.Linear(128, latent_dim)
    
    def forward(self, x):
        # Downsampling blocks
        for i in range(4):
            x = self.conv_layers[i](x)
            # x = self.bn_layers[i](x)
            x = self.gn_layers[i](x)
            x = F.silu(x)
            x = self.residual_blocks[i](x)
        
        # Flatten and process - use reshape instead of view
        x = x.reshape(x.size(0), -1)
        x = self.dense_0(x)
        x = self.ln_0(x)

        x = F.silu(x)
        
        # VAE outputs
        mu = self.dense_mu(x)
        log_var = self.dense_logvar(x)
        
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, bottle_neck, features):
        super().__init__()
        self.latent_dim = latent_dim
        self.bottle_neck = bottle_neck
        self.features = features
        
        # Initial processing
        self.dense_0 = nn.Linear(latent_dim, 128, bias=False)
        self.ln_0 = nn.LayerNorm(128)
        
        # Reshape layer
        block_size = bottle_neck**2 * features
        self.dense_1 = nn.Linear(128, block_size, bias=False)
        self.ln_1 = nn.LayerNorm(block_size)
        
        # Upsampling layers
        self.tonv_layers = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=features, out_channels=features, bias=False, kernel_size=2, stride=2, padding=0) for _ in range(4)
        ])
        # self.bn_layers = nn.ModuleList([
        #    nn.BatchNorm2d(features, momentum=0.1, eps=1e-5) for _ in range(4)
        #])
        self.gn_layers = nn.ModuleList([
            nn.GroupNorm(num_groups=8, num_channels=features) for _ in range(4)
        ])
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(features) for i in range(4)
        ])
        
        # Final output
        self.out_conv = nn.Conv2d(features, 4, 3, padding=1)
    
    def forward(self, x):
        # Initial processing
        x = self.dense_0(x)
        x = self.ln_0(x)
        x = F.silu(x)
        # Reshape to spatial
        x = self.dense_1(x)
        x = self.ln_1(x)
        x = F.silu(x)
        x = x.reshape(x.size(0), self.features, self.bottle_neck, self.bottle_neck)
        
        # Upsampling blocks
        for i in range(4):
            # Conv and residual
            x = self.tonv_layers[i](x)
            x = self.gn_layers[i](x)
            x = F.silu(x)
            x = self.residual_blocks[i](x)
        
        # Final output
        x = self.out_conv(x)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=128, base_features=32, block_size=8, input_size=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_features = base_features
        self.block_size = block_size
        self.input_size = input_size
        
        self.encoder = Encoder(latent_dim, base_features, input_size)
        self.decoder = Decoder(latent_dim, block_size, base_features)
        
        filters = torch.tensor([
            [[[0.5, 0.5], [0.5, 0.5]]],
            [[[0.5, -0.5], [0.5, -0.5]]],
            [[[0.5, 0.5], [-0.5, -0.5]]],
            [[[0.5, -0.5], [-0.5, 0.5]]]
        ], dtype=torch.float32)
        self.register_buffer('filters', filters)

        self.fdwt = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        self.fdwt.weight = nn.Parameter(filters, requires_grad=False)
        
        self.idwt = nn.ConvTranspose2d(4, 1, kernel_size=2, stride=2, bias=False)
        self.idwt.weight = nn.Parameter(filters, requires_grad=False)
        

    # NOTE: THIS IS NOT A VAE UNLESS YOU UNCOMMENT THE BELOW USE OF STD & EPS
    def reparameterize(self, mu, log_var):
        # std = torch.exp(0.5 * log_var)
        # eps = torch.randn_like(std)
        return mu # + std 
    
    def forward(self, x):
        x = self.fdwt(x)
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_waves = self.decoder(z)
        x_recon = self.idwt(x_waves)
        return x_recon, x_waves, mu, log_var
