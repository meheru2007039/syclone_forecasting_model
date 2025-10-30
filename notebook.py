import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from pathlib import Path
from datetime import datetime
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import inception_v3
import torchvision.models as models
from scipy import linalg
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import random

# ============================================================================
# DATA LOADER UTILITIES
# ============================================================================

def get_all_storms(root_dir, years):
    """
    Scan the data directory and collect all available storm names across specified years.

    Args:
        root_dir: Root directory containing year folders
        years: List of year folder names (e.g., ['2005_0', '2016_0', '2022_0'])

    Returns:
        Dictionary mapping storm_name -> {'year': year, 'num_timesteps': count}
    """
    root_path = Path(root_dir)
    all_storms = {}

    for year_folder in years:
        year_path = root_path / year_folder
        if not year_path.exists():
            print(f"Warning: Year folder {year_folder} not found at {year_path}")
            continue

        # Handle nested directory structure (year/year/storm)
        nested_path = year_path / year_folder
        if nested_path.exists():
            year_path = nested_path

        # Iterate through storm folders
        for cyclone_folder in year_path.iterdir():
            if not cyclone_folder.is_dir():
                continue

            storm_name = cyclone_folder.name

            # Count timestep folders
            timestep_folders = [f for f in cyclone_folder.iterdir() if f.is_dir()]
            num_timesteps = len(timestep_folders)

            # Store storm info
            if storm_name not in all_storms:
                all_storms[storm_name] = {
                    'year': year_folder,
                    'num_timesteps': num_timesteps,
                    'num_samples': max(0, num_timesteps - 1)  # consecutive pairs
                }

    return all_storms


def create_random_storm_split(root_dir, years, num_val_storms=5, test_storms=None,
                              seed=42, min_timesteps=5):
    """
    Create a random storm-level split for train/val/test sets.

    Args:
        root_dir: Root directory containing year folders
        years: List of year folder names
        num_val_storms: Number of storms to use for validation
        test_storms: List of storm names to use for testing (kept as-is)
        seed: Random seed for reproducibility
        min_timesteps: Minimum number of timesteps a storm must have to be included

    Returns:
        Dictionary with 'train_storms', 'val_storms', 'test_storms' lists
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Get all available storms
    all_storms = get_all_storms(root_dir, years)

    if not all_storms:
        raise ValueError(f"No storms found in {root_dir} for years {years}")

    # Filter storms by minimum timesteps
    valid_storms = {
        name: info for name, info in all_storms.items()
        if info['num_timesteps'] >= min_timesteps
    }

    print(f"\nFound {len(all_storms)} total storms, {len(valid_storms)} with >= {min_timesteps} timesteps")

    # Initialize test storms (keep as specified)
    if test_storms is None:
        test_storms = []

    # Get list of storm names available for train/val split
    available_storms = [name for name in valid_storms.keys() if name not in test_storms]

    if len(available_storms) < num_val_storms:
        raise ValueError(
            f"Not enough storms for split. Available: {len(available_storms)}, "
            f"Requested val: {num_val_storms}"
        )

    # Randomly select validation storms
    random.shuffle(available_storms)
    val_storms = available_storms[:num_val_storms]
    train_storms = available_storms[num_val_storms:]

    # Print split statistics
    print(f"\nStorm-level split:")
    print(f"  Training: {len(train_storms)} storms")
    print(f"  Validation: {len(val_storms)} storms")
    print(f"  Test: {len(test_storms)} storms")

    # Calculate sample counts
    train_samples = sum(valid_storms[s]['num_samples'] for s in train_storms)
    val_samples = sum(valid_storms[s]['num_samples'] for s in val_storms)
    test_samples = sum(valid_storms[s]['num_samples'] for s in test_storms if s in valid_storms)

    print(f"\nExpected sample counts:")
    print(f"  Training: {train_samples} samples")
    print(f"  Validation: {val_samples} samples")
    print(f"  Test: {test_samples} samples")

    print(f"\nValidation storms: {val_storms}")
    print(f"Test storms: {test_storms}")

    return {
        'train_storms': train_storms,
        'val_storms': val_storms,
        'test_storms': test_storms,
        'storm_info': valid_storms
    }


# ============================================================================
# DATA LOADER
# ============================================================================

class CycloneDataset(Dataset):
    def __init__(self, root_dir, years, gridsat_type='GRIDSAT_data.npy', 
                 era5_type='ERA5_data.npy', transform=None, max_samples=None,
                 test_storm=None, val_storm=None, mode='train', img_size=256):
        self.root_dir = Path(root_dir)
        self.gridsat_type = gridsat_type
        self.era5_type = era5_type
        self.transform = transform
        self.img_size = img_size
        self.samples = []
        self.test_storm = test_storm if isinstance(test_storm, list) else [test_storm] if test_storm else []
        self.val_storm = val_storm if isinstance(val_storm, list) else [val_storm] if val_storm else []
        self.mode = mode
        
        self._load_samples(years, max_samples)
        
    def _load_samples(self, years, max_samples):
        for year_folder in years:
            year_path = self.root_dir / year_folder
            if not year_path.exists():
                print(f"Warning: Year folder {year_folder} not found at {year_path}")
                continue
        
            nested_path = year_path / year_folder
            if nested_path.exists():
                year_path = nested_path
                
            for cyclone_folder in year_path.iterdir():
                if not cyclone_folder.is_dir():
                    continue
                
                storm_name = cyclone_folder.name
            
                if self.mode == 'train':
                    if storm_name in self.test_storm or storm_name in self.val_storm:
                        continue
                elif self.mode == 'val':
                    if storm_name not in self.val_storm:
                        continue
                elif self.mode == 'test':
                    if storm_name not in self.test_storm:
                        continue
                
                timestep_folders = sorted([f for f in cyclone_folder.iterdir() if f.is_dir()])
                
                for i in range(len(timestep_folders) - 1):
                    current_time = timestep_folders[i]
                    next_time = timestep_folders[i + 1]
                    
                    current_gridsat = current_time / self.gridsat_type
                    current_era5 = current_time / self.era5_type
                    next_gridsat = next_time / self.gridsat_type
                    
                    if (current_gridsat.exists() and current_era5.exists() and 
                        next_gridsat.exists()):
                        
                        timestamp = current_time.name
                        
                        self.samples.append({
                            'current_gridsat': str(current_gridsat),
                            'current_era5': str(current_era5),
                            'next_gridsat': str(next_gridsat),
                            'timestamp': timestamp,
                            'storm_name': storm_name,
                            'year': year_folder
                        })
                        
                        if max_samples and len(self.samples) >= max_samples:
                            return
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        current_gridsat = np.load(sample_info['current_gridsat']).astype(np.float32)
        current_gridsat = np.nan_to_num(current_gridsat, nan=0.0)
        
        current_era5 = np.load(sample_info['current_era5']).astype(np.float32)
        current_era5 = np.nan_to_num(current_era5, nan=0.0)
        
        next_gridsat = np.load(sample_info['next_gridsat']).astype(np.float32)
        next_gridsat = np.nan_to_num(next_gridsat, nan=0.0)
        
        if current_gridsat.ndim == 3:
            current_gridsat = current_gridsat[0:1, :, :]
        elif current_gridsat.ndim == 2:
            current_gridsat = current_gridsat[np.newaxis, ...]
        
        if next_gridsat.ndim == 3:
            next_gridsat = next_gridsat[0:1, :, :]
        elif next_gridsat.ndim == 2:
            next_gridsat = next_gridsat[np.newaxis, ...]
        
        if current_era5.ndim == 3:
            current_era5 = current_era5[0:4, :, :]
        elif current_era5.ndim == 2:
            current_era5 = np.stack([current_era5] * 4, axis=0)
        
        assert current_gridsat.shape[0] == 1, f"GRIDSAT should have 1 channel, got {current_gridsat.shape[0]}"
        assert current_era5.shape[0] == 4, f"ERA5 should have 4 channels, got {current_era5.shape[0]}"
        assert next_gridsat.shape[0] == 1, f"Next GRIDSAT should have 1 channel, got {next_gridsat.shape[0]}"
        
        current_gridsat = torch.from_numpy(current_gridsat)
        current_era5 = torch.from_numpy(current_era5)
        next_gridsat = torch.from_numpy(next_gridsat)
        
        target_size = (self.img_size, self.img_size)
        
        current_gridsat = F.interpolate(
            current_gridsat.unsqueeze(0), 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        current_era5 = F.interpolate(
            current_era5.unsqueeze(0), 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        next_gridsat = F.interpolate(
            next_gridsat.unsqueeze(0), 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        eps = 1e-8
        
        gs_min, gs_max = float(current_gridsat.min()), float(current_gridsat.max())
        if gs_max - gs_min > eps:
            current_gridsat = (current_gridsat - gs_min) / (gs_max - gs_min + eps)
            current_gridsat = current_gridsat * 2 - 1
        
        for c in range(current_era5.shape[0]):
            e_min = float(current_era5[c].min())
            e_max = float(current_era5[c].max())
            if e_max - e_min > eps:
                current_era5[c] = (current_era5[c] - e_min) / (e_max - e_min + eps)
                current_era5[c] = current_era5[c] * 2 - 1
        
        ngs_min, ngs_max = float(next_gridsat.min()), float(next_gridsat.max())
        if ngs_max - ngs_min > eps:
            next_gridsat = (next_gridsat - ngs_min) / (ngs_max - ngs_min + eps)
            next_gridsat = next_gridsat * 2 - 1
        
        if self.transform:
            current_gridsat = self.transform(current_gridsat)
            current_era5 = self.transform(current_era5)
            next_gridsat = self.transform(next_gridsat)
        
        return {
            'gridsat': current_gridsat,
            'era5': current_era5,
            'target': next_gridsat,
            'timestamp': sample_info['timestamp'],
            'storm_name': sample_info['storm_name'],
            'year': sample_info['year']
        }


def get_dataloaders(root_dir, batch_size=4, num_workers=2, 
                                    train_years=['2005_0', '2016_0', '2022_0'], 
                                    val_years=['2022_0'],
                                    test_years=['2022_0'],
                                    gridsat_type='GRIDSAT_data.npy',
                                    era5_type='ERA5_data.npy',
                                    max_samples=None,
                                    test_storm=["2022349N13068"],
                                    val_storm=["2022345N17125"],
                                    img_size=256):
    
    train_dataset = CycloneDataset(
        root_dir=root_dir,
        years=train_years,
        gridsat_type=gridsat_type,
        era5_type=era5_type,
        max_samples=max_samples,
        test_storm=test_storm,
        val_storm=val_storm,
        mode='train',
        img_size=img_size
    )
    
    # Validation dataset (only val storms)
    val_dataset = CycloneDataset(
        root_dir=root_dir,
        years=val_years,
        gridsat_type=gridsat_type,
        era5_type=era5_type,
        test_storm=test_storm,
        val_storm=val_storm,
        mode='val',
        img_size=img_size
    )
    
    # Test dataset (only test storms)
    test_dataset = CycloneDataset(
        root_dir=root_dir,
        years=test_years,
        gridsat_type=gridsat_type,
        era5_type=era5_type,
        test_storm=test_storm,
        val_storm=val_storm,
        mode='test',
        img_size=img_size
    )
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# ============================================================================
# CONDITION ENCODING
# ============================================================================

class SpatialPositionalEncoding(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        
        # Create 2D positional encoding
        pos_encoding = self._get_2d_positional_encoding(height, width, channels)
        self.register_buffer('pos_encoding', pos_encoding)
    
    def _get_2d_positional_encoding(self, h, w, d_model):
        pe = torch.zeros(d_model, h, w, dtype=torch.float32)

        # Create position indices (float tensors)
        y_pos = torch.arange(0, h, dtype=torch.float32).unsqueeze(1).repeat(1, w)
        x_pos = torch.arange(0, w, dtype=torch.float32).unsqueeze(0).repeat(h, 1)

        # Calculate division term for pairs (same length as d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             -(np.log(10000.0) / d_model))

        # Fill channels in groups of 4: [sin(y), cos(y), sin(x), cos(x)]
        for i in range(0, d_model, 4):
            k = i // 2
            # sin(y)
            if i < d_model:
                pe[i, :, :] = torch.sin(y_pos * div_term[k])
            # cos(y)
            if i + 1 < d_model:
                pe[i + 1, :, :] = torch.cos(y_pos * div_term[k])
            # sin(x)
            if i + 2 < d_model:
                pe[i + 2, :, :] = torch.sin(x_pos * div_term[k])
            # cos(x)
            if i + 3 < d_model:
                pe[i + 3, :, :] = torch.cos(x_pos * div_term[k])

        return pe
    
    def forward(self, x):
        _, _, h, w = x.shape
        if h != self.height or w != self.width:
            pe = self.pos_encoding.unsqueeze(0)[None] if False else self.pos_encoding.unsqueeze(0)
            pe = F.interpolate(pe, size=(h, w), mode='bilinear', align_corners=False)
            pe = pe.squeeze(0)
            return x + pe
        return x + self.pos_encoding.unsqueeze(0)

class TemporalEncoding(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.year_embed = nn.Embedding(50, embed_dim // 4)  # 2000-2050
        self.month_embed = nn.Embedding(12, embed_dim // 4)
        self.day_embed = nn.Embedding(31, embed_dim // 4)
        self.hour_embed = nn.Embedding(24, embed_dim // 4)
        
        self.projection = nn.Linear(embed_dim, embed_dim)
    
    def parse_timestamp(self, timestamp_str):
        timestamp_str = timestamp_str.split('.')[0]
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H_%M_%S")
        
        return (
            dt.year - 2000,  
            dt.month - 1,     
            dt.day - 1,       
            dt.hour
        )

    
    def forward(self, timestamps):
        if isinstance(timestamps, (list, tuple)) and isinstance(timestamps[0], str):
            batch_size = len(timestamps)
            years, months, days, hours = [], [], [], []
            
            for ts in timestamps:
                y, m, d, h = self.parse_timestamp(ts)
                years.append(y)
                months.append(m)
                days.append(d)
                hours.append(h)
            
            years = torch.tensor(years, dtype=torch.long)
            months = torch.tensor(months, dtype=torch.long)
            days = torch.tensor(days, dtype=torch.long)
            hours = torch.tensor(hours, dtype=torch.long)
        else:
            years = timestamps[:, 0].long()
            months = timestamps[:, 1].long()
            days = timestamps[:, 2].long()
            hours = timestamps[:, 3].long()

        device = self.year_embed.weight.device
        years = years.to(device)
        months = months.to(device)
        days = days.to(device)
        hours = hours.to(device)
        
        year_emb = self.year_embed(years)
        month_emb = self.month_embed(months)
        day_emb = self.day_embed(days)
        hour_emb = self.hour_embed(hours)
        
        # Concatenate all components
        temporal_emb = torch.cat([year_emb, month_emb, day_emb, hour_emb], dim=-1)
        
        return self.projection(temporal_emb)

class StormEncoding(nn.Module):
    def __init__(self, max_storms=1000, embed_dim=128):
        super().__init__()
        self.storm_embed = nn.Embedding(max_storms, embed_dim)
        self.storm_name_to_idx = {}
        self.next_idx = 0
    
    def get_storm_idx(self, storm_name):
        if storm_name not in self.storm_name_to_idx:
            self.storm_name_to_idx[storm_name] = self.next_idx
            self.next_idx += 1
        return self.storm_name_to_idx[storm_name]
    
    def forward(self, storm_names):
        if isinstance(storm_names, (list, tuple)) and isinstance(storm_names[0], str):
            indices = [self.get_storm_idx(name) for name in storm_names]
            indices = torch.tensor(indices, dtype=torch.long)
        else:
            indices = storm_names.long()
        
        device = self.storm_embed.weight.device
        indices = indices.to(device)
        
        return self.storm_embed(indices)

class ConditionEncoder(nn.Module):
    def __init__(self, 
                 gridsat_channels=1,
                 era5_channels=4,
                 img_size=64,
                 embed_dim=128,
                 output_channels=64):
        super().__init__()
        self.output_channels = output_channels
        # Spatial positional encoding for images
        self.spatial_pos_gridsat = SpatialPositionalEncoding(
            gridsat_channels, img_size, img_size
        )
        self.spatial_pos_era5 = SpatialPositionalEncoding(
            era5_channels, img_size, img_size
        )
        
        # Temporal and storm encodings
        self.temporal_encoder = TemporalEncoding(embed_dim)
        self.storm_encoder = StormEncoding(embed_dim=embed_dim)
        
        # Combine temporal and storm embeddings
        self.context_projection = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Process concatenated images with positional encoding
        total_input_channels = gridsat_channels + era5_channels
        self.image_encoder = nn.Sequential(
            nn.Conv2d(total_input_channels, output_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        )
        self.context_to_spatial = nn.Sequential(
            nn.Linear(embed_dim, output_channels),
            nn.SiLU()
        )
        
    def forward(self, gridsat, era5, timestamps, storm_names):
        batch_size = gridsat.shape[0]
        img_size = gridsat.shape[-1]
        
        # Add spatial positional encoding to images
        gridsat_pos = self.spatial_pos_gridsat(gridsat)
        era5_pos = self.spatial_pos_era5(era5)
        
        # Concatenate and encode images
        combined_images = torch.cat([gridsat_pos, era5_pos], dim=1)
        image_features = self.image_encoder(combined_images)
        
        # Encode temporal and storm information
        temporal_emb = self.temporal_encoder(timestamps)
        storm_emb = self.storm_encoder(storm_names)
        
        # Combine temporal and storm context
        context_emb = torch.cat([temporal_emb, storm_emb], dim=-1)
        context_emb = self.context_projection(context_emb)
        
        context_spatial = self.context_to_spatial(context_emb)  # (B, output_channels)
        context_spatial = context_spatial.view(batch_size, -1, 1, 1)
        context_spatial = F.interpolate(context_spatial, size=(img_size, img_size), mode='bilinear', align_corners=False)
        
        encoded_condition = image_features + context_spatial
        
        return encoded_condition, context_emb

# ============================================================================
# DIFFUSION MODEL AND SCHEDULER
# ============================================================================

class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        
    def add_noise(self, original, noise, t):
        original_shape = original.shape
        batch_size = original_shape[0]
        
        sqrt_alpha_cumprod = self.sqrt_alpha_cumprod[t].reshape(batch_size)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod[t].reshape(batch_size)
        
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)
            
        return sqrt_alpha_cumprod * original + sqrt_one_minus_alpha_cumprod * noise
    
    def sample_prev_timestep(self, xt, noise_pred, t):
        x0 = (xt - (self.sqrt_one_minus_alpha_cumprod[t] * noise_pred)) / self.sqrt_alpha_cumprod[t]
        x0 = torch.clamp(x0, -1.0, 1.0)
        
        mean = xt - ((self.betas[t] * noise_pred) / (self.sqrt_one_minus_alpha_cumprod[t]))
        mean = mean / torch.sqrt(self.alphas[t])
        
        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alpha_cumprod[t - 1]) / (1 - self.alpha_cumprod[t])
            variance = variance * self.betas[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0

def get_time_embeddings(time_steps, t_emb_dim):
    factor = 10000 ** (
        torch.arange(start=0, end=t_emb_dim // 2, device=time_steps.device) / (t_emb_dim // 2)
    )
    t_emb = time_steps[:, None].repeat(1, t_emb_dim // 2) / factor
    t_emb = torch.cat((torch.sin(t_emb), torch.cos(t_emb)), dim=-1)
    return t_emb

class VAEEncoder(nn.Module):
    """Encodes images to latent space (mu, logvar)"""
    def __init__(self, in_channels=1, latent_dim=64, base_channels=64):
        super().__init__()
        # 256 -> 128 -> 64 -> 32
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
                    
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),

            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
        )
        
        self.fc_mu = nn.Conv2d(base_channels * 4, latent_dim, 1)
        self.fc_logvar = nn.Conv2d(base_channels * 4, latent_dim, 1)
        
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=64, out_channels=1, base_channels=64):
        super().__init__()
        # Designed to invert the encoder above: latent 32x32 -> 64 -> 128 -> 256
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, base_channels * 4, 1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
            
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
            
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            
            nn.ConvTranspose2d(base_channels, base_channels, 4, 2, 1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),

            nn.Conv2d(base_channels, out_channels, 3, 1, 1),
        )
        
    def forward(self, z):
        return self.decoder(z)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample, num_heads):
        super().__init__()
        self.down_sample = down_sample
        self.resnet_conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )
        self.resnet_conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if down_sample else nn.Identity()
        
    def forward(self, x, t_emb):
        out = x
        resnet_input = out
        
        out = self.resnet_conv1(out)
        out = out + self.t_emb_layers(t_emb)[:, :, None, None]
        out = self.resnet_conv2(out)
        out = out + self.residual_input_conv(resnet_input)
        
        def _fn(out, t_emb):
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norm(in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attention(in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out_local = out + out_attn
            skip_local = out_local
            out_local = self.down_sample_conv(out_local)
            return out_local, skip_local

        # Use checkpointing to reduce memory during training
        if self.training:
            out, skip = checkpoint(_fn, out, t_emb, use_reentrant=False)
        else:
            out, skip = _fn(out, t_emb)

        return out, skip

class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads):
        super().__init__()
        self.resnet_conv1 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        ])
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            ),
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
        ])
        self.resnet_conv2 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        ])
        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        ])
        
    def forward(self, x, t_emb):
        out = x

        def _fn(out, t_emb):
            # First ResNet block
            resnet_input = out
            out_local = self.resnet_conv1[0](out_local := out)
            out_local = out_local + self.t_emb_layers[0](t_emb)[:, :, None, None]
            out_local = self.resnet_conv2[0](out_local)
            out_local = out_local + self.residual_input_conv[0](resnet_input)

            # Attention
            batch_size, channels, h, w = out_local.shape
            in_attn = out_local.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norm(in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attention(in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out_local = out_local + out_attn

            # Second ResNet block
            resnet_input = out_local
            out_local = self.resnet_conv1[1](out_local)
            out_local = out_local + self.t_emb_layers[1](t_emb)[:, :, None, None]
            out_local = self.resnet_conv2[1](out_local)
            out_local = out_local + self.residual_input_conv[1](resnet_input)

            return out_local

        # Use checkpointing to reduce memory during training
        if self.training:
            out = checkpoint(_fn, out, t_emb, use_reentrant=False)
        else:
            out = _fn(out, t_emb)

        return out

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample, num_heads, skip_channels=None):
        super().__init__()
        self.up_sample = up_sample
        self.skip_channels = skip_channels
        
        actual_in_channels = in_channels + (skip_channels if skip_channels else 0)
        
        self.resnet_conv1 = nn.Sequential(
            nn.GroupNorm(8, actual_in_channels),
            nn.SiLU(),
            nn.Conv2d(actual_in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )
        self.resnet_conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv = nn.Conv2d(actual_in_channels, out_channels, kernel_size=1)
        
    def forward(self, x, t_emb, skip=None):
        if self.up_sample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        
        out = x
        resnet_input = out

        def _fn(out, t_emb):
            out_local = out
            out_local = self.resnet_conv1(out_local)
            out_local = out_local + self.t_emb_layers(t_emb)[:, :, None, None]
            out_local = self.resnet_conv2(out_local)
            out_local = out_local + self.residual_input_conv(resnet_input)

            batch_size, channels, h, w = out_local.shape
            in_attn = out_local.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norm(in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attention(in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out_local = out_local + out_attn

            return out_local

        # Use checkpointing to reduce memory during training
        if self.training:
            out = checkpoint(_fn, out, t_emb, use_reentrant=False)
        else:
            out = _fn(out, t_emb)

        return out

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, channel_mults=(1, 2, 4, 8),
                 t_emb_dim=128, num_heads=4, gridsat_channels=1, era5_channels=4, img_size=64,
                 condition_embed_dim=128):
        super().__init__()
        
        self.t_emb_dim = t_emb_dim
        self.condition_embed_dim = condition_embed_dim
        
        # Condition encoder for ERA5, GRIDSAT, timestamps, and storm names
        self.condition_encoder = ConditionEncoder(
            gridsat_channels=gridsat_channels,
            era5_channels=era5_channels,
            img_size=img_size,
            embed_dim=condition_embed_dim,
            output_channels=base_channels
        )
        
        # Time embedding layers (for diffusion timestep)
        self.time_embed = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 4, t_emb_dim * 4)
        )
        
        # Combine diffusion time embedding with condition context embedding
        self.combined_embed = nn.Sequential(
            nn.Linear(t_emb_dim * 4 + condition_embed_dim, t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 4, t_emb_dim * 4)
        )
        
        self.init_conv = nn.Conv2d(base_channels + in_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder
        self.encoders = nn.ModuleList()
        curr_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels_enc = base_channels * mult
            self.encoders.append(
                DownBlock(curr_channels, out_channels_enc, t_emb_dim * 4, 
                             down_sample=(i < len(channel_mults) - 1), num_heads=num_heads)
            )
            curr_channels = out_channels_enc
        
        # Middle
        self.mid_block = MidBlock(curr_channels, curr_channels, t_emb_dim * 4, num_heads)
        
        # Decoder
        self.decoders = nn.ModuleList()
        reversed_mults = list(reversed(channel_mults))[1:]  # Skip bottleneck level
        
        for decoder_idx, mult in enumerate(reversed_mults):
            out_channels_dec = base_channels * mult
            skip_channels = out_channels_dec  # Skip connection has same channels as output
                
            self.decoders.append(
                UpBlock(curr_channels, out_channels_dec, t_emb_dim * 4,
                           up_sample=True, num_heads=num_heads, skip_channels=skip_channels)
            )
            curr_channels = out_channels_dec
            
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, noisy_target, gridsat, era5, timestamps, storm_names, t,
                encoded_condition=None, context_emb=None):
        if encoded_condition is None or context_emb is None:
            encoded_condition, context_emb = self.condition_encoder(
                gridsat, era5, timestamps, storm_names
            )
        x = torch.cat([encoded_condition, noisy_target], dim=1)

        # Get diffusion time embedding
        t_emb = get_time_embeddings(t, self.t_emb_dim)
        t_emb = self.time_embed(t_emb)

        # Combine diffusion time embedding with condition context
        combined_emb = torch.cat([t_emb, context_emb], dim=-1)
        combined_emb = self.combined_embed(combined_emb)

        # Initial convolution
        x = self.init_conv(x)

        # Encoder
        skips = []
        for i, encoder in enumerate(self.encoders):
            x, skip = encoder(x, combined_emb)
            skips.append(skip)

        # Middle
        x = self.mid_block(x, combined_emb)

        # Decoder with skip connections
        for i, decoder in enumerate(self.decoders):
            skip_idx = len(skips) - 2 - i  # -2 because we skip the bottleneck level
            skip = skips[skip_idx] if skip_idx >= 0 else None
            x = decoder(x, combined_emb, skip)

        # Final convolution
        x = self.final_conv(x)

        return x
# ============================================================================
# LPIPS Loss (Learned Perceptual Image Patch Similarity)
# ============================================================================

class LPIPSLoss(nn.Module):
    def __init__(self, use_gpu=True):
        super().__init__()
        
        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True).features
        
        # Extract feature layers
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(4)])   # relu1_2
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(4, 9)])  # relu2_2
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(9, 16)]) # relu3_3
        self.slice4 = nn.Sequential(*[vgg[i] for i in range(16, 23)]) # relu4_3
        self.slice5 = nn.Sequential(*[vgg[i] for i in range(23, 30)]) # relu5_3
        
        # Freeze VGG parameters (keep LPIPS weighting layers trainable)
        for param in vgg.parameters():
            param.requires_grad = False
            
        # Normalization for ImageNet pretrained model
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Linear layers for weighting features
        self.lins = nn.ModuleList([
            nn.Conv2d(64, 1, 1, bias=False),
            nn.Conv2d(128, 1, 1, bias=False),
            nn.Conv2d(256, 1, 1, bias=False),
            nn.Conv2d(512, 1, 1, bias=False),
            nn.Conv2d(512, 1, 1, bias=False),
        ])
        
        # Initialize weights
        for lin in self.lins:
            nn.init.constant_(lin.weight, 1.0)
            
    def normalize_tensor(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # Ensure mean/std are on the same device and dtype as input
        mean = self.mean.to(x.device).to(x.dtype)
        std = self.std.to(x.device).to(x.dtype)
        return (x - mean) / std
    
    def extract_features(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
    
    def forward(self, pred, target):
        # Normalize inputs
        pred = self.normalize_tensor(pred)
        target = self.normalize_tensor(target)
        
        # Extract features
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        # Compute weighted feature differences
        loss = 0
        for pred_feat, target_feat, lin in zip(pred_features, target_features, self.lins):
            # Normalize features
            pred_feat_norm = pred_feat / (pred_feat.norm(dim=1, keepdim=True) + 1e-10)
            target_feat_norm = target_feat / (target_feat.norm(dim=1, keepdim=True) + 1e-10)
            
            # Compute difference and weight
            diff = (pred_feat_norm - target_feat_norm) ** 2
            weighted_diff = lin(diff)
            
            # Spatial average
            loss += weighted_diff.mean()
            
        return loss
# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class DiffusionLoss(nn.Module):
    """
    Extended version of your original DiffusionLoss with VAE + LPIPS support.
    """
    def __init__(self, loss_type='mse', use_l1=False, l1_weight=0.1, 
                 use_lpips=False, lpips_weight=0.1, 
                 use_vae=False, kl_weight=1e-6):
        super().__init__()
        self.loss_type = loss_type
        self.use_l1 = use_l1
        self.l1_weight = l1_weight
        self.use_lpips = use_lpips
        self.lpips_weight = lpips_weight
        self.use_vae = use_vae
        self.kl_weight = kl_weight
        
        if loss_type == 'mse':
            self.base_loss = nn.MSELoss()
        elif loss_type == 'l1':
            self.base_loss = nn.L1Loss()
        elif loss_type == 'smooth_l1':
            self.base_loss = nn.SmoothL1Loss()
        elif loss_type == 'huber':
            self.base_loss = nn.HuberLoss(delta=1.0)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        if self.use_lpips:
            self.lpips = LPIPSLoss()
    
    def kl_divergence(self, mu, logvar):
        raw = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
        # number of latent elements per sample
        n_latents = mu[0].numel()
        mean_per_latent = raw / float(n_latents)
        return raw, mean_per_latent
    
    def forward(self, pred_noise, true_noise, pred_img=None, target_img=None, mu=None, logvar=None):
        losses = {}

        loss = self.base_loss(pred_noise, true_noise)
        losses['base'] = loss
        
        if self.use_l1 and self.loss_type == 'mse':
            l1_loss = F.l1_loss(pred_noise, true_noise)
            loss = loss + self.l1_weight * l1_loss
            losses['l1'] = l1_loss
        
        if self.use_lpips and pred_img is not None and target_img is not None:
            lpips_loss = self.lpips(pred_img, target_img)
            loss = loss + self.lpips_weight * lpips_loss
            losses['lpips'] = lpips_loss
        
        if self.use_vae and mu is not None and logvar is not None:
            kl_raw, kl_per_latent = self.kl_divergence(mu, logvar)
            loss = loss + self.kl_weight * kl_raw
            # expose both raw and normalized KL values for logging
            losses['kl_raw'] = kl_raw
            losses['kl_per_latent'] = kl_per_latent
        
        losses['total'] = loss
        
        if not self.use_lpips and not self.use_vae:
            return loss
        else:
            return losses

# ============================================================================
# EVALUATION AND METRICS
# ============================================================================

class MetricsCalculator:
    @staticmethod
    def mae(pred, target):
        return torch.mean(torch.abs(pred - target)).item()
    
    @staticmethod
    def mse(pred, target):
        return F.mse_loss(pred, target).item()
    
    @staticmethod
    def rmse(pred, target):
        return torch.sqrt(F.mse_loss(pred, target)).item()
    
    @staticmethod
    def psnr(pred, target, max_val=2.0):
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return 100.0
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
        return psnr.item()
    
    @staticmethod
    def ssim(pred, target, window_size=11, size_average=True):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        sigma = 1.5
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / (2 * sigma**2)) 
                             for x in range(window_size)])
        gauss = gauss / gauss.sum()
        
        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)
        window = window.to(pred.device)
        
        channel = pred.size(1)
        window = window.expand(channel, 1, window_size, window_size).contiguous()
        
        mu1 = F.conv2d(pred, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(target, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean().item()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

def evaluate_model(model, dataloader, scheduler, device, 
                   num_samples=100, vae_decoder=None, condition_downsample=None):
    model.eval()
    if vae_decoder is not None:
        vae_decoder.eval()
    
    metrics = {
        'mae': [],
        'mse': [],
        'rmse': [],
        'psnr': [],
        'ssim': []
    }
    
    calc = MetricsCalculator()
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i * dataloader.batch_size >= num_samples:
                break
            
            gridsat = batch['gridsat'].to(device)
            era5 = batch['era5'].to(device)
            target = batch['target'].to(device)
            timestamps = batch['timestamp']
            storm_names = batch['storm_name']
            
            # Compute encoded condition (no gradients needed) and downsample
            # to latent spatial size when requested.
            with torch.no_grad():
                encoded_cond, context_emb = model.condition_encoder(
                    gridsat, era5, timestamps, storm_names
                )
                if condition_downsample is not None:
                    encoded_cond = condition_downsample(encoded_cond)
            
            batch_size = target.shape[0]
        
            if vae_decoder is not None:
                latent_spatial_size = target.shape[-1] // 8  # e.g., 256 -> 32
                # UNet's in_channels includes both noise and condition channels
                latent_channels = model.init_conv.in_channels - model.condition_encoder.output_channels
                
                # Start from random latent noise
                generated = torch.randn(
                    batch_size, latent_channels, 
                    latent_spatial_size, latent_spatial_size, 
                    device=device
                )
            else:
                generated = torch.randn_like(target).to(device)
            
            for t in reversed(range(scheduler.num_timesteps)):
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                noise_pred = model(generated, gridsat, era5, timestamps, storm_names, t_batch,
                                   encoded_condition=encoded_cond, context_emb=context_emb)
                generated, _ = scheduler.sample_prev_timestep(generated, noise_pred, t)
            
            if vae_decoder is not None:
                generated = vae_decoder(generated)
            
            # Calculate metrics (same as before, now on final images)
            metrics['mae'].append(calc.mae(generated, target))
            metrics['mse'].append(calc.mse(generated, target))
            metrics['rmse'].append(calc.rmse(generated, target))
            metrics['psnr'].append(calc.psnr(generated, target))
            metrics['ssim'].append(calc.ssim(generated, target))
    
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return avg_metrics



def save_comparison_images(model, dataloader, scheduler, 
                          device, save_dir, num_samples=8, vae_decoder=None, condition_downsample=None):
    model.eval()
    if vae_decoder is not None:
        vae_decoder.eval()
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        batch = next(iter(dataloader))
        
        gridsat = batch['gridsat'][:num_samples].to(device)
        era5 = batch['era5'][:num_samples].to(device)
        target = batch['target'][:num_samples].to(device)
        timestamps = batch['timestamp'][:num_samples]
        storm_names = batch['storm_name'][:num_samples]
        
        # Compute encoded condition (no gradients needed) and downsample
        # to latent spatial size when requested.
        with torch.no_grad():
            encoded_cond, context_emb = model.condition_encoder(
                gridsat, era5, timestamps, storm_names
            )
            if condition_downsample is not None:
                encoded_cond = condition_downsample(encoded_cond)
        
        batch_size = target.shape[0]
        
        if vae_decoder is not None:
            latent_spatial_size = target.shape[-1] // 8  # e.g., 256 -> 32
            # UNet's in_channels includes both noise and condition channels
            latent_channels = model.init_conv.in_channels - model.condition_encoder.output_channels
            
            generated = torch.randn(
                batch_size, latent_channels,
                latent_spatial_size, latent_spatial_size,
                device=device
            )
        else:
            generated = torch.randn_like(target).to(device)
        
        for t in reversed(range(scheduler.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            noise_pred = model(generated, gridsat, era5, timestamps, storm_names, t_batch,
                               encoded_condition=encoded_cond, context_emb=context_emb)
            generated, _ = scheduler.sample_prev_timestep(generated, noise_pred, t)
        
        if vae_decoder is not None:
            generated = vae_decoder(generated)
        
        # Convert to numpy (same as before)
        gridsat_np = gridsat.cpu().numpy()
        target_np = target.cpu().numpy()
        generated_np = generated.cpu().numpy()

        # Plotting (exact same as before)
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        
        for i in range(num_samples):
            axes[i, 0].imshow(gridsat_np[i, 0], cmap='gray', vmin=-1, vmax=1)
            axes[i, 0].set_title(f'Input t\n{storm_names[i]}\n{timestamps[i]}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(target_np[i, 0], cmap='gray', vmin=-1, vmax=1)
            axes[i, 1].set_title(f'Ground Truth t+1')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(generated_np[i, 0], cmap='gray', vmin=-1, vmax=1)
            axes[i, 2].set_title(f'Predicted t+1')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison images to {save_dir / 'comparison.png'}")


def plot_training_curves(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_losses, label='Training Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves to {save_path}")

# ============================================================================
# TRAINING LOOP
# ============================================================================

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Create directories
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metrics directory
        self.metrics_dir = Path(config['output_dir']) / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics tracking
        self.metrics_history = []
        
        self.use_vae = config.get('use_vae', False)
        if self.use_vae:
            
            self.vae_encoder = VAEEncoder(
                in_channels=1,
                latent_dim=config.get('latent_dim', 64),
                base_channels=config.get('vae_base_channels', 64)
            ).to(self.device)
            
            self.vae_decoder = VAEDecoder(
                latent_dim=config.get('latent_dim', 64),
                out_channels=1,
                base_channels=config.get('vae_base_channels', 64)
            ).to(self.device)
        
        # UNet channels setup
        if self.use_vae:
            unet_in_channels = config.get('latent_dim', 64)
            unet_out_channels = config.get('latent_dim', 64)
            unet_img_size = config.get('latent_spatial_size', config['img_size'] // 8)
        else:
            # Original pixel space dimensions
            unet_in_channels = 1
            unet_out_channels = 1
            unet_img_size = config['img_size']
        
        self.unet = ConditionalUNet(
            in_channels=unet_in_channels,
            out_channels=unet_out_channels,
            base_channels=config['base_channels'],
            channel_mults=config['channel_mults'],
            t_emb_dim=config['t_emb_dim'],
            num_heads=config['num_heads'],
            gridsat_channels=1,
            era5_channels=4,
            img_size=config['img_size'],
            condition_embed_dim=config['embed_dim']
        ).to(self.device)
        
        # Noise scheduler
        self.scheduler = LinearNoiseScheduler(
            num_timesteps=config['num_timesteps'],
            beta_start=config['beta_start'],
            beta_end=config['beta_end']
        )
        
        # Move scheduler tensors to device
        self.scheduler.betas = self.scheduler.betas.to(self.device)
        self.scheduler.alphas = self.scheduler.alphas.to(self.device)
        self.scheduler.alpha_cumprod = self.scheduler.alpha_cumprod.to(self.device)
        self.scheduler.sqrt_alpha_cumprod = self.scheduler.sqrt_alpha_cumprod.to(self.device)
        self.scheduler.sqrt_one_minus_alpha_cumprod = self.scheduler.sqrt_one_minus_alpha_cumprod.to(self.device)
        
        self.criterion = DiffusionLoss(
            loss_type=config['loss_type'],
            use_l1=config.get('use_l1', False),
            l1_weight=config.get('l1_weight', 0.1),
            use_lpips=config.get('use_lpips', False),
            lpips_weight=config.get('lpips_weight', 0.1),
            use_vae=self.use_vae,
            kl_weight=config.get('kl_weight', 1e-6)
        )
        self.criterion = self.criterion.to(self.device)
        if self.use_vae:

            self.condition_downsample = nn.Sequential(
                nn.Conv2d(config['base_channels'], config['base_channels'],
                    kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, config['base_channels']),
                nn.SiLU(),
                nn.Conv2d(config['base_channels'], config['base_channels'],
                    kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, config['base_channels']),
                nn.SiLU(),
                nn.Conv2d(config['base_channels'], config['base_channels'],
                    kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, config['base_channels']),
                nn.SiLU()
            ).to(self.device)
        else:
            self.condition_downsample = nn.Identity().to(self.device)
            
        # Collect parameters for optimization
        params = list(self.unet.parameters()) + list(self.condition_downsample.parameters())

        if self.use_vae:
            params += (list(self.vae_encoder.parameters()) + 
                      list(self.vae_decoder.parameters()))
        
        self.optimizer = AdamW(params, lr=config['learning_rate'], 
                              weight_decay=config['weight_decay'])
        
        # Learning rate scheduler
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=config['num_epochs'],
            eta_min=config['min_lr']
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, dataloader):
        self.unet.train()
        self.condition_downsample.train()
        if self.use_vae:
            self.vae_encoder.train()
            self.vae_decoder.train()
        
        epoch_loss = 0
        epoch_losses_dict = {} 
        num_batches = 0
        
        pbar = tqdm(dataloader, desc='Training')
        for batch in pbar:
            gridsat = batch['gridsat'].to(self.device)
            era5 = batch['era5'].to(self.device)
            target = batch['target'].to(self.device)
            timestamps = batch['timestamp']
            storm_names = batch['storm_name']
            
            batch_size = target.shape[0]
            
            with torch.no_grad():
                encoded_cond, context_emb = self.unet.condition_encoder(
                    gridsat, era5, timestamps, storm_names
                )
                if self.use_vae:
                    encoded_cond = self.condition_downsample(encoded_cond)

            encoded_cond = encoded_cond.detach()
            context_emb = context_emb.detach()
            
            if self.use_vae:
                mu, logvar = self.vae_encoder(target)
                latent_std = torch.exp(0.5 * logvar)
                print(f"  Latent stats: μ={mu.mean():.3f}±{mu.std():.3f}, σ={latent_std.mean():.3f}")
                z = self.vae_encoder.reparameterize(mu, logvar)
                diffusion_input = z  # Diffusion works on latent
            else:
                diffusion_input = target  # Diffusion works on pixels
                mu, logvar = None, None
            
            # Sample random timesteps
            t = torch.randint(0, self.scheduler.num_timesteps, 
                            (batch_size,), device=self.device)
            
            # Add noise
            noise = torch.randn_like(diffusion_input)
            noisy_input = self.scheduler.add_noise(diffusion_input, noise, t)
            
            noise_pred = self.unet(noisy_input, gridsat, era5, timestamps, storm_names, t,
                                   encoded_condition=encoded_cond, context_emb=context_emb)
            
            if self.use_vae and self.config.get('use_lpips', False):
                sqrt_alpha_cumprod_t = self.scheduler.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_cumprod_t = self.scheduler.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
                pred_z = (noisy_input - sqrt_one_minus_alpha_cumprod_t * noise_pred) / sqrt_alpha_cumprod_t
                pred_img = self.vae_decoder(pred_z)
            else:
                pred_img = None
            

            loss_output = self.criterion(
                noise_pred, noise, 
                pred_img=pred_img, 
                target_img=target if self.use_vae else None,
                mu=mu, 
                logvar=logvar
            )
            
            if isinstance(loss_output, dict):
                loss = loss_output['total']
                for key, value in loss_output.items():
                    if key not in epoch_losses_dict:
                        epoch_losses_dict[key] = 0
                    # accumulate scalar values; some items may be tensors
                    try:
                        epoch_losses_dict[key] += value.item()
                    except Exception:
                        # fallback: convert to float
                        epoch_losses_dict[key] += float(value)
            
            # Track mu/logvar stats for VAE
            if mu is not None and logvar is not None:
                mu_mean = mu.mean().item()
                logvar_mean = logvar.mean().item()
                if 'mu_mean' not in epoch_losses_dict:
                    epoch_losses_dict['mu_mean'] = 0.0
                if 'logvar_mean' not in epoch_losses_dict:
                    epoch_losses_dict['logvar_mean'] = 0.0
                epoch_losses_dict['mu_mean'] += mu_mean
                epoch_losses_dict['logvar_mean'] += logvar_mean
            else:
                loss = loss_output
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            all_params = list(self.unet.parameters()) + list(self.condition_downsample.parameters())
            if self.use_vae:
                all_params += list(self.vae_encoder.parameters()) + list(self.vae_decoder.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            
            self.optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / num_batches
        
        for key in epoch_losses_dict:
            epoch_losses_dict[key] /= num_batches
        
        return avg_loss, epoch_losses_dict
    
    @torch.no_grad()
    def validate(self, dataloader):
        self.unet.eval()
        if self.use_vae:
            self.vae_encoder.eval()
            self.vae_decoder.eval()
        
        epoch_loss = 0
        epoch_losses_dict = {}
        num_batches = 0
        
        for batch in tqdm(dataloader, desc='Validation'):
            gridsat = batch['gridsat'].to(self.device)
            era5 = batch['era5'].to(self.device)
            target = batch['target'].to(self.device)
            timestamps = batch['timestamp']
            storm_names = batch['storm_name']
            
            batch_size = target.shape[0]
            
            with torch.no_grad():
                encoded_cond, context_emb = self.unet.condition_encoder(
                    gridsat, era5, timestamps, storm_names
                )
                if self.use_vae:
                    encoded_cond = self.condition_downsample(encoded_cond)

            encoded_cond = encoded_cond.detach()
            context_emb = context_emb.detach()
            
            if self.use_vae:
                mu, logvar = self.vae_encoder(target)
                z = self.vae_encoder.reparameterize(mu, logvar)
                diffusion_input = z
            else:
                diffusion_input = target
                mu, logvar = None, None
            
            # Sample random timesteps
            t = torch.randint(0, self.scheduler.num_timesteps,
                            (batch_size,), device=self.device)
            
            # Add noise
            noise = torch.randn_like(diffusion_input)
            noisy_input = self.scheduler.add_noise(diffusion_input, noise, t)
            
            # Predict noise
            noise_pred = self.unet(noisy_input, gridsat, era5, timestamps, storm_names, t,
                                   encoded_condition=encoded_cond, context_emb=context_emb)
            
            if self.use_vae and self.config.get('use_lpips', False):
                sqrt_alpha_cumprod_t = self.scheduler.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_cumprod_t = self.scheduler.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
                pred_z = (noisy_input - sqrt_one_minus_alpha_cumprod_t * noise_pred) / sqrt_alpha_cumprod_t
                pred_img = self.vae_decoder(pred_z)
            else:
                pred_img = None
            
            # Calculate loss
            loss_output = self.criterion(
                noise_pred, noise,
                pred_img=pred_img,
                target_img=target if self.use_vae else None,
                mu=mu,
                logvar=logvar
            )
            
            # Handle both scalar and dict returns
            if isinstance(loss_output, dict):
                loss = loss_output['total']
                for key, value in loss_output.items():
                    if key not in epoch_losses_dict:
                        epoch_losses_dict[key] = 0
                    try:
                        epoch_losses_dict[key] += value.item()
                    except Exception:
                        epoch_losses_dict[key] += float(value)
            
            if mu is not None and logvar is not None:
                mu_mean = mu.mean().item()
                logvar_mean = logvar.mean().item()
                if 'mu_mean' not in epoch_losses_dict:
                    epoch_losses_dict['mu_mean'] = 0.0
                if 'logvar_mean' not in epoch_losses_dict:
                    epoch_losses_dict['logvar_mean'] = 0.0
                epoch_losses_dict['mu_mean'] += mu_mean
                epoch_losses_dict['logvar_mean'] += logvar_mean
            else:
                loss = loss_output
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Average individual losses
        for key in epoch_losses_dict:
            epoch_losses_dict[key] /= num_batches
        
        return avg_loss, epoch_losses_dict
    
    def save_checkpoint(self, epoch, train_loss, val_loss, metrics=None):
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
        
        checkpoint = {
            'epoch': epoch,
            'unet_state_dict': self.unet.state_dict(),
            'condition_downsample_state_dict': self.condition_downsample.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.use_vae:
            checkpoint['vae_encoder_state_dict'] = self.vae_encoder.state_dict()
            checkpoint['vae_decoder_state_dict'] = self.vae_decoder.state_dict()
        
        # Save metrics to history
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'is_best': is_best
        }
        if metrics:
            epoch_metrics.update(metrics)
        
        self.metrics_history.append(epoch_metrics)
        
        metrics_df = pd.DataFrame(self.metrics_history)
        metrics_df.to_csv(self.metrics_dir / 'training_metrics.csv', index=False)

        if is_best:
            best_path = self.checkpoint_dir / f'best_model_{epoch}.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        if 'condition_downsample_state_dict' in checkpoint:
            try:
                self.condition_downsample.load_state_dict(checkpoint['condition_downsample_state_dict'])
            except Exception:
                print('Warning: could not load condition_downsample_state_dict from checkpoint')
        
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        if self.use_vae and 'vae_encoder_state_dict' in checkpoint:
            self.vae_encoder.load_state_dict(checkpoint['vae_encoder_state_dict'])
            self.vae_decoder.load_state_dict(checkpoint['vae_decoder_state_dict'])
        
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']
    
    def train(self, train_loader, val_loader):
        start_epoch = 0
        
        if self.config.get('resume_checkpoint'):
            start_epoch = self.load_checkpoint(self.config['resume_checkpoint']) + 1
        
        print(f"Starting training from epoch {start_epoch}")
        
        for epoch in range(start_epoch, self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            # KL annealing: gradually increase kl_weight over first N epochs
            if self.use_vae:
                kl_anneal_epochs = self.config.get('kl_anneal_epochs', 10)
                target_kl_w = self.config.get('kl_weight', self.criterion.kl_weight)
                if epoch < kl_anneal_epochs:
                    current_kl_w = target_kl_w * float((epoch + 1) / max(1, kl_anneal_epochs))
                else:
                    current_kl_w = target_kl_w
                # update the criterion's kl weight used in loss computation
                self.criterion.kl_weight = current_kl_w
                print(f"KL weight (annealed): {current_kl_w:.6e}")
            
            # Train
            train_loss, train_losses_dict = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            print(f"Training Loss: {train_loss:.6f}")
            

            if train_losses_dict:
                print("Training Loss Components:")
                for key, value in train_losses_dict.items():
                    # scaled lpips
                    if key == 'lpips':
                        lpips_w = self.config.get('lpips_weight', self.criterion.lpips_weight if hasattr(self.criterion, 'lpips_weight') else 0.0)
                        scaled = lpips_w * value
                        print(f"  {key} (raw): {value:.6f}  scaled: {scaled:.6f}")
                    # kl entries
                    elif key == 'kl_raw':
                        scaled = self.criterion.kl_weight * value
                        kl_per = train_losses_dict.get('kl_per_latent', None)
                        if kl_per is not None:
                            print(f"  kl_raw: {value:.6f}  scaled: {scaled:.6e}  per_latent: {kl_per:.6f}")
                        else:
                            print(f"  kl_raw: {value:.6f}  scaled: {scaled:.6e}")
                    elif key == 'kl_per_latent':
                        # already printed above alongside kl_raw
                        continue
                    else:
                        print(f"  {key}: {value:.6f}")
            
            # Validate
            val_loss, val_losses_dict = self.validate(val_loader)
            self.val_losses.append(val_loss)
            print(f"Validation Loss: {val_loss:.6f}")
            
            if val_losses_dict:
                print("Validation Loss Components:")
                for key, value in val_losses_dict.items():
                    if key == 'lpips':
                        lpips_w = self.config.get('lpips_weight', self.criterion.lpips_weight if hasattr(self.criterion, 'lpips_weight') else 0.0)
                        scaled = lpips_w * value
                        print(f"  {key} (raw): {value:.6f}  scaled: {scaled:.6f}")
                    elif key == 'kl_raw':
                        scaled = self.criterion.kl_weight * value
                        kl_per = val_losses_dict.get('kl_per_latent', None)
                        if kl_per is not None:
                            print(f"  kl_raw: {value:.6f}  scaled: {scaled:.6e}  per_latent: {kl_per:.6f}")
                        else:
                            print(f"  kl_raw: {value:.6f}  scaled: {scaled:.6e}")
                    elif key == 'kl_per_latent':
                        continue
                    else:
                        print(f"  {key}: {value:.6f}")
            
            # Learning rate step
            self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {current_lr:.6e}")
            
            metrics = None
            if (epoch + 1) % self.config.get('eval_every', 5) == 0:
                metrics = evaluate_model(
                    self.unet,
                    val_loader,
                    self.scheduler,
                    self.device,
                    num_samples=self.config.get('eval_samples', 100),
                    vae_decoder=self.vae_decoder if self.use_vae else None,
                    condition_downsample=self.condition_downsample
                )
            
            # Combine loss components with metrics
            all_metrics = {}
            if train_losses_dict:
                for key, value in train_losses_dict.items():
                    all_metrics[f'train_{key}'] = value
            if val_losses_dict:
                for key, value in val_losses_dict.items():
                    all_metrics[f'val_{key}'] = value
            if metrics:
                all_metrics.update(metrics)
            
            # Save checkpoint and metrics
            self.save_checkpoint(epoch, train_loss, val_loss, all_metrics)
            
            if val_loss < self.best_val_loss:
                print(f"New best validation loss: {val_loss:.6f}")
            
            if epoch % 20 == 0:
                self.save_checkpoint(epoch, train_loss, val_loss, all_metrics)
            
            if epoch % 10 == 0:
                save_comparison_images(
                    self.unet,
                    val_loader,
                    self.scheduler,
                    self.device,
                    self.output_dir / f'epoch_{epoch + 1}',
                    num_samples=8,
                    vae_decoder=self.vae_decoder if self.use_vae else None,
                    condition_downsample=self.condition_downsample
                )
            
            # Plot training curves
            if (epoch + 1) % self.config['plot_every'] == 0:
                plot_training_curves(
                    self.train_losses,
                    self.val_losses,
                    self.output_dir / 'training_curves.png'
                )
        
        final_metrics = evaluate_model(
            self.unet,
            val_loader,
            self.scheduler,
            self.device,
            num_samples=self.config.get('eval_samples', 100),
            vae_decoder=self.vae_decoder if self.use_vae else None,
            condition_downsample=self.condition_downsample
        )
        
        print("\nFinal Metrics:")
        for metric, value in final_metrics.items():
            if value is not None:
                print(f"{metric.upper()}: {value:.6f}")
        
        # Save final metrics
        with open(self.output_dir / 'final_metrics.json', 'w') as f:
            json.dump(final_metrics, f, indent=4)
        
        # Save final state of metrics history
        metrics_df = pd.DataFrame(self.metrics_history)
        metrics_df.to_csv(self.metrics_dir / 'final_training_metrics.csv', index=False)
        
        print(f"\nTraining complete! Results saved to {self.output_dir}")

def main():
    # Configuration
    config = {
        # Data
        'root_dir': r'/kaggle/input/setcd-dataset',
        'train_years': ['2005_0', '2016_0', '2022_0'],
        'test_storm': ['2022349N13068'],  # Keep test set as-is
        'num_val_storms': 5,  # Number of storms for validation
        'random_seed': 42,  # For reproducible splits
        'min_timesteps': 5,  # Minimum timesteps per storm
        'val_years': ['2022_0'],  # Separate validation years (unused if random split)
        'batch_size': 8,
        'num_workers': 4,
        'img_size': 256,  # Full resolution
        'latent_spatial_size': 32,  # 256/8 = 32

    'base_channels': 128,
        'channel_mults': (1, 2, 4, 8),
        'num_heads': 4,
        'embed_dim': 128,
        'condition_channels': 64,
        't_emb_dim': 128,

        # Diffusion
        'num_timesteps': 1000,
    'beta_start': 8.5e-4,
    'beta_end': 0.012,
        'use_vae': True,
        'latent_dim': 64,
        'vae_base_channels': 64,

        # Losses
        'use_lpips': True,
        'lpips_weight': 0.005,
        'kl_weight': 5e-6,
        'kl_anneal_epochs': 10,

        # Training
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'min_lr': 1e-6,
        'weight_decay': 1e-4,
        'loss_type': 'mse',
        'use_l1': False,
        'l1_weight': 0.1,
        # Eval and Checkpoints
        'eval_every': 10,
        'eval_samples': 50,
        'save_every': 10,
        'sample_every': 10,
        'plot_every': 5,
        'checkpoint_dir': './checkpoints',
        'output_dir': './outputs',
        'resume_checkpoint': None,
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("Configuration:")
    print(json.dumps(config, indent=4))

    # Create random storm-level split
    print("\n" + "="*80)
    print("Creating random storm-level split...")
    print("="*80)

    storm_split = create_random_storm_split(
        root_dir=config['root_dir'],
        years=config['train_years'],
        num_val_storms=config['num_val_storms'],
        test_storms=config['test_storm'],
        seed=config['random_seed'],
        min_timesteps=config['min_timesteps']
    )

    # Update config with selected storms
    config['val_storm'] = storm_split['val_storms']

    print("\n" + "="*80)
    print("Creating dataloaders...")
    print("="*80)

    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=config['root_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        train_years=config['train_years'],
        val_years=config.get('val_years', config['train_years']),  # Use separate val_years if specified
        test_years=config.get('test_years', config['train_years']),
        test_storm=config['test_storm'],
        val_storm=config['val_storm'],  # Now contains 5 random storms
        img_size=config['img_size'],  # Pass img_size to dataloader
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(config)
    
    # Start training
    print("\nStarting training...")
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()