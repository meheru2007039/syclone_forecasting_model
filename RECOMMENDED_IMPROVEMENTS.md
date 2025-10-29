# Recommended Improvements for Stable Diffusion Model

## Overview

This document provides optional code improvements to enhance model performance and training efficiency. The critical bug has been fixed in the main file. These are suggested enhancements based on Stable Diffusion best practices.

---

## 1. Increase LPIPS Weight (Recommended)

**Current:** `lpips_weight: 0.005`
**Recommended:** `lpips_weight: 0.05` to `0.1`

**Rationale:** LPIPS loss helps produce perceptually better images. Your current weight is 10-20x lower than typical SD implementations, which may result in blurry outputs.

**Change in `main()` config (line 1881):**
```python
# Current
'lpips_weight': 0.005,

# Recommended
'lpips_weight': 0.05,  # or 0.1 for stronger perceptual loss
```

---

## 2. Reduce Latent Dimension (Optional - Performance)

**Current:** `latent_dim: 64`
**Recommended:** `latent_dim: 16` or `8`

**Rationale:** Stable Diffusion uses 4-channel latents. Your 64 channels is 16x larger, which increases:
- Memory usage
- Training time
- Model size

Consider if this complexity is necessary for cyclone forecasting.

**Benefits of reduction:**
- ~4x faster training (if reduced to 16 channels)
- ~4x less memory usage
- Similar or better quality (less overfitting)

**Changes required:**
```python
# In main() config:
'latent_dim': 16,  # or 8
```

**Test both values and compare:**
- latent_dim=16: More expressive than SD but still efficient
- latent_dim=8: 2x SD, good balance
- latent_dim=4: Same as SD

---

## 3. Add Exponential Moving Average (EMA) for Weights

**Rationale:** SD uses EMA weights for inference, which provides more stable and higher-quality generations.

**Implementation:** Add to Trainer class:

```python
# In Trainer.__init__, after model initialization:
from copy import deepcopy

self.ema_unet = deepcopy(self.unet)
self.ema_unet.eval()
for param in self.ema_unet.parameters():
    param.requires_grad = False

if self.use_vae:
    self.ema_vae_decoder = deepcopy(self.vae_decoder)
    self.ema_vae_decoder.eval()
    for param in self.ema_vae_decoder.parameters():
        param.requires_grad = False

self.ema_decay = 0.9999

# Add update_ema method:
def update_ema(self):
    """Update EMA model weights"""
    with torch.no_grad():
        for ema_param, param in zip(self.ema_unet.parameters(), self.unet.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

        if self.use_vae:
            for ema_param, param in zip(self.ema_vae_decoder.parameters(),
                                        self.vae_decoder.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

# Call in train_epoch after optimizer.step():
self.update_ema()

# Use EMA models for evaluation:
# In evaluate_model and save_comparison_images, pass self.ema_unet instead of self.unet
```

---

## 4. Add Learning Rate Warmup

**Rationale:** Warmup helps stabilize training in early epochs.

**Implementation:** Replace CosineAnnealingLR with:

```python
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)

# In Trainer.__init__:
num_training_steps = config['num_epochs'] * len(train_loader)
num_warmup_steps = num_training_steps // 10  # 10% warmup

self.lr_scheduler = get_cosine_schedule_with_warmup(
    self.optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# In train(), call scheduler.step() after each batch, not each epoch:
# Move self.lr_scheduler.step() from after validate() to after optimizer.step()
```

---

## 5. Add Mixed Precision Training

**Rationale:** Speeds up training by ~2x with minimal quality loss.

**Implementation:**

```python
# In Trainer.__init__:
from torch.cuda.amp import autocast, GradScaler

self.scaler = GradScaler() if self.device.type == 'cuda' else None
self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'

# In train_epoch, wrap forward pass:
if self.use_amp:
    with autocast():
        noise_pred = self.unet(...)
        loss_output = self.criterion(...)
else:
    noise_pred = self.unet(...)
    loss_output = self.criterion(...)

# Replace backward pass:
self.optimizer.zero_grad()
if self.use_amp:
    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
    self.scaler.step(self.optimizer)
    self.scaler.update()
else:
    loss.backward()
    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
    self.optimizer.step()
```

---

## 6. Increase Weight Decay

**Current:** `weight_decay: 1e-4`
**Recommended:** `weight_decay: 1e-2`

**Rationale:** SD uses 0.01, which helps prevent overfitting.

**Change:**
```python
'weight_decay': 1e-2,
```

---

## 7. Add Gradient Accumulation

**Rationale:** Allows effective larger batch sizes when GPU memory is limited.

**Implementation:**

```python
# In config:
'accumulation_steps': 4,  # Effective batch_size = batch_size * accumulation_steps

# In train_epoch:
self.optimizer.zero_grad()

for accumulation_step, batch in enumerate(pbar):
    # ... forward pass and loss calculation ...

    loss = loss / self.config['accumulation_steps']  # Scale loss
    loss.backward()

    if (accumulation_step + 1) % self.config['accumulation_steps'] == 0:
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
```

---

## 8. Separate VAE Reconstruction Loss Monitoring

**Rationale:** Monitor VAE quality independently from diffusion.

**Implementation:**

```python
# In train_epoch, add periodic VAE reconstruction check:
if batch_idx % 100 == 0:  # Every 100 batches
    with torch.no_grad():
        mu, logvar = self.vae_encoder(target)
        z = self.vae_encoder.reparameterize(mu, logvar)
        recon = self.vae_decoder(z)
        recon_loss = F.mse_loss(recon, target)
        print(f"  VAE Reconstruction Loss: {recon_loss.item():.6f}")
```

---

## 9. Add Classifier-Free Guidance Support

**Rationale:** Improves conditioning control during inference.

**Implementation:**

```python
# In ConditionalUNet.forward, add unconditional path:
def forward(self, noisy_target, gridsat, era5, timestamps, storm_names, t,
            encoded_condition=None, context_emb=None, unconditional=False):

    if unconditional:
        # Return unconditional prediction (no conditioning)
        encoded_condition = torch.zeros_like(encoded_condition)
        context_emb = torch.zeros_like(context_emb)

    # ... rest of forward pass ...

# During inference, use guidance:
def sample_with_guidance(model, ..., guidance_scale=7.5):
    # Predict with conditioning
    noise_pred_cond = model(x, ..., unconditional=False)

    # Predict without conditioning
    noise_pred_uncond = model(x, ..., unconditional=True)

    # Combine with guidance
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

    return noise_pred
```

---

## 10. Monitor Latent Statistics

**Rationale:** Helps diagnose VAE training issues (posterior collapse, mode collapse).

**Implementation:**

```python
# In train_epoch, after VAE encoding:
if self.use_vae and batch_idx % 100 == 0:
    with torch.no_grad():
        latent_mean = z.mean().item()
        latent_std = z.std().item()
        mu_mean = mu.mean().item()
        logvar_mean = logvar.mean().item()

        print(f"  Latent stats: mean={latent_mean:.3f}, std={latent_std:.3f}")
        print(f"  VAE stats: μ={mu_mean:.3f}, log(σ²)={logvar_mean:.3f}")

        # Check for posterior collapse
        if latent_std < 0.1:
            print("  WARNING: Possible posterior collapse detected!")
```

---

## Priority Ranking

### Must Do (High Impact):
1. ✅ **Already Fixed:** condition_downsample bug
2. 🔴 **Increase lpips_weight** to 0.05-0.1 (easy, high impact)

### Should Do (Medium Impact):
3. 🟡 **Add EMA weights** (moderate complexity, high quality improvement)
4. 🟡 **Reduce latent_dim** to 16 (easy, significant speedup)
5. 🟡 **Add mixed precision training** (moderate complexity, 2x speedup)

### Nice to Have (Lower Impact):
6. 🟢 **Add learning rate warmup** (moderate complexity)
7. 🟢 **Increase weight_decay** to 1e-2 (easy)
8. 🟢 **Add gradient accumulation** (low complexity)
9. 🟢 **Monitor VAE reconstruction** (low complexity)
10. 🟢 **Add classifier-free guidance** (higher complexity, for inference only)

---

## Testing After Changes

After implementing any changes:

1. **Dimension Test:**
```python
# Run a single forward pass with dummy data
batch_size = 2
gridsat = torch.randn(batch_size, 1, 256, 256).cuda()
era5 = torch.randn(batch_size, 4, 256, 256).cuda()
target = torch.randn(batch_size, 1, 256, 256).cuda()
timestamps = ['2020-01-01 00_00_00'] * batch_size
storm_names = ['TEST'] * batch_size

# Test full forward pass
# Should run without errors
```

2. **Training Test:**
```python
# Train for 1 epoch with max_samples=10
# Verify no dimension errors
```

3. **Generation Test:**
```python
# Generate a few samples
# Verify output is (B, 1, 256, 256)
```

---

## Conclusion

These improvements are optional but recommended based on Stable Diffusion best practices. Start with the high-priority items (lpips_weight, EMA) for maximum benefit with minimal effort.

The critical bug has been fixed in the main notebook.py file, so your model should now train successfully without any changes to this file.

---

Generated by Claude Code - Optional Improvements Guide
