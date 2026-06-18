import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

_root = Path(__file__).parent

sys.path.insert(0, str(_root / 'guided-diffusion'))
from guided_diffusion.script_util import (create_model_and_diffusion, model_and_diffusion_defaults)

@torch.no_grad()
def topk_attended_positions_batch(dinov2, normalize_fn,  x_batch, k, min_dist=2):
    B = x_batch.shape[0]

    holder = []
    def hook(m, inp, out):
        holder.append(out)

    handle = dinov2.blocks[-1].attn.qkv.register_forward_hook(hook)

    _ = dinov2(normalize_fn(x_batch))

    handle.remove()

    qkv_out  = holder[0]
    H        = dinov2.blocks[-1].attn.num_heads
    head_dim = qkv_out.shape[2] // 3 // H

    qkv = qkv_out.reshape(B, qkv_out.shape[1], 3, H, head_dim)
    q   = qkv[:, :, 0].transpose(1, 2)
    k_  = qkv[:, :, 1].transpose(1, 2)

    attn = (q @ k_.transpose(-2, -1)) * (head_dim ** -0.5)
    attn = attn.softmax(dim=-1)
    attn_cls = attn[:, :, 0, 1:].mean(dim=1)

    indices = []
    for b in range(B):
        scores = attn_cls[b].clone()
        selected = []
        for _ in range(k):
            idx = scores.argmax().item()
            selected.append(idx)
            row, col = idx // 16, idx % 16
            for r in range(max(0, row - min_dist), min(16, row + min_dist + 1)):
                for c in range(max(0, col - min_dist), min(16, col + min_dist + 1)):
                    scores[r * 16 + c] = -1.0
        indices.append(selected)
    return indices

def extract_crops_batch(x, indices, crop_size):
    PATCH_PX = 14
    HALF     = crop_size // 2
    B        = x.shape[0]
    crops    = []
    for b in range(B):
        for idx in indices[b]:
            row, col = idx // 16, idx % 16
            cy = row * PATCH_PX + PATCH_PX // 2
            cx = col * PATCH_PX + PATCH_PX // 2
            y1 = max(0, min(cy - HALF, 224 - crop_size))
            x1 = max(0, min(cx - HALF, 224 - crop_size))
            crop = x[b:b+1, :, y1:y1+crop_size, x1:x1+crop_size]
            if crop_size != 224:
                crop = F.interpolate(crop, (224, 224), mode='bicubic', align_corners=False)
            crops.append(crop)
    return torch.cat(crops, dim=0)

class CropRSClassifier(nn.Module):
    def __init__(self, dino, normalizer, prototypes, sigma, m_per_crop, k_crops, crop_size, min_patch_dist=2):
        super().__init__()
        self.dino = dino
        self.normalizer = normalizer
        self.prototypes = prototypes  
        self.sigma      = sigma
        self.m_per_crop = m_per_crop
        self.k_crops    = k_crops
        self.crop_size  = crop_size
        self.min_patch_dist = min_patch_dist

    def classify_logits(self, x):
        feats = self.dino(self.normalizer(x))
        feats = F.normalize(feats, dim=-1)
        return feats @ self.prototypes.T * 100

    def forward(self, x):
        B = x.shape[0]
        with torch.no_grad():
            indices = topk_attended_positions_batch(self.dino, self.normalizer, x.detach(), self.k_crops, self.min_patch_dist)

        crops = extract_crops_batch(x, indices, self.crop_size)
        agg = torch.zeros(B * self.k_crops, self.prototypes.shape[0], device=x.device)

        for _ in range(self.m_per_crop):
            noise = torch.randn_like(crops) * self.sigma
            noisy = (crops + noise).clamp(0, 1)
            agg   = agg + self.classify_logits(noisy)

        agg = agg / self.m_per_crop
        agg = agg.view(B, self.k_crops, -1).mean(dim=1)
        return agg


def load_pixel_ddpm(ckpt_path, device):
    """Load OpenAI guided-diffusion 256x256 unconditional ImageNet model."""
    defaults = model_and_diffusion_defaults()
    defaults.update(dict(
        image_size=256, num_channels=256, num_res_blocks=2,
        attention_resolutions='32,16,8', class_cond=False,
        diffusion_steps=1000, learn_sigma=True,
        noise_schedule='linear', num_heads=4,
        num_head_channels=64, resblock_updown=True,
        use_fp16=False, use_scale_shift_norm=True,
    ))
    model, diffusion = create_model_and_diffusion(**defaults)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    return model.to(device).eval(), diffusion


class DiffusionRSClassifier(nn.Module):
    """
    For each crop:
      1. resize to DDPM size (256)
      2. forward noise to timestep t (small, e.g. t=50)
      3. one DDPM denoising step (back to t-1)
      4. resize back to 224
      5. add Gaussian noise sigma * eps (for RS)
      6. DINOv2 → classify
      7. repeat M times, average logits
    """

    def __init__(self, dino, normalizer, prototypes, sigma, m_per_crop,
                 k_crops, crop_size, min_patch_dist,
                 ddpm_model, diffusion, ddpm_timestep=50):
        super().__init__()
        self.dino           = dino
        self.normalizer     = normalizer
        self.prototypes     = prototypes
        self.sigma          = sigma
        self.m_per_crop     = m_per_crop
        self.k_crops        = k_crops
        self.crop_size      = crop_size
        self.min_patch_dist = min_patch_dist
        self.ddpm           = ddpm_model
        self.diffusion      = diffusion
        self.ddpm_timestep  = ddpm_timestep

    def classify_logits(self, x):
        feats = self.dino(self.normalizer(x))
        feats = F.normalize(feats, dim=-1)
        return feats @ self.prototypes.T * 100

    def one_step_denoise(self, x):
        """
        Add noise to timestep t, then take one DDPM reverse step.
        x: (B, 3, 224, 224) in [0,1]
        Returns: (B, 3, 224, 224) in [0,1]
        """
        B = x.shape[0]
        device = x.device

        # resize to DDPM training size 256
        x256 = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False)
        # [0,1] → [-1,1] (DDPM convention)
        x_ddpm = (x256 - 0.5) * 2.0

        # forward noise to timestep t
        t   = torch.tensor([self.ddpm_timestep] * B, device=device).long()
        eps = torch.randn_like(x_ddpm)
        x_t = self.diffusion.q_sample(x_ddpm, t, noise=eps)

        # one reverse step: t → t-1
        out = self.diffusion.p_mean_variance(self.ddpm, x_t, t,
                                              clip_denoised=True)
        # use the mean (deterministic part) as the denoised output
        x_clean_ddpm = out['mean']

        # [-1,1] → [0,1]
        x_clean = (x_clean_ddpm + 1.0) / 2.0
        x_clean = x_clean.clamp(0, 1)

        # resize back to 224
        x_clean = F.interpolate(x_clean, (224, 224),
                                mode='bicubic', align_corners=False)
        return x_clean

    def forward(self, x):
        B = x.shape[0]
        with torch.no_grad():
            indices = topk_attended_positions_batch(
                self.dino, self.normalizer, x.detach(),
                self.k_crops, self.min_patch_dist,
            )

        crops = extract_crops_batch(x, indices, self.crop_size)
        # crops: (B*k, 3, 224, 224)

        agg = torch.zeros(B * self.k_crops, self.prototypes.shape[0],
                          device=x.device)

        for _ in range(self.m_per_crop):
            # ── one DDPM step ──
            denoised = self.one_step_denoise(crops)
            # ── then RS Gaussian noise ──
            noise   = torch.randn_like(denoised) * self.sigma
            noisy   = (denoised + noise).clamp(0, 1)
            # ── classify ──
            agg     = agg + self.classify_logits(noisy)

        agg = agg / self.m_per_crop
        agg = agg.view(B, self.k_crops, -1).mean(dim=1)
        return agg