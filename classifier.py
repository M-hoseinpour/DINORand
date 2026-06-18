import torch
import torch.nn as nn
import torch.nn.functional as F

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
