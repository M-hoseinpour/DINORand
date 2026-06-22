import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.checkpoint import checkpoint

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
        use_checkpoint=False,
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
        x_clean_ddpm = out['pred_xstart']

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

        def one_iteration(crops_in):
            denoised = self.one_step_denoise(crops_in)
            noise    = torch.randn_like(denoised) * self.sigma
            noisy    = (denoised + noise).clamp(0, 1)
            return self.classify_logits(noisy)

        agg = torch.zeros(B * self.k_crops, self.prototypes.shape[0],
                        device=x.device)

        for _ in range(self.m_per_crop):
            agg = agg + checkpoint(one_iteration, crops, use_reentrant=False)

        agg = agg / self.m_per_crop
        agg = agg.view(B, self.k_crops, -1).mean(dim=1)
        return agg




def position_to_text(idx, grid_size=16):
    """
    Map a position in the 16x16 attention grid to a text descriptor.
    
    Grid divided into 3x3 = 9 regions:
        rows 0-4   = upper
        rows 5-10  = middle
        rows 11-15 = lower
        cols 0-4   = left
        cols 5-10  = center
        cols 11-15 = right
    """
    row = idx // grid_size
    col = idx %  grid_size
 
    if   row <= 4:  vert = 'upper'
    elif row <= 10: vert = 'middle'
    else:           vert = 'lower'
 
    if   col <= 4:  horiz = 'left'
    elif col <= 10: horiz = 'center'
    else:           horiz = 'right'
 
    # 9 unique combinations
    if vert == 'middle' and horiz == 'center':
        return 'center of an image'
    elif vert == 'middle':
        return f'{horiz} side of an image'
    elif horiz == 'center':
        return f'{vert} part of an image'
    else:
        return f'{vert}-{horiz} part of an image'

# Precompute all 9 unique position descriptions
ALL_POSITION_TEXTS = []
_seen = set()
for r in range(16):
    for c in range(16):
        t = position_to_text(r * 16 + c)
        if t not in _seen:
            _seen.add(t)
            ALL_POSITION_TEXTS.append(t)

class TextEmbeddingCache:
    """Cache the dino.txt text embeddings for the 9 position descriptors."""
    def __init__(self, dino_txt_model, device):
        self.device = device
        self.text_to_idx = {t: i for i, t in enumerate(ALL_POSITION_TEXTS)}
        with torch.no_grad():
            # dino.txt expects a list of strings for encode_text
            text_features = dino_txt_model.encode_text(ALL_POSITION_TEXTS)
            text_features = F.normalize(text_features, dim=-1)
            self.text_features = text_features.to(device)
            # shape: (9, text_dim)
 
    def get_text_features(self):
        return self.text_features
 
    def text_idx(self, text):
        return self.text_to_idx[text]

        
class TextCoherentCropRSClassifier(nn.Module):
    """
    Variant of CropRSClassifier that selects noise samples per crop
    based on agreement with position-descriptive text via dino.txt.
    
    For each crop:
        Generate M noisy versions
        For each noisy version, compute dino.txt image features
        Score each version by its similarity to the crop's expected
            position-text features
        Use top-K most coherent versions (or weighted average)
            for classification
    
    Then aggregate per-crop logits as in the baseline.
    
    Args:
        dino:           frozen DINOv2 backbone (for classification)
        normalizer:     ImageNet normalize transform
        prototypes:     (1000, feat_dim) class prototypes
        sigma:          Gaussian noise std for RS smoothing
        m_per_crop:     M, number of noise samples per crop
        k_crops:        K, number of crops per image
        crop_size:      crop size in pixels
        min_patch_dist: NMS distance for crop selection
        dino_txt:       loaded dino.txt model (must have encode_image,
                            encode_text)
        text_cache:     TextEmbeddingCache instance
        selection:      'top1' | 'top3' | 'weighted'
        top_k_select:   K for top-K selection (used when selection='topK')
        temperature:    softmax temperature for 'weighted' selection
    """
    def __init__(self, dino, normalizer, prototypes, sigma, m_per_crop,
                 k_crops, crop_size, min_patch_dist,
                 dino_txt, text_cache,
                 selection='weighted', top_k_select=3, temperature=10.0):
        super().__init__()
        self.dino           = dino
        self.normalizer     = normalizer
        self.prototypes     = prototypes
        self.sigma          = sigma
        self.m_per_crop     = m_per_crop
        self.k_crops        = k_crops
        self.crop_size      = crop_size
        self.min_patch_dist = min_patch_dist
        self.dino_txt       = dino_txt
        self.text_cache     = text_cache
        self.selection      = selection
        self.top_k_select   = top_k_select
        self.temperature    = temperature
 
    def classify_logits(self, x):
        """Classification via frozen DINOv2 + prototype cosine similarity."""
        feats = self.dino(self.normalizer(x))
        feats = F.normalize(feats, dim=-1)
        return feats @ self.prototypes.T * 100
 
    @torch.no_grad()
    def text_coherence_scores(self, noisy_crops, expected_text):
        """
        For each noisy crop, compute its similarity to the expected
        position-text features.
        
        noisy_crops:    (M, 3, 224, 224)
        expected_text:  string, e.g. 'upper-right part of an image'
        
        Returns: (M,) scores
        """
        # Encode noisy crops via dino.txt
        # dino.txt encode_image expects normalized inputs
        feats = self.dino_txt.encode_image(self.normalizer(noisy_crops))
        feats = F.normalize(feats, dim=-1)
        # feats: (M, embed_dim)
 
        text_idx  = self.text_cache.text_idx(expected_text)
        text_feat = self.text_cache.get_text_features()[text_idx:text_idx+1]
        # text_feat: (1, embed_dim)
 
        scores = (feats @ text_feat.T).squeeze(-1)   # (M,)
        return scores
 
    def forward(self, x):
        B = x.shape[0]
        device = x.device
 
        # Standard attention-based crop selection (same as baseline)
        with torch.no_grad():
            indices = topk_attended_positions_batch(
                self.dino, self.normalizer, x.detach(),
                self.k_crops, self.min_patch_dist,
            )
 
        crops = extract_crops_batch(x, indices, self.crop_size)
        # crops: (B*K, 3, 224, 224)
 
        # For text scoring we need per-crop expected text
        expected_texts = []
        for b in range(B):
            for pos in indices[b]:
                expected_texts.append(position_to_text(pos))
        # expected_texts: list of B*K strings
 
        # Process all crops in parallel
        # Generate all M noisy versions for all B*K crops
        # Memory: (B*K*M, 3, 224, 224) — could be large
 
        # We loop over M to save memory
        per_crop_logits_accum = torch.zeros(B * self.k_crops,
                                             self.prototypes.shape[0],
                                             device=device)
        per_crop_text_scores_accum = torch.zeros(B * self.k_crops,
                                                   self.m_per_crop,
                                                   device=device)
        # Also store per-sample logits so we can apply selection later
        all_logits = []   # list of (B*K, num_classes) tensors, one per m
 
        for m_idx in range(self.m_per_crop):
            noise = torch.randn_like(crops) * self.sigma
            noisy = (crops + noise).clamp(0, 1)
 
            # DINOv2 classification logits
            logits_m = self.classify_logits(noisy)   # (B*K, num_classes)
            all_logits.append(logits_m)
 
            # dino.txt text coherence scores
            with torch.no_grad():
                # Encode all noisy crops
                feats_txt = self.dino_txt.encode_image(self.normalizer(noisy))
                feats_txt = F.normalize(feats_txt, dim=-1)
                # (B*K, text_embed_dim)
 
                # For each crop, get its expected text feature
                # Build a tensor of expected text features in same order as crops
                text_feats_all = self.text_cache.get_text_features()   # (9, dim)
                expected_indices = torch.tensor(
                    [self.text_cache.text_idx(t) for t in expected_texts],
                    device=device
                )
                expected_text_feats = text_feats_all[expected_indices]
                # (B*K, dim)
 
                # Cosine similarity per crop
                scores_m = (feats_txt * expected_text_feats).sum(dim=-1)
                # (B*K,)
                per_crop_text_scores_accum[:, m_idx] = scores_m
 
        # Stack all logits into (B*K, M, num_classes)
        all_logits_stacked = torch.stack(all_logits, dim=1)
        # per_crop_text_scores_accum: (B*K, M)
 
        # Selection / weighting
        if self.selection == 'top1':
            # Pick the single most coherent noise sample per crop
            best_idx = per_crop_text_scores_accum.argmax(dim=1)   # (B*K,)
            selected = all_logits_stacked[
                torch.arange(B * self.k_crops), best_idx
            ]
            # selected: (B*K, num_classes)
            per_crop_logits = selected
 
        elif self.selection == 'top3':
            # Average top-K most coherent
            k_sel = min(self.top_k_select, self.m_per_crop)
            _, top_idx = per_crop_text_scores_accum.topk(k_sel, dim=1)
            # top_idx: (B*K, k_sel)
 
            # Gather logits for those indices
            top_idx_expanded = top_idx.unsqueeze(-1).expand(
                -1, -1, all_logits_stacked.shape[-1]
            )
            selected = all_logits_stacked.gather(1, top_idx_expanded)
            # selected: (B*K, k_sel, num_classes)
            per_crop_logits = selected.mean(dim=1)
 
        elif self.selection == 'weighted':
            # Softmax weight by text score, then weighted sum
            weights = F.softmax(per_crop_text_scores_accum * self.temperature,
                                  dim=1)
            # weights: (B*K, M)
            per_crop_logits = (weights.unsqueeze(-1) *
                                all_logits_stacked).sum(dim=1)
            # (B*K, num_classes)
 
        else:
            raise ValueError(f'unknown selection: {self.selection}')
 
        # Aggregate across crops (same as baseline)
        per_crop_logits = per_crop_logits.view(B, self.k_crops, -1)
        return per_crop_logits.mean(dim=1)