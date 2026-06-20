import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from classifier import CropRSClassifier
from data import imagenet_val_loader
from utils import seed_eveything
from autoattack import AutoAttack
from typing import cast

p = argparse.ArgumentParser()

p.add_argument('--prototypes-path',  type=str, default=None)
p.add_argument('--batch-size',   type=int,   default=16)
p.add_argument('--crop-size', type=int, default=144)
p.add_argument('--k-crops', type=int, default=7)
p.add_argument('--imagenet-val', type=str, default=None)
p.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'imagenet'])
p.add_argument('--eps', type=float, default=None)
p.add_argument('--sigma', type=float, default=0.2)
p.add_argument("--noise-steps-per-crop-sample", type=int, default=10)
p.add_argument("--n-iter", type=int, default=100)
p.add_argument("--eot-iter", type=int, default=20)
p.add_argument("--n-samples", type=int, default=100)
p.add_argument('--start-idx', type=int, default=0,    help='Start sample index')
p.add_argument('--end-idx',   type=int, default=None, help='End sample index (exclusive)')
p.add_argument('--min-patch-dist', type=int, default=2)
p.add_argument('--ddpm-path', type=str, default=None)
p.add_argument('--ddpm-timestep', type=int, default=50)
p.add_argument('--use-ddpm', action='store_true')

if __name__ == "__main__":
    args = p.parse_args()

    is_cuda = torch.cuda.is_available()
    device  = torch.device('cuda' if is_cuda else 'cpu')
    seed_eveything(is_cuda)

    IMAGENET_VAL_PATH = '/content/val'
    PROTOTYPES_PATH   = '/content/imagenet_prototypes.pt'

    default_eps = 8/255
    if args.dataset == "imagenet":
        assert args.imagenet_val, 'provide --imagenet-val path'
        assert os.path.exists(args.prototypes_path), "provide prototypes path"

        n_classes = 1000
        default_eps = 4/255

    eps = args.eps if args.eps is not None else default_eps
    prototypes = torch.load(args.prototypes_path, map_location=device)

    n_eval = args.n_samples or 512
    start_idx = args.start_idx
    end_idx   = args.end_idx if args.end_idx is not None else n_eval

    dinov2 = cast(nn.Module, torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14'))
    dinov2 = dinov2.to(device).eval()
    for param in dinov2.parameters():
        param.requires_grad = False

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])

    if args.use_ddpm:
        from classifier import DiffusionRSClassifier, load_pixel_ddpm
        assert args.ddpm_path, 'provide --ddpm-path when --use-ddpm'

        print(f'Loading DDPM from {args.ddpm_path}')
        ddpm_model, diffusion = load_pixel_ddpm(args.ddpm_path, device)

        classifier = DiffusionRSClassifier(
            dino=dinov2, normalizer=normalizer, prototypes=prototypes,
            sigma=args.sigma, m_per_crop=args.noise_steps_per_crop_sample,
            k_crops=args.k_crops, crop_size=args.crop_size,
            min_patch_dist=args.min_patch_dist,
            ddpm_model=ddpm_model, diffusion=diffusion,
            ddpm_timestep=args.ddpm_timestep,
        ).to(device).eval()
    else:
        classifier = CropRSClassifier(
            dino=dinov2, normalizer=normalizer, prototypes=prototypes,
            sigma=args.sigma, m_per_crop=args.noise_steps_per_crop_sample,
            k_crops=args.k_crops, crop_size=args.crop_size,
            min_patch_dist=args.min_patch_dist,
        ).to(device).eval()

    val_loader = imagenet_val_loader(args.imagenet_val, batch_size=args.batch_size)

    x_all, y_all = [], []
    for x, y in val_loader:
        x_all.append(x); y_all.append(y)
        if sum(t.shape[0] for t in x_all) >= args.n_samples:
            break

    x_test = torch.cat(x_all)[:args.n_samples].to(device)
    y_test = torch.cat(y_all)[:args.n_samples].to(device)
    print(f'Loaded {len(x_test)} samples\n')

    adversary = AutoAttack(
        classifier, norm='Linf', eps=eps,
        version='rand', device=str(device), verbose=True,
        log_path=f'attack_log_{start_idx}_{end_idx}.txt'
    )
    adversary.attacks_to_run       = ['apgd-ce', 'apgd-dlr']
    adversary.seed = 0

    adversary.apgd.eot_iter        = args.eot_iter
    adversary.apgd.n_iter          = args.n_iter
    adversary.apgd.n_restarts      = 1

    adversary.square.n_queries     = 5000

    print(f'Processing samples {start_idx} to {end_idx}')

    ckpt_dir = (f'ckpt_eps{int(eps*255)}_n{n_eval}'
                f'_k{args.k_crops}_sigma{args.sigma}'
                f'_crop{args.crop_size}_m{args.noise_steps_per_crop_sample}'
                f'_iter{args.n_iter}_eot{args.eot_iter}')

    os.makedirs(ckpt_dir, exist_ok=True)

    chunk_size = args.batch_size
    chunk_size = args.batch_size
    all_y_adv, all_labels = [], []

    for start in range(start_idx, end_idx, chunk_size):
        chunk_end = min(start + chunk_size, end_idx)
        ckpt_path = os.path.join(ckpt_dir, f'chunk_{start:04d}_{chunk_end:04d}.pt')

        if os.path.exists(ckpt_path):
            print(f'[resume] chunk {start}-{chunk_end}')
            ckpt = torch.load(ckpt_path, map_location='cpu')
            all_y_adv.append(ckpt['y_adv'])
            all_labels.append(ckpt['labels'])
            continue

        x_chunk = x_test[start:chunk_end]
        y_chunk = y_test[start:chunk_end]
        x_adv_chunk, y_adv_chunk = adversary.run_standard_evaluation(x_chunk, y_chunk, bs=len(x_chunk), return_labels=True)

        torch.save({'y_adv': y_adv_chunk.cpu(), 'labels': y_chunk.cpu()}, ckpt_path)

        all_y_adv.append(y_adv_chunk.cpu())
        all_labels.append(y_chunk.cpu())
        print(f'[saved] chunk {start}-{chunk_end}')

    y_adv_all  = torch.cat(all_y_adv)
    labels_all = torch.cat(all_labels)
    correct    = (y_adv_all == labels_all).sum().item()

    print(f'\n[{start_idx}-{end_idx}] robust accuracy: 'f'{correct/len(labels_all)*100:.2f}% ({correct}/{len(labels_all)})')


