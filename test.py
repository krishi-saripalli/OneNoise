import os
import json
import torch
import math
import argparse

from torchvision.utils import save_image, make_grid
from on_utils.helpers import seed_everything, load_infd_ae_components

from inference.inference import Inference
# from inference.example_noises import horizontal_blends
from inference.inference_helpers import smooth_linear_gradient

seed_everything(31415)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
def parse_args():
    parser = argparse.ArgumentParser(description="OneNoise Inference Script")
    
    parser.add_argument('--out_dir', type=str, default='./results', help="Base directory for results/checkpoints")
    parser.add_argument('--exp_name', type=str, required=True, help="Experiment name (directory containing config.json and model checkpoint)")
    parser.add_argument('--checkpoint', type=str, default=None, help="Specific checkpoint file (e.g., model-10.pt). If None, uses the latest.")
    parser.add_argument('--sample_timesteps', type=int, default=50, help='Number of diffusion timesteps for sampling (DDIM)')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device for inference')
    parser.add_argument('--seed', type=int, default=31415, help='Random seed')
    parser.add_argument('--output_file', type=str, default='output.png', help='Name for the output image grid file')

    parser.add_argument('--latent_diffusion', action='store_true', help='Enable latent diffusion mode.')
    parser.add_argument('--ae_config_path', type=str, default=None, help='Path to the AE model config YAML (required if --latent_diffusion)')
    parser.add_argument('--ae_checkpoint_path', type=str, default=None, help='Path to the AE model checkpoint .pth (required if --latent_diffusion)')

    args = parser.parse_args()
    return args

args = parse_args()
seed_everything(args.seed)
device = torch.device(args.device)

exp_dir = os.path.join(args.out_dir, args.exp_name)
config_path = os.path.join(exp_dir, 'config.json')
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found at {config_path}")

print(f"Loading training config from: {config_path}")
with open(config_path, 'r') as f:
    config_dict = json.load(f)
    config_dict.update(vars(args)) 
    config = argparse.Namespace(**config_dict)

infd_decoder = None
infd_renderer = None
infd_quantizer = None
if config.latent_diffusion:
    print("INFO: Latent diffusion mode enabled. Loading INFD AE...")
    infd_decoder, infd_renderer, infd_quantizer = load_infd_ae_components(
        ae_config_path=config.ae_config_path,
        ae_checkpoint_path=config.ae_checkpoint_path,
        device=device
    )
    if infd_quantizer is None:
        print("WARNING: Latent diffusion enabled but INFD quantizer was not loaded. " +
              "This might be an issue if the diffusion model expects quantized latents.")

inf = Inference(config, device=device, 
                is_latent_diffusion=config.latent_diffusion,
                infd_decoder=infd_decoder,
                infd_renderer=infd_renderer,
                infd_quantizer=infd_quantizer)

H = 256
W = 256

mask_raw = smooth_linear_gradient(W=W, kernel_width=(W // 2), blur_iter=100)

mask = "/users/ksaripal/BVC/infd/OneNoise/inference/masks/axe.png"

voronoi_params1_start = {
    'scale': 1.0, 
    'distortion_intensity': 0.0, 
    'distortion_scale_multiplier': 0.0
}
voronoi_params1_end = {
    'scale': 0.0, 
    'distortion_intensity': 0.0, 
    'distortion_scale_multiplier': 1.0
}

cond_pairs = [
    (
        {'cls': 'voro', 'sbsparams': voronoi_params1_start},
        {'cls': 'voro', 'sbsparams': voronoi_params1_end}
    )
]

imgs = []
with torch.no_grad():
    for i, (c1, c2) in enumerate(cond_pairs):
        img = inf.slerp_mask(mask=mask,
                                blending_factor=0.3,
                                dict1=c1,
                                dict2=c2,
                                H=H,
                                W=W)
        imgs.append(img)

imgs = torch.cat(imgs, dim=0)
grid = make_grid(imgs, nrow=int(math.sqrt(imgs.shape[0])), padding=10)
save_image(grid, args.output_file)

print(f"Output grid saved to {args.output_file}")
