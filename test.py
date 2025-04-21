import os
# os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '1'
# os.environ['TORCHINDUCTOR_CACHE_DIR'] = './torchinductor_cache'
import json
import torch
import math
import argparse
import sys
import yaml

from types import SimpleNamespace
from torchvision.utils import save_image, make_grid
from utils.helpers import seed_everything, load_config_recursive, dict_to_namespace, load_infd_ae_components

from inference.inference import Inference
from inference.example_noises import horizontal_blends
from inference.inference_helpers import smooth_linear_gradient

seed_everything(31415)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
def parse_args():
    parser = argparse.ArgumentParser(description="OneNoise Inference Script")
    
    # --- Standard OneNoise Args (subset needed for inference) ---
    parser.add_argument('--model_config', type=str, default='tiny', help='Model configuration (extra_tiny, tiny, medium, large)')
    parser.add_argument('--out_dir', type=str, default='./results', help="Base directory for results/checkpoints")
    parser.add_argument('--exp_name', type=str, required=True, help="Experiment name (directory containing config.json and model checkpoint)")
    parser.add_argument('--checkpoint', type=str, default=None, help="Specific checkpoint file (e.g., model-10.pt). If None, uses the latest.")
    parser.add_argument('--sample_timesteps', type=int, default=50, help='Number of diffusion timesteps for sampling (DDIM)')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device for inference')
    parser.add_argument('--seed', type=int, default=31415, help='Random seed')
    parser.add_argument('--output_file', type=str, default='output.png', help='Name for the output image grid file')

    # --- Latent Diffusion Args ---
    parser.add_argument('--latent_diffusion', action='store_true', help='Enable latent diffusion mode.')
    parser.add_argument('--ae_config_path', type=str, default=None, help='Path to the AE model config YAML (required if --latent_diffusion)')
    parser.add_argument('--ae_checkpoint_path', type=str, default=None, help='Path to the AE model checkpoint .pth (required if --latent_diffusion)')

    args = parser.parse_args()
    return args
# ------------------------

# load config from .json
# json_path = os.path.join('./pretrained/tiny_spherical/config.json') # Removed hardcoded path
# with open(json_path, 'r') as f:
#     config = json.load(f)
#     config = SimpleNamespace(**config)

args = parse_args()
seed_everything(args.seed)
device = torch.device(args.device)

# Load the training config from the experiment directory
exp_dir = os.path.join(args.out_dir, args.exp_name)
config_path = os.path.join(exp_dir, 'config.json')
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found at {config_path}")

print(f"Loading training config from: {config_path}")
with open(config_path, 'r') as f:
    # Load config dict, then update with command-line args where applicable
    config_dict = json.load(f)
    config_dict.update(vars(args)) # Add/overwrite with command-line args
    config = argparse.Namespace(**config_dict) # Convert back to Namespace

# config.out_dir = 'pretrained'
# config.exp_name = 'tiny_spherical'

# You can change the number of diffusion timesteps here (~30-40 is usually fine)
# config.sample_timesteps = 75


infd_decoder = None
infd_renderer = None
if config.latent_diffusion:
    print("INFO: Latent diffusion mode enabled. Loading INFD AE...")
    # Use the helper function to load AE components
    infd_decoder, infd_renderer = load_infd_ae_components(
        ae_config_path=config.ae_config_path,
        ae_checkpoint_path=config.ae_checkpoint_path,
        device=device # Use the device specified by args
    )

inf = Inference(config, device=device, 
                is_latent_diffusion=config.latent_diffusion,
                infd_decoder=infd_decoder,
                infd_renderer=infd_renderer)

# Optionally you can compile the model for faster inference. 
# This has some initial overhead but it's faster for repeated forward calls.
# inf.model.model.forward = torch.compile(inf.model.model.forward)

# Image size:
H = 256
W = 1024

# Create a smooth linear gradient for noise blending:
mask = smooth_linear_gradient(W=W, kernel_width=128, blur_iter=200_000)
mask = mask.unsqueeze(0).unsqueeze(0).expand(1, H, W)
mask = mask.to(device)

cond_pairs = horizontal_blends()
cond_pairs = [cond_pairs[i] for i in [0,7,6]] # grab a few pairs of noise configurations
imgs = []
with torch.no_grad():
    for i, (c1, c2) in enumerate(cond_pairs):
        img = inf.slerp_mask(mask=mask,
                                blending_factor=1.,
                                dict1=c1,
                                dict2=c2,
                                H=H,
                                W=W)
        imgs.append(img)

imgs = torch.cat(imgs, dim=0)
grid = make_grid(imgs, nrow=int(math.sqrt(imgs.shape[0])), padding=10)
save_image(grid, args.output_file)

print(f"Output grid saved to {args.output_file}")
