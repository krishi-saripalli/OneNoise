import os
# os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '1'
# os.environ['TORCHINDUCTOR_CACHE_DIR'] = './torchinductor_cache'
import json
import torch
import math
import argparse
import sys
import yaml


#TODO: hacky, need to come up with a better way to do this
# Add project root to sys.path to allow finding the 'utils' module
# Get the directory of the current script (OneNoise/test.py)
# os.path.dirname(os.path.abspath(__file__)) would be /path/to/BVC/infd/OneNoise
# The project root is one level up from the 'OneNoise' directory.
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from types import SimpleNamespace
from torchvision.utils import save_image, make_grid
from on_utils.helpers import seed_everything, load_config_recursive, dict_to_namespace, load_infd_ae_components

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
infd_quantizer = None
if config.latent_diffusion:
    print("INFO: Latent diffusion mode enabled. Loading INFD AE...")
    # Use the helper function to load AE components
    infd_decoder, infd_renderer, infd_quantizer = load_infd_ae_components(
        ae_config_path=config.ae_config_path,
        ae_checkpoint_path=config.ae_checkpoint_path,
        device=device # Use the device specified by args
    )
    if infd_quantizer is None:
        print("WARNING: Latent diffusion enabled but INFD quantizer was not loaded. " +
              "This might be an issue if the diffusion model expects quantized latents.")

inf = Inference(config, device=device, 
                is_latent_diffusion=config.latent_diffusion,
                infd_decoder=infd_decoder,
                infd_renderer=infd_renderer,
                infd_quantizer=infd_quantizer)

# Optionally you can compile the model for faster inference. 
# This has some initial overhead but it's faster for repeated forward calls.
# inf.model.model.forward = torch.compile(inf.model.model.forward)

# Image size for final output by INFD AE:
H = 256
W = 256 # Changed from 1024 to 256, then to 512, now trying 256 again alongside smaller latent size

# Create a smooth linear gradient for noise blending:
#mask_raw = smooth_linear_gradient(W=W, kernel_width=128, blur_iter=200_000)
mask_raw = smooth_linear_gradient(W=W, kernel_width=(W // 2), blur_iter=100)
print(f"[test.py] mask_raw (from smooth_linear_gradient) min: {mask_raw.min().item():.4f}, max: {mask_raw.max().item():.4f}, mean: {mask_raw.mean().item():.4f}, shape: {mask_raw.shape}") # DEBUG PRINT


mask = "/users/ksaripal/BVC/infd/OneNoise/inference/masks/axe.png" # Expands to (1, 1, H, W)


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
# cond_pairs = horizontal_blends() # Old way, using multiple unknown classes
# cond_pairs = [cond_pairs[i] for i in [0,7,6]] # grab a few pairs of noise configurations

imgs = []
with torch.no_grad():
    for i, (c1, c2) in enumerate(cond_pairs):
        print(f"--- Interpolating Pair {i+1} ---")
        print(f"c1 cls: {c1.get('cls')}, sbsparams: {c1.get('sbsparams')}")
        print(f"c2 cls: {c2.get('cls')}, sbsparams: {c2.get('sbsparams')}")
        img = inf.slerp_mask(mask=mask,
                                blending_factor=0.0,
                                dict1=c1,
                                dict2=c2,
                                H=H,
                                W=W)
        imgs.append(img)

imgs = torch.cat(imgs, dim=0)
grid = make_grid(imgs, nrow=int(math.sqrt(imgs.shape[0])), padding=10)
save_image(grid, args.output_file)

print(f"Output grid saved to {args.output_file}")
