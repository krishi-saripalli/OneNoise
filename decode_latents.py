import argparse
import os
import sys
import yaml # PyYAML
import torch
import h5py
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm

# Add INFD project root to sys.path to allow imports like models, on_utils
# This assumes this script is in <project_root>/OneNoise/decode_latents.py
_this_file_path = os.path.abspath(__file__)
_one_noise_dir_path = os.path.dirname(_this_file_path)
_project_root_path = os.path.dirname(_one_noise_dir_path)

if _project_root_path not in sys.path:
    sys.path.insert(0, _project_root_path)

# Now we can import from the project root
import models # From INFD
from on_utils.helpers import load_config_recursive # From INFD/on_utils
from utils.geometry import make_coord_cell_grid # Added import

def calculate_latent_stats_batched(latent_batch_tensor: torch.Tensor):
    """Calculates per-channel and overall mean/variance for a batch of latent tensors."""
    # latent_batch_tensor is expected to be (B, C, H, W)
    if latent_batch_tensor.ndim != 4:
        raise ValueError(f"Expected latent_batch_tensor to be 4D (B,C,H,W), got {latent_batch_tensor.shape}")

    batch_per_channel_mean = latent_batch_tensor.mean(dim=[2, 3]) # Result (B, C)
    batch_per_channel_var = latent_batch_tensor.var(dim=[2, 3], unbiased=False) # Result (B, C)
    
    flattened_latents = latent_batch_tensor.view(latent_batch_tensor.shape[0], -1) # (B, C*H*W)
    batch_overall_mean = flattened_latents.mean(dim=1) # Result (B)
    batch_overall_var = flattened_latents.var(dim=1, unbiased=False) # Result (B)
    
    return {
        "batch_per_channel_mean": batch_per_channel_mean,
        "batch_per_channel_var": batch_per_channel_var,
        "batch_overall_mean": batch_overall_mean,
        "batch_overall_var": batch_overall_var,
    }

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device.lower() else "cpu")
    print(f"Using device: {device} for AE operations and batched stats calculations.")

    # 1. Load AE Model (needed for decoding the single sample)
    print(f"Loading AE training config from: {args.ae_config_path}")
    ae_cfg_dict = load_config_recursive(args.ae_config_path)
    model_config_dict = ae_cfg_dict['model']
    ae_model = models.make(model_config_dict)
    print("Instantiating AE model structure...")
    checkpoint = torch.load(args.ae_checkpoint_path, map_location='cpu')
    print(f"Loading AE checkpoint from: {args.ae_checkpoint_path}")

    state_dict = None
    if 'model' in checkpoint and isinstance(checkpoint['model'], dict) and 'sd' in checkpoint['model']:
        state_dict = checkpoint['model']['sd']
        print("  (Extracting state dict from checkpoint['model']['sd'])")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("  (Extracting state dict from checkpoint['state_dict'])")
    elif 'sd' in checkpoint:
        state_dict = checkpoint['sd']
        print("  (Extracting state dict from checkpoint['sd'])")
    elif checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
        is_likely_state_dict = True
        for v in checkpoint.values():
            if not isinstance(v, torch.Tensor) and not isinstance(v, np.ndarray):
                if not isinstance(v, (int, float, bool, str, list, tuple, dict, type(None))):
                    is_likely_state_dict = False; break
        if is_likely_state_dict: print("  (Checkpoint itself appears to be a state_dict)"); state_dict = checkpoint
        
    if state_dict is None:
        keys_found = list(checkpoint.keys())
        model_keys_found = list(checkpoint.get('model', {}).keys()) if isinstance(checkpoint.get('model'), dict) else None
        raise KeyError(f"Could not find a valid state dict. Top-level: {keys_found}. Under 'model': {model_keys_found}")

    if list(state_dict.keys())[0].startswith('module.'):
        print("  (Removing 'module.' prefix)"); state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    ae_model.load_state_dict(state_dict); print("AE Checkpoint loaded."); ae_model.to(device); ae_model.eval()

    decoder = ae_model.decoder; renderer = ae_model.renderer; quantizer = getattr(ae_model, 'quantizer', None)
    if decoder is None or renderer is None: raise AttributeError("AE model missing 'decoder' or 'renderer'.")

    print(f"Accessing latents from HDF5: {args.latent_hdf5_path}")
    if not os.path.exists(args.latent_hdf5_path): raise FileNotFoundError(f"HDF5 not found: {args.latent_hdf5_path}")

    with h5py.File(args.latent_hdf5_path, 'r') as f:
        if args.noise_type not in f: raise KeyError(f"Noise type '{args.noise_type}' not in HDF5. Available: {list(f.keys())}")
        
        latents_dataset = f[args.noise_type]
        num_samples_for_type = latents_dataset.shape[0]
        print(f"Found {num_samples_for_type} samples for '{args.noise_type}'.")

        if not (0 <= args.sample_index < num_samples_for_type):
            raise IndexError(f"Decode index {args.sample_index} out of bounds for '{args.noise_type}' (0 to {num_samples_for_type - 1}).")

        all_b_per_channel_means, all_b_per_channel_vars, all_b_overall_means, all_b_overall_vars = [], [], [], []
        
        stats_batch_size = args.stats_batch_size
        num_batches = (num_samples_for_type + stats_batch_size - 1) // stats_batch_size
        print(f"\nCalculating statistics in {num_batches} batches (size {stats_batch_size})...")

        for i in tqdm(range(num_batches), desc=f"Processing '{args.noise_type}' latents"):
            start_idx, end_idx = i * stats_batch_size, min((i + 1) * stats_batch_size, num_samples_for_type)
            latent_np_batch = latents_dataset[start_idx:end_idx]
            latent_tensor_batch = torch.from_numpy(latent_np_batch).float().to(device)

            if latent_tensor_batch.ndim == 3: latent_tensor_batch = latent_tensor_batch.unsqueeze(1)
            if latent_tensor_batch.ndim != 4:
                print(f"Warning: Skipping stats for batch {i} due to shape {latent_tensor_batch.shape} (expected 4D)."); continue

            stats = calculate_latent_stats_batched(latent_tensor_batch)
            all_b_per_channel_means.append(stats["batch_per_channel_mean"].cpu()) # Move to CPU before accumulating
            all_b_per_channel_vars.append(stats["batch_per_channel_var"].cpu())
            all_b_overall_means.append(stats["batch_overall_mean"].cpu())
            all_b_overall_vars.append(stats["batch_overall_var"].cpu())

        if all_b_overall_means:
            avg_per_channel_mean = torch.cat(all_b_per_channel_means, dim=0).mean(dim=0)
            avg_per_channel_var = torch.cat(all_b_per_channel_vars, dim=0).mean(dim=0)
            avg_overall_mean = torch.cat(all_b_overall_means, dim=0).mean()
            avg_overall_var = torch.cat(all_b_overall_vars, dim=0).mean()
            print(f"\n--- Aggregate Latent Statistics for '{args.noise_type}' ---")
            print(f"  Avg Per-Channel Mean: {avg_per_channel_mean.tolist()}")
            print(f"  Avg Per-Channel Variance: {avg_per_channel_var.tolist()}")
            print(f"  Avg Overall Mean: {avg_overall_mean.item():.6f}")
            print(f"  Avg Overall Variance: {avg_overall_var.item():.6f}")
        else: print(f"\nNo stats calculated for '{args.noise_type}'.")
        
        print(f"\n--- Decoding Sample {args.sample_index} for '{args.noise_type}' ---")
        latent_np_decode = latents_dataset[args.sample_index]
        print(f"  Loaded latent {args.sample_index} for decode, shape: {latent_np_decode.shape}")

    latent_tensor_decode = torch.from_numpy(latent_np_decode).float().unsqueeze(0).to(device)
    if latent_tensor_decode.ndim == 3 and latent_np_decode.ndim == 2: latent_tensor_decode = latent_tensor_decode.unsqueeze(1)
    print(f"  Latent tensor for AE decode: {latent_tensor_decode.shape}")
    if latent_tensor_decode.ndim != 4: raise ValueError(f"Latent for decode must be 4D, got {latent_tensor_decode.shape}")

    print("Decoding latent sample...")
    with torch.no_grad():
        latents_for_decoder = quantizer(latent_tensor_decode)[0] if quantizer else latent_tensor_decode
        coord, cell = make_coord_cell_grid(shape=(args.output_size, args.output_size), device=device, bs=latents_for_decoder.shape[0])
        features = decoder(latents_for_decoder)
        reconstructed_image = renderer(features, coord=coord, cell=cell)
        reconstructed_image = (reconstructed_image + 1) / 2; reconstructed_image = reconstructed_image.clamp(0., 1.)
    print(f"  Decoded image shape: {reconstructed_image.shape}")

    out_path_is_dir = "/" in args.output_path or args.output_path == "decoded_latent_sample.png"
    final_out_dir = args.output_path if out_path_is_dir and "/" in args.output_path else "."
    if out_path_is_dir: os.makedirs(final_out_dir, exist_ok=True)
    filename = os.path.join(final_out_dir, f"decoded_{args.noise_type}_sample_{args.sample_index}_size{args.output_size}.png") if out_path_is_dir else args.output_path
    save_image(reconstructed_image, filename); print(f"Reconstructed image saved to: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode sample & calculate latent stats from HDF5 using INFD AE.")
    
    parser.add_argument("--latent_hdf5_path", type=str, required=True, help="Path to HDF5 latents file.")
    parser.add_argument("--ae_config_path", type=str, required=True, help="Path to AE model config YAML.")
    parser.add_argument("--ae_checkpoint_path", type=str, required=True, help="Path to AE model checkpoint .pth.")
    parser.add_argument("--noise_type", type=str, default="voronoi", help="Noise type in HDF5 to analyze.")
    parser.add_argument("--sample_index", type=int, default=0, help="Index of specific latent to decode.")
    parser.add_argument("--stats_batch_size", type=int, default=256, help="Batch size for HDF5 reading during statistics calculation.")
    parser.add_argument("--output_path", type=str, default="decoded_latent_sample.png", help="Output path/dir for decoded image.")
    parser.add_argument("--output_size", type=int, default=256, help="Output image H/W.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device (e.g., 'cuda:0', 'cpu').")

    parsed_args = parser.parse_args()
    main(parsed_args) 