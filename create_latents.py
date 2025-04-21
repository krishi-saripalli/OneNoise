import argparse
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
import h5py 
import numpy as np 
from tqdm import tqdm 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    print(f"Adding {project_root} to sys.path")
    sys.path.insert(0, project_root)

# infd imports
import models
import utils 
import datasets

# Import helpers from utils instead of defining locally
from utils.helpers import load_config_recursive, dict_to_namespace 

from OneNoise.noise_data import HDF5Dataset 

def main(args):
    print(f"Using device: {args.device}")
    device = torch.device(args.device)

    print(f"Loading AE training config from: {args.ae_train_config}")
    cfg_dict = load_config_recursive(args.ae_train_config)
    cfg = dict_to_namespace(cfg_dict)

    print("Instantiating AE model structure...")
    model_config_dict = cfg_dict['model'] 
    model = models.make(model_config_dict)

    print(f"Loading AE checkpoint from: {args.ae_checkpoint_path}")
    if not os.path.exists(args.ae_checkpoint_path):
        proj_rel_path = os.path.join(project_root, args.ae_checkpoint_path)
        if os.path.exists(proj_rel_path):
            checkpoint_path = proj_rel_path
            print(f"  (Resolved relative path to: {checkpoint_path})")
        else:
            raise FileNotFoundError(f"Checkpoint file not found at {args.ae_checkpoint_path} or {proj_rel_path}")
    else:
        checkpoint_path = args.ae_checkpoint_path

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model' in checkpoint and isinstance(checkpoint['model'], dict) and 'sd' in checkpoint['model']:
        state_dict = checkpoint['model']['sd']
        print("  (Extracting state dict from checkpoint['model']['sd'])")
    else:
        keys_found = list(checkpoint.keys())
        model_keys_found = list(checkpoint.get('model', {}).keys()) if isinstance(checkpoint.get('model'), dict) else None
        raise KeyError(f"Could not find state dict at checkpoint['model']['sd']. \
                     Top-level keys: {keys_found}. \
                     Keys under 'model': {model_keys_found}")

    if state_dict and list(state_dict.keys())[0].startswith('module.'):
        print("  (Removing 'module.' prefix from state dict keys)")
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    print("Checkpoint loaded successfully.")

    model.to(device)
    model.eval()
    encoder = model.encoder
    print(f"Encoder with {sum(p.numel() for p in model.encoder.parameters())} parameters loaded and set to evaluation mode.")

    # --- Dataset/DataLoader Creation (Direct HDF5Dataset) ---
    print("\nCreating DataLoader using HDF5Dataset directly...")
    dataset_split = 'train'
    if dataset_split not in cfg_dict['datasets']:
        raise ValueError(f"Dataset split '{dataset_split}' not found in config.")

    # get HDF5 Dataset args from  INFD config
    inner_dataset_args = cfg_dict['datasets'][dataset_split]['args']['dataset']['args']
    loader_config = cfg_dict['datasets'][dataset_split]['loader']
    
    print("  Instantiating HDF5Dataset with args:", inner_dataset_args)
    dataset = HDF5Dataset(
        **inner_dataset_args, # Unpack args from config
        is_latent=False,    
        rank=0,             
        world_size=1 #only support single GPU
    )
    print(f"  HDF5Dataset created. Length: {len(dataset)}")

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=loader_config.get('num_workers', 0),
        pin_memory=True,
        drop_last=False
    )
    print(f"DataLoader created. Batch size: {args.batch_size}, Workers: {loader_config.get('num_workers', 0)}")

    first_batch_tuple = next(iter(data_loader))
    sample_image_tensor = first_batch_tuple[0][0:1].to(device)
    sample_label_tensor = first_batch_tuple[1][0:1]
    print(f"  Input image shape (sample): {sample_image_tensor.shape}")
    print(f"  Label tensor shape (sample): {sample_label_tensor.shape}")

    with torch.no_grad():
        sample_latent = encoder(sample_image_tensor)
        latent_shape = sample_latent.shape 
        latent_dtype = sample_latent.dtype
        print(f"  Detected latent shape (per sample): {latent_shape[1:]}, Dtype: {latent_dtype}")

    # need noise types defined in the config to know the mapping from cls_label index to name
    config_noise_types = inner_dataset_args['noise_types']
    if not config_noise_types:
        raise ValueError(f"Could not find 'noise_types' in HDF5Dataset args. Needed for mapping class labels.")
    print(f"  Noise types defined in config: {config_noise_types}")

    print(f"\nStarting latent generation...")
    total_samples = len(dataset)
    print(f"Total samples to process: {total_samples}")
    
    latents_by_type = {ntype: [] for ntype in config_noise_types}
    processed_count = 0

    with torch.no_grad():
        for batch_tuple in tqdm(data_loader, desc="Encoding Batches"):
            # batch_tuple = (image_batch, label_batch, param_batch)
            input_tensor = batch_tuple[0].to(device) 
            label_tensor_batch = batch_tuple[1]      # label tensor [B, C, H, W] (one-hot spatial)
            
            latent_batch = encoder(input_tensor).cpu()
            
            # Convert spatial one-hot label tensor back to class indices
            # Shape [B, C, H, W] -> Need indices [B]
            # Take value at center pixel (or [0,0]) as representative
            # Assuming H, W >= 1
            cls_indices = label_tensor_batch.argmax(dim=1)[:, 0, 0] # Get index along class dim, take top-left pixel
            
            for i in range(latent_batch.shape[0]):
                latent_sample = latent_batch[i]
                label_index = cls_indices[i].item()
                if label_index >= len(config_noise_types):
                    print(f"Warning: Encountered cls_label index {label_index} which is out of bounds for config_noise_types (len={len(config_noise_types)}). Skipping sample.")
                    continue
                noise_type = config_noise_types[label_index]
                latents_by_type[noise_type].append(latent_sample)
            
            processed_count += latent_batch.shape[0]

    print(f"\nEncoding finished. Processed {processed_count} samples.")

    print(f"Saving latents to single HDF5 file: {args.output_path}...")
    
    active_noise_types = [ntype for ntype, latents in latents_by_type.items() if latents]
    if not active_noise_types:
        print("Error: No latents were collected for any noise type. Cannot save.")
        sys.exit(1)
    print(f"  Noise types with collected samples: {active_noise_types}")

    counts_per_type = {ntype: len(latents_by_type[ntype]) for ntype in active_noise_types}
    first_count = counts_per_type[active_noise_types[0]]
    is_consistent = all(count == first_count for count in counts_per_type.values())
    
    if not is_consistent:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Error: Inconsistent number of samples collected per noise type!")
        for ntype, count in counts_per_type.items():
            print(f"  - {ntype}: {count} samples")
        print("HDF5Dataset requires a consistent 'num_images_per_type' for correct indexing.")
        print("Cannot proceed with saving.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        sys.exit(1)
    else:
        num_images_per_type = first_count
        print(f"  Consistent number of samples per type found: {num_images_per_type}")

    with h5py.File(args.output_path, 'w') as f:
        f.attrs['num_images_per_type'] = num_images_per_type
        f.attrs['noise_types'] = active_noise_types

        for ntype in active_noise_types:
            latents_list = latents_by_type[ntype]
            print(f"  Saving dataset for '{ntype}' ({len(latents_list)} samples)... ")
            
            latents_tensor = torch.stack(latents_list, dim=0)
            latents_np = latents_tensor.numpy().astype(np.float32) 
            
            expected_latent_shape_inner = latent_shape[1:]
            if latents_np.shape[1:] != tuple(expected_latent_shape_inner):
                 print(f"    Warning: Shape mismatch for '{ntype}'! Expected {expected_latent_shape_inner}, got {latents_np.shape[1:]}")

            print(f"    Dataset Shape: {latents_np.shape}, Dtype: {latents_np.dtype}")
            f.create_dataset(ntype, data=latents_np, compression="gzip")

    print(f"Latent saving process complete. File saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate latents using a pre-trained AE and save to a single HDF5 file")
    parser.add_argument("--ae_train_config", type=str, required=True, help="Path to the AE training config YAML file (e.g., cfgs/ae_custom_h5.yaml)")
    parser.add_argument("--ae_checkpoint_path", type=str, required=True, help="Path to the AE checkpoint (.pth file)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output HDF5 file containing latents.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing images through the encoder.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Torch device (e.g., 'cuda:0', 'cpu')")

    args = parser.parse_args()
    main(args)