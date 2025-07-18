import argparse
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
import h5py 
import numpy as np 
from tqdm import tqdm 
import on_utils
from torchvision import transforms
from torch.utils.data import Dataset

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    print(f"Adding {project_root} to sys.path")
    sys.path.insert(0, project_root)

# infd imports
import models
import on_utils 
import datasets

from on_utils.helpers import load_config_recursive, dict_to_namespace 

from OneNoise.noise_data import HDF5Dataset 


class LatentGenerator:
    def __init__(self, args):
        self.args = args
        self.cfg_dict = load_config_recursive(args.ae_train_config)
        self.encoder = self.load_ae_encoder(args)
        self.data_loader, self.dataset = self.load_h5_dataset(args, self.cfg_dict)

    def generate_latents(self):
        device = torch.device(self.args.device)
        data_loader = self.data_loader
        encoder = self.encoder
        dataset = self.dataset
        args = self.args
        
        # Extract inner_dataset_args from config
        dataset_split = 'train'
        inner_dataset_args = self.cfg_dict['datasets'][dataset_split]['args']['dataset']['args']
        
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
                image_batch, label_batch, param_batch = batch_tuple
                input_tensor = image_batch.to(device) 
                
                latent_batch = encoder(input_tensor).cpu()
                # Since Cutmix was disabled, take value at center pixel (or [0,0]) as representative
                cls_indices = label_batch.argmax(dim=1)[:, 0, 0] # Get index along class dim, take top-left pixel
                
                for i in range(latent_batch.shape[0]):
                    latent_sample = latent_batch[i]
                    label_index = cls_indices[i].item()
                    assert label_index < len(config_noise_types), f"Encountered cls_label index {label_index} which is out of bounds for config_noise_types (len={len(config_noise_types)}). Skipping sample."
                    noise_type = config_noise_types[label_index]
                    latents_by_type[noise_type].append(latent_sample)
                
                processed_count += latent_batch.shape[0]
        print(f"Saving latents to HDF5 file: {self.args.output_path}...")
        
        active_noise_types = [ntype for ntype, latents in latents_by_type.items() if latents]
        assert active_noise_types, "No latents were collected for any noise type. Cannot save."
        
        counts_per_type = {ntype: len(latents_by_type[ntype]) for ntype in active_noise_types}
        first_count = counts_per_type[active_noise_types[0]]
        is_consistent = all(count == first_count for count in counts_per_type.values())   
        num_images_per_type = first_count
        assert is_consistent, "Inconsistent number of samples per noise type. Cannot save."

        with h5py.File(self.args.output_path, 'w') as f:
            f.attrs['num_images_per_type'] = num_images_per_type
            f.attrs['noise_types'] = active_noise_types

            for ntype in active_noise_types:
                latents_list = latents_by_type[ntype]
                print(f"Saving dataset for '{ntype}' ({len(latents_list)} samples)... ")
                latents_tensor = torch.stack(latents_list, dim=0)
                latents_np = latents_tensor.numpy().astype(np.float32) 
                
                expected_latent_shape_inner = latent_shape[1:]
                assert latents_np.shape[1:] == tuple(expected_latent_shape_inner), f"Shape mismatch for '{ntype}'! Expected {expected_latent_shape_inner}, got {latents_np.shape[1:]}"
                f.create_dataset(ntype, data=latents_np, compression="gzip")

        print(f"All latents saved to {self.args.output_path}")

    def load_ae_encoder(self):
        device = torch.device(self.args.device)

        print(f"Loading AE training config from: {self.args.ae_train_config}")
        
        print("Instantiating AE model structure...")
        model_config_dict = self.cfg_dict['model'] 
        model = models.make(model_config_dict)

        print(f"Loading AE checkpoint from: {self.args.ae_checkpoint_path}")
        if not os.path.exists(self.args.ae_checkpoint_path):
            proj_rel_path = os.path.join(project_root, self.args.ae_checkpoint_path)
            if os.path.exists(proj_rel_path):
                checkpoint_path = proj_rel_path
                print(f"  (Resolved relative path to: {checkpoint_path})")
            else:
                raise FileNotFoundError(f"Checkpoint file not found at {self.args.ae_checkpoint_path} or {proj_rel_path}")
        else:
            checkpoint_path = self.args.ae_checkpoint_path

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
            print("(Removing 'module.' prefix from state dict keys)")
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
        print(f"Encoder with {sum(p.numel() for p in model.encoder.parameters())} parameters loaded and set to evaluation mode.")
        return model.encoder

    def load_h5_dataset(self, args, cfg_dict):
        dataset_split = 'train'
        if dataset_split not in cfg_dict['datasets']:
            raise ValueError(f"Dataset split '{dataset_split}' not found in config.")

        inner_dataset_args = cfg_dict['datasets'][dataset_split]['args']['dataset']['args']
        loader_config = cfg_dict['datasets'][dataset_split]['loader']
        
        print("Instantiating HDF5Dataset with args:", inner_dataset_args)
        dataset = HDF5Dataset(
            **inner_dataset_args,
            is_latent=False,    
            rank=0,             
            world_size=1 #only support single GPU
        )
        print(f"HDF5Dataset created. Length: {len(dataset)}")
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=loader_config.get('num_workers', 0),
            pin_memory=True,
            drop_last=False
        )
        print(f"DataLoader created. Batch size: {args.batch_size}, Workers: {loader_config.get('num_workers', 0)}")

        return data_loader, dataset


def main(args):
    latent_generator = LatentGenerator(args)
    latent_generator.generate_latents()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate latents using a pre-trained AE and save to a single HDF5 file")
    parser.add_argument("--ae_train_config", type=str, required=True, help="Path to the AE training config YAML file (e.g., cfgs/ae_custom_h5.yaml)")
    parser.add_argument("--ae_checkpoint_path", type=str, required=True, help="Path to the AE checkpoint (.pth file)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output HDF5 file containing latents.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing images through the encoder.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Torch device (e.g., 'cuda:0', 'cpu')")

    args = parser.parse_args()
    main(args)