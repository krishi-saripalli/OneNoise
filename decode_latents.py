import argparse
import os
import torch
import h5py
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm

import models
from OneNoise.on_utils.helpers import load_config_recursive
from utils.geometry import make_coord_cell_grid
from OneNoise.noise_data import HDF5Dataset


class LatentDecoder:
    """
    A class to decode a single latent vector from a HDF5 file using a pre-trained Autoencoder.
    """
    def __init__(self, args):
        """
        Initializes the LatentDecoder.

        Args:
            args (argparse.Namespace): Command-line arguments.
        """
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device.lower() else "cpu")
        print(f"Using device: {self.device}")

        self.ae_cfg_dict = load_config_recursive(args.ae_config_path)
        self.ae_model = self._load_ae_model()
        self.decoder = self.ae_model.decoder
        self.renderer = self.ae_model.renderer
        self.quantizer = getattr(self.ae_model, 'quantizer', None)
        assert self.decoder is not None and self.renderer is not None

    def _load_ae_model(self):
        """Loads the Autoencoder model from config and checkpoint."""
        print("Loading Autoencoder model...")
        model_config_dict = self.ae_cfg_dict['model']
        ae_model = models.make(model_config_dict)

        checkpoint = torch.load(self.args.ae_checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model']['sd']

        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

        ae_model.load_state_dict(state_dict)
        ae_model.to(self.device)
        ae_model.eval()
        print("Model loaded successfully.")
        return ae_model

    def _get_conditioning_data(self):
        """
        Initializes a HDF5Dataset to retrieve the conditioning data (class labels and substance parameters)
        for a specific sample.
        """
        h5_dataset_args = self.ae_cfg_dict['datasets']['train']['args']['dataset']['args']
        h5_provider_dataset_init_args = {
            **h5_dataset_args,
            "noise_types": [self.args.noise_type],
            "augment": False,
            "cutmix": 0,
            "cutmix_prob": 0,
            "is_latent": False,
        }
        
        # Remove keys that are not part of HDF5Dataset's __init__
        for key in ['rank', 'world_size', 'max_samples']:
            h5_provider_dataset_init_args.pop(key, None)

        cond_map_provider_dataset = HDF5Dataset(**h5_provider_dataset_init_args)
        _, cls_labels, sbs_params = cond_map_provider_dataset[self.args.sample_index]

        assert cls_labels.ndim == 3, "Class labels should have 3 dimensions."
        cls_labels = cls_labels.unsqueeze(0).to(self.device)
        sbs_params = sbs_params.unsqueeze(0).to(self.device)

        return cls_labels, sbs_params

    def decode_and_save(self):
        """
        Decodes a latent vector specified by sample_index and saves the resulting image.
        """
        assert os.path.exists(self.args.latent_hdf5_path), f"Latent file not found at {self.args.latent_hdf5_path}"
        
        cls_labels, sbs_params = self._get_conditioning_data()

        with h5py.File(self.args.latent_hdf5_path, 'r') as f:
            assert self.args.noise_type in f, f"Noise type '{self.args.noise_type}' not found in HDF5 file."
            latents_dataset = f[self.args.noise_type]
            num_samples = latents_dataset.shape[0]
            assert 0 <= self.args.sample_index < num_samples, f"sample_index {self.args.sample_index} is out of range for {num_samples} samples."
            
            latent_np = latents_dataset[self.args.sample_index]

        latent_tensor = torch.from_numpy(latent_np).float().unsqueeze(0).to(self.device)
        if latent_tensor.ndim == 3: # Handle cases where channel dimension might be missing
             latent_tensor = latent_tensor.unsqueeze(1)
        assert latent_tensor.ndim == 4, "Latent tensor must be 4D (B, C, H, W)."

        with torch.no_grad():
            # The decoding pipeline must match the training process exactly.
            latents_conv = self.decoder[0](latent_tensor)
            latents_post_q = self.quantizer(latents_conv)[0] if self.quantizer else latents_conv
            
            core_decoder_module = self.decoder[1]
            coord, cell = make_coord_cell_grid(shape=(self.args.output_size, self.args.output_size), device=self.device, bs=latents_post_q.shape[0])
            
            features = core_decoder_module(latents_post_q, classes=cls_labels, substance_params=sbs_params)
            reconstructed_image = self.renderer(features, coord=coord, cell=cell)
            
            # Denormalize from [-1, 1] to [0, 1]
            reconstructed_image = (reconstructed_image + 1) / 2
            reconstructed_image = reconstructed_image.clamp(0., 1.)

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(self.args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        save_image(reconstructed_image, self.args.output_path)
        print(f"Saved decoded image to {self.args.output_path}")

def main(args):
    """Main function to run the latent decoding process."""
    decoder = LatentDecoder(args)
    decoder.decode_and_save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode a latent vector from an HDF5 file.")
    parser.add_argument("--latent_hdf5_path", type=str, required=True)
    parser.add_argument("--ae_config_path", type=str, required=True)
    parser.add_argument("--ae_checkpoint_path", type=str, required=True)
    parser.add_argument("--noise_type", type=str, default="voronoi")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="decoded_latent_sample.png")
    parser.add_argument("--output_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parsed_args = parser.parse_args()
    main(parsed_args) 