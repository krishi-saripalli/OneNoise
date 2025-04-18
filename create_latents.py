import argparse
import os
import sys
import yaml
import torch

# --- Add parent directory (project root) to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    print(f"Adding {project_root} to sys.path")
    sys.path.insert(0, project_root)
# -------------------------------------------------------

# --- Imports from infd project ---
import models # Should import models.make
import utils # Need utils.load_config or similar
# ----------------------------------


def load_config_recursive(path, loaded_files=None):
    """Loads a YAML config, handling _base_ includes."""
    if loaded_files is None:
        loaded_files = set()
    if not os.path.isabs(path):
        path = os.path.join(project_root, path)

    if path in loaded_files:
        return {}
    loaded_files.add(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    if cfg is None: # Handle empty files
        return {}

    base_cfg = {}
    if '_base_' in cfg:
        bases = cfg['_base_']
        if isinstance(bases, str):
            bases = [bases]
        for base_path in bases:
            # If base_path starts with 'cfgs/', assume it's relative to project_root
            # Otherwise, assume it's relative to the current config file's directory
            if base_path.startswith('cfgs/'):
                base_full_path = os.path.normpath(os.path.join(project_root, base_path))
            else:
                # Original logic: Assume relative to the current config file's directory
                base_full_path = os.path.normpath(os.path.join(os.path.dirname(path), base_path))
            # --- End Updated Logic ---

            base_cfg_part = load_config_recursive(base_full_path, loaded_files)
            # Simple merge: current config overrides base
            base_cfg.update(base_cfg_part)
        del cfg['_base_']

    # Merge base and current config
    base_cfg.update(cfg)
    return base_cfg

def dict_to_namespace(d):
    """Recursively converts a dictionary to a namespace."""
    if isinstance(d, dict):
        ns = argparse.Namespace()
        for key, value in d.items():
            setattr(ns, key, dict_to_namespace(value))
        return ns
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

def main(args):
    print(f"Using device: {args.device}")
    device = torch.device(args.device)

    # 1. Load Training Config
    print(f"Loading AE training config from: {args.ae_train_config}")
    cfg_dict = load_config_recursive(args.ae_train_config)
    cfg = dict_to_namespace(cfg_dict)

    # 2. Instantiate Model
    print("Instantiating AE model structure...")
    model_config_dict = cfg_dict['model'] 
    model = models.make(model_config_dict)
    print("Model structure instantiated.")

    # 3. Load Checkpoint
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

    # --- Simplified State Dict Extraction (based on confirmed structure) ---
    if 'model' in checkpoint and isinstance(checkpoint['model'], dict) and 'sd' in checkpoint['model']:
        state_dict = checkpoint['model']['sd']
        print("  (Extracting state dict from checkpoint['model']['sd'])")
    else:
        keys_found = list(checkpoint.keys())
        model_keys_found = list(checkpoint.get('model', {}).keys()) if isinstance(checkpoint.get('model'), dict) else None
        raise KeyError(f"Could not find state dict at checkpoint['model']['sd']. \
                     Top-level keys: {keys_found}. \
                     Keys under 'model': {model_keys_found}")
    # --- End Simplified State Dict Extraction ---

    # Handle potential DataParallel/DDP prefix 'module.'
    if state_dict and list(state_dict.keys())[0].startswith('module.'):
        print("  (Removing 'module.' prefix from state dict keys)")
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    print("Checkpoint loaded successfully.")

    # 4. Prepare Encoder for Testing
    model.to(device)
    model.eval()
    encoder = model.encoder # Assuming the model has an 'encoder' attribute
    print("Encoder extracted and set to evaluation mode.")

    # 5. Dummy Tensor Test
    print("\nPerforming dummy tensor test...")
    # -- Determine input size from config --
    input_size = 256 # Default
    if hasattr(cfg, 'datasets') and hasattr(cfg.datasets, 'train') and hasattr(cfg.datasets.train, 'args'):
        if hasattr(cfg.datasets.train.args, 'resize_inp'):
            input_size = cfg.datasets.train.args.resize_inp
            print(f"  (Inferred input size {input_size}x{input_size} from cfg.datasets.train.args.resize_inp)")
        elif hasattr(cfg.datasets.train.args, 'final_crop_gt'): # Fallback
             input_size = cfg.datasets.train.args.final_crop_gt
             print(f"  (Inferred input size {input_size}x{input_size} from cfg.datasets.train.args.final_crop_gt)")
        else:
             print(f"  (Could not find input size in dataset config, using default {input_size}x{input_size})")
    else:
        print(f"  (Could not find dataset config, using default input size {input_size}x{input_size})")

    # -- Create dummy input --
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
    print(f"  Created dummy input tensor with shape: {dummy_input.shape}")

    # -- Forward pass --
    with torch.no_grad():
        dummy_latent = encoder(dummy_input)
    print(f"  Encoder forward pass successful.")
    print(f"  Output latent shape: {dummy_latent.shape}")

    # -- Verify output shape --
    expected_shape = None
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'args') and hasattr(cfg.model.args, 'z_shape'):
         expected_shape = list(cfg.model.args.z_shape)
         expected_shape.insert(0, 1) 
         print(f"  Expected latent shape (from cfg.model.args.z_shape): {expected_shape}")
    else:
         print("  Could not find 'z_shape' in model config to verify output shape.")

    if expected_shape and list(dummy_latent.shape) == expected_shape:
        print("  Output shape matches expected z_shape. Test PASSED.")
    elif expected_shape:
        print("  WARNING: Output shape does NOT match expected z_shape. Test FAILED.")
    else:
        print("  Could not verify output shape against config.")

    print("\nInitial loading and testing finished.")
    print("TODO: Add HDF5 dataset loading, iteration, and latent saving.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load pre-trained AE encoder and test")
    parser.add_argument("--ae_train_config", type=str, required=True, help="Path to the AE training config YAML file (e.g., cfgs/ae_custom_h5.yaml)")
    parser.add_argument("--ae_checkpoint_path", type=str, required=True, help="Path to the AE checkpoint (.pth file)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Torch device (e.g., 'cuda:0', 'cpu')")

    args = parser.parse_args()
    main(args)