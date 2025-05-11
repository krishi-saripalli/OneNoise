import os
import sys
import yaml
import argparse
import torch
import random
import numpy as np

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_config_recursive(path, loaded_files=None):
    # NOTE: Assumes relative paths like 'cfgs/...' are relative to project root
    # which is determined based on the location of *this* helpers.py file.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if loaded_files is None:
        loaded_files = set()
    
    original_path = path # Keep original path for error messages

    if not os.path.isabs(path):
        # Try resolving relative to project root first (common case for cfgs/)
        root_path = os.path.join(project_root, path)
        if os.path.exists(root_path):
            path = root_path
        else:
            # Fallback: try resolving relative to the *caller's* directory? 
            # This is tricky without knowing the caller. Let's stick to project root relative.
            # A better approach might be to require cfgs/ paths to be project root relative
            # and other relative paths to be relative to the *including* config file.
            # For now, assume paths are either absolute or project-root-relative.
             if not path.startswith('cfgs/'): # If it doesn't start with cfgs/ and isn't absolute, it's ambiguous
                  print(f"Warning: Relative path '{original_path}' provided for config loading is ambiguous. Assuming it's relative to project root '{project_root}'.")
             # If root_path didn't exist, this will likely fail below, which is intended.
             path = root_path

    if path in loaded_files:
        return {}
    loaded_files.add(path)

    if not os.path.exists(path):
         # Check adjacent to OneNoise/utils/helpers.py as a last resort (might happen if called from script in utils)
         utils_rel_path = os.path.join(os.path.dirname(__file__), path)
         if os.path.exists(utils_rel_path):
             path = utils_rel_path
         else:
             raise FileNotFoundError(f"Config file not found. Checked: '{path}' (resolved from '{original_path}') and '{utils_rel_path}'")

    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        return {}

    base_cfg = {}
    if '_base_' in cfg:
        bases = cfg['_base_']
        if isinstance(bases, str):
            bases = [bases]
        # Critical change: Resolve base paths relative to the *directory of the current config file* (path)
        current_config_dir = os.path.dirname(path)
        # Need project root again for resolving cfgs/-prefixed base paths
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        for base_path_rel in bases:
             # Check if the base path looks like it's already project-root relative
             if base_path_rel.startswith('cfgs/'):
                 base_full_path = os.path.normpath(os.path.join(project_root, base_path_rel))
             else:
                 # Otherwise, assume relative to the current config's directory
                 base_full_path = os.path.normpath(os.path.join(current_config_dir, base_path_rel))
             # print(f"DEBUG: Loading base config '{base_path_rel}' resolved to '{base_full_path}'")
             base_cfg_part = load_config_recursive(base_full_path, loaded_files)
             # Correctly merge nested dictionaries
             def merge_dicts(target, source):
                for key, value in source.items():
                    if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                        merge_dicts(target[key], value)
                    else:
                        target[key] = value
                return target
             base_cfg = merge_dicts(base_cfg, base_cfg_part)
        del cfg['_base_']

    # Merge base config with current config (current overrides base)
    def merge_final(base, overlay):
         for key, value in overlay.items():
             if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                 merge_final(base[key], value)
             else:
                 base[key] = value
         return base

    final_cfg = merge_final(base_cfg, cfg)
    return final_cfg

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

@torch.no_grad()
def load_infd_ae_components(ae_config_path, ae_checkpoint_path, device):
    """Loads the INFD AE decoder and renderer from config and checkpoint.

    Args:
        ae_config_path (str): Path to the AE model config YAML.
        ae_checkpoint_path (str): Path to the AE model checkpoint .pth.
        device (torch.device): The device to load the components onto.

    Returns:
        tuple: (decoder, renderer, quantizer) loaded, frozen, and on the specified device.
    """
    print("INFO: Loading INFD AE components...")
    if ae_config_path is None or ae_checkpoint_path is None:
        raise ValueError("AE config path and checkpoint path are required.")

    # Assumes OneNoise is a subdirectory of the INFD project root
    one_noise_utils_dir = os.path.dirname(__file__)
    one_noise_dir = os.path.dirname(one_noise_utils_dir)
    project_root = os.path.dirname(one_noise_dir) # Should be the infd directory
    infd_utils_dir = os.path.join(project_root, 'utils') # Path to infd/utils

    # Add only the project root to the beginning of sys.path
    if project_root not in sys.path:
        print(f"Adding {project_root} to sys.path for INFD model loading")
        sys.path.insert(0, project_root)

    try:
        # Attempt the import *after* potentially modifying sys.path
        import models
        # We don't need the specific check below anymore, subsequent imports will handle it.
        # import utils.geometry
    except ImportError as e:
        print(f"Error: Could not import 'models' or 'utils.geometry' from INFD project root at {project_root}")
        print("Please ensure you are running from the INFD directory and OneNoise is a subdirectory.")
        # Re-add the raise to stop execution if import fails
        raise e

    print(f"  Loading AE config from: {ae_config_path}")
    ae_cfg_dict = load_config_recursive(ae_config_path)

    print("  Instantiating AE model structure...")
    ae_model_config_dict = ae_cfg_dict['model']
    # Instantiate on CPU first to avoid potential CUDA OOM if checkpoint is large
    ae_model = models.make(ae_model_config_dict).cpu()

    print(f"  Loading AE checkpoint from: {ae_checkpoint_path}")
    # Resolve potential relative path for checkpoint
    checkpoint_path = ae_checkpoint_path
    if not os.path.isabs(checkpoint_path) and not os.path.exists(checkpoint_path):
         # Try resolving relative to project root (assuming OneNoise parent is root)
         proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
         proj_rel_path = os.path.join(proj_root, checkpoint_path)
         if os.path.exists(proj_rel_path):
              checkpoint_path = proj_rel_path
              print(f"    (Resolved relative path to: {checkpoint_path})")
         else:
             # Try resolving relative to INFD root as another guess
             infd_rel_path = os.path.join(project_root, checkpoint_path)
             if os.path.exists(infd_rel_path):
                 checkpoint_path = infd_rel_path
                 print(f"    (Resolved relative path to: {checkpoint_path})")
             else:
                  # Try resolving relative to OneNoise root as final guess
                  one_noise_rel_path = os.path.join(project_root, checkpoint_path)
                  if os.path.exists(one_noise_rel_path):
                       checkpoint_path = one_noise_rel_path
                       print(f"    (Resolved relative path to: {checkpoint_path})")
                  else:
                     raise FileNotFoundError(f"AE Checkpoint file not found at {ae_checkpoint_path}, {proj_rel_path}, {infd_rel_path} or {one_noise_rel_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract state dict (handle potential nesting and 'module.' prefix)
    if 'model' in checkpoint and isinstance(checkpoint['model'], dict) and 'sd' in checkpoint['model']:
        state_dict = checkpoint['model']['sd']
        print("    (Extracting state dict from checkpoint['model']['sd'])")
    elif 'state_dict' in checkpoint:
         state_dict = checkpoint['state_dict']
         print("    (Extracting state dict from checkpoint['state_dict'])")
    else:
        keys_found = list(checkpoint.keys())
        raise KeyError(f"Could not find AE state dict. Found keys: {keys_found}")

    if state_dict and list(state_dict.keys())[0].startswith('module.'):
        print("    (Removing 'module.' prefix from state dict keys)")
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    # Load the full AE state dict
    ae_model.load_state_dict(state_dict)
    print("  AE Checkpoint loaded successfully into temporary model.")

    infd_decoder = ae_model.decoder
    infd_renderer = ae_model.renderer
    infd_quantizer = getattr(ae_model, 'quantizer', None)
    if infd_quantizer is not None:
        print("  AE Quantizer found and extracted.")
        infd_quantizer.to(device)
        infd_quantizer.eval()
        infd_quantizer.requires_grad_(False)
    else:
        print("AE Quantizer not found (or not used by this AE model).")

    infd_decoder.to(device)
    infd_renderer.to(device)

    infd_decoder.eval()
    infd_renderer.eval()
    infd_decoder.requires_grad_(False)
    infd_renderer.requires_grad_(False)

    print("  INFD Decoder and Renderer extracted, frozen, and moved to device.")
    del ae_model
    del state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return infd_decoder, infd_renderer, infd_quantizer
