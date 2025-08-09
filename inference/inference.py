import os
import math
import torch
import random
import torch.nn.functional as F

from ema_pytorch import EMA
from inference.inference_helpers import slerp
from network.diffusion import load_diffusion_model, extract
from torchvision.utils import save_image, make_grid
from torchvision.io import read_image
from on_utils.helpers import seed_everything, count_parameters, load_infd_ae_components
from scipy.ndimage import distance_transform_edt
from network.helpers import exists, default, identity

# Import the INFD utility for creating coord and cell grids
from utils.geometry import make_coord_cell_grid

from config.noise_config import noise_types, noise_aliases, param_names, ntype_to_params, ntype_to_params_map

def preproc_mask(mask, blending_factor=1.0, H=None, W=None, invert=False):
    '''
    Preprocesses a binary mask for blending between two noise types.
    mask:               binary mask (file path or (H,W) tensor) -- optionally can be a [0,1] tensor (smooth mask)
    blending_factor:    how gradual the blending should be -- closer to zero makes the blending more abrupt, closer to one makes it more gradual
                        good values are typically in [0.2, 0.4] range. This can be specified per pixel as well: (H, W) tensor.
    H, W:               height and width of the output mask
    invert:             whether to invert the mask
    '''
    if isinstance(mask, torch.Tensor):
        # if mask is not a binary mask (already smooth), then we dont need to do much preprocessing
        if len(mask.unique()) != 2:
            return mask.float().pow(blending_factor)
        
    if isinstance(mask, str):
        mask = read_image(mask).float()

    mask = mask / 255.0
    mask = mask.mean(dim = 0, keepdim = True) # (1, H, W)
    mask = (mask > 0.5).float()
    if invert: mask = 1. - mask
    if H and W:
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0), (H, W), mode='nearest').squeeze(0)

    dst = distance_transform_edt(mask.cpu().numpy())
    if dst.max() > 0: # normalize distance transform
        dst = dst / dst.max()
    dst = torch.from_numpy(dst).float().cuda()
    dst = dst.pow(blending_factor)
    return dst

# convenience functions for creating noise configurations:
def cls_idx(ntype):
    return noise_types.index(noise_aliases[ntype])

def param_idx(param):
    return param_names.index(param)

def dict2cond(dict, H=1, W=1):
    # converts a dictionary of noise parameters to conditioning tensors for the model
    noise_idx = cls_idx(dict['cls'])

    sbsparams = torch.zeros(1, len(param_names))
    for k, v in dict['sbsparams'].items():
        sbsparams[0, param_idx(k)] = v

    classes = torch.zeros(1, len(noise_types))
    classes[0, noise_idx] = 1.

    if H > 1 and W > 1:
        sbsparams = sbsparams.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        classes = classes.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

    return sbsparams, classes
    
def sample_parameters(ntype):
    # randomly sample noise parameters for a given noise type
    params = ntype_to_params_map[ntype] # list of parameters to sample
    dict_ = {
        'cls': ntype,
        'sbsparams': {
            p: random.random() for p in params
        }
    }
    return dict_

def default(val, d):
    if exists(val):
        return val
    return d

@torch.no_grad()
def decode_latent(latent: torch.Tensor, decoder, renderer, quantizer,
                  target_H: int, target_W: int, device: torch.device,
                  substance_params = None,
                  classes = None):
    """Decodes a latent tensor to an image using the pre-trained INFD components.

    The decoder can optionally receive spatial conditioning in the form of
    `substance_params` (noise parameter grid) and `classes` (one-hot class grid).
    These are the exact tensors that the diffusion U-Net was conditioned on
    earlier in the pipeline, so passing them through ensures the SPADE blocks
    inside the decoder receive identical information.
    """
    if decoder is None or renderer is None:
        raise ValueError("Decoder and Renderer must be provided for latent decoding.")
    
    latent_post_conv = decoder[0](latent)
    
    # VQ layer from INFD section 4.1
    if quantizer is not None:
        quantized_latents, _loss, _info = quantizer(latent_post_conv)
        latents_for_decoder = quantized_latents
    else:
        latents_for_decoder = latent_post_conv
    
    B = latents_for_decoder.shape[0]

    coord, cell = make_coord_cell_grid(shape=(target_H, target_W), device=device, bs=B)
    
    features = decoder[1](latents_for_decoder,
                       classes=classes,
                       substance_params=substance_params)
    img = renderer(features, coord=coord, cell=cell)  # renderer output in [-1, 1]
    img = (img + 1) / 2
    img = img.clamp(0., 1.)
    return img

class Inference():
    def __init__(self, config, model=None, device=None, save_dir=None, seed=None,
                 is_latent_diffusion=False, infd_decoder=None, infd_renderer=None, infd_quantizer=None) -> None:
        self.config = config
        self.is_latent_diffusion = is_latent_diffusion
        self.infd_decoder = infd_decoder
        self.infd_renderer = infd_renderer
        self.infd_quantizer = infd_quantizer

        if self.is_latent_diffusion and (self.infd_decoder is None or self.infd_renderer is None):
            raise ValueError("Inference created in latent mode, but infd_decoder or infd_renderer is missing.")

        # number of noise types and parameters:
        self.num_types = len(noise_types)
        self.num_params = len(param_names)

        self.steps = config.sample_timesteps

        self.dev = default(device, torch.device('cuda:0'))

        if exists(model):
            self.model = model
        else:
            self.model = load_diffusion_model(config, device=self.dev)

        if isinstance(self.model, EMA):
            self.model = self.model.ema_model

        self.emb_dim = self.model.model.cond_dim

        self.out_dir = None
        if self.config.exp_name is not None:
            self.out_dir = os.path.join(config.out_dir, config.exp_name, 'out')
            os.makedirs(self.out_dir, exist_ok=True)
        self.save_dir = default(save_dir, lambda x: os.path.join(self.out_dir, x) if self.out_dir else x)

        if exists(seed):
            seed_everything(seed)
    
    def sample(self, params, classes, cond_scale=3., noise=None, **kwargs):
        params, classes = params.to(self.dev), classes.to(self.dev) 
        
        if noise is not None:
            noise = noise.to(self.dev)
            if noise.shape[1] != self.model.channels:
                raise ValueError(f"Provided noise has {noise.shape[1]} channels, but model expects {self.model.channels} channels.")

        output = self.model.sample(params, cond_scale=cond_scale, classes=classes, noise=noise,
                                    return_latent=self.is_latent_diffusion, **kwargs) 

        if self.is_latent_diffusion:
            if self.infd_decoder is None or self.infd_renderer is None:
                raise ValueError("INFD decoder or renderer is not available for latent diffusion mode in sample method.")
            

            latents_for_pipeline = output # TODO: do I need to apply tanh here?
            
            render_H = self.config.image_size
            render_W = self.config.image_size
            # Pass the same conditioning used by the diffusion model to the decoder
            return decode_latent(
                latents_for_pipeline,
                self.infd_decoder,
                self.infd_renderer,
                self.infd_quantizer,
                render_H,
                render_W,
                self.dev,
                substance_params=params,
                classes=classes,
            )
        else:
            return output

    def generate(self, sbsparams, classes, class_emb=None, noise=None, filename=None):
        '''
        The primary function for generating samples from the model.

        sbsparams:              (B, num_params, H, W) or (B, num_params) tensor of noise parameters
        classes:                (B, num_types, H, W) or (B, num_types) tensor of one-hot class labels
        class_emb (optional):   (B, emb_dim, H, W) tensor of class embeddings, if provided `classes` is ignored
        noise (optional):       (B, self.model.channels, H, W) tensor of gaussian noise, if not provided random noise is used
        filename (optional):    filepath to save the generated image

        Returns:
        img:                    (B, C, H, W) tensor of generated images (decoded if in latent diffusion mode)
        '''
        B = sbsparams.shape[0]
        # H_cond, W_cond are from sbsparams, e.g., 256x256, used for conditioning spatial size
        if len(sbsparams.shape) == 4: # sbsparams is (B, num_params, H_cond, W_cond)
            H_cond, W_cond = sbsparams.shape[-2:]
        elif len(sbsparams.shape) == 2: # sbsparams is (B, num_params)
            H_cond, W_cond = 256, 256 # Default conditioning spatial size
        else:
            raise ValueError(f"sbsparams has an unexpected shape: {sbsparams.shape}")

        # Determine the actual noise dimensions for the diffusion process
        # For latent diffusion, this is self.model.image_size (e.g., 64x64)
        # For pixel-space diffusion, this is H_cond, W_cond (e.g., 256x256)
        process_H = self.model.image_size if self.is_latent_diffusion else H_cond
        process_W = self.model.image_size if self.is_latent_diffusion else W_cond

        actual_noise_to_use: torch.Tensor
        if noise is None:
            actual_noise_to_use = torch.randn(B, self.model.channels, process_H, process_W, device=self.dev)
        else:
            # Validate channel dimension of provided noise
            if noise.shape[1] != self.model.channels:
                raise ValueError(f"Provided noise has {noise.shape[1]} channels, but model expects {self.model.channels} channels.")
            
            # Validate batch size
            if noise.shape[0] != B:
                raise ValueError(f"Provided noise batch size {noise.shape[0]} inconsistent with sbsparams batch size {B}.")

            # If spatial dimensions of provided noise don't match process_H, process_W, resize it.
            # This handles the case where full_grid provides noise at (256,256) but latent process needs (64,64)
            if noise.shape[2] != process_H or noise.shape[3] != process_W:
                # print(f"Warning: Provided noise spatial dimensions ({noise.shape[2]},{noise.shape[3]}) mismatch process dimensions ({process_H},{process_W}). Resizing.")
                actual_noise_to_use = F.interpolate(noise, size=(process_H, process_W), mode='bilinear', align_corners=False, antialias=True)
            else:
                actual_noise_to_use = noise
        
        output_from_sampler = self.model.ddim_sample_fast(
            params=sbsparams,
            classes=classes,
            noise=actual_noise_to_use,
            class_emb=class_emb,
        )

        if self.is_latent_diffusion:
            if self.infd_decoder is None or self.infd_renderer is None:
                raise ValueError("INFD decoder or renderer is not available for latent diffusion mode in generate method.")
            
            latents_for_pipeline = output_from_sampler

            img_final = decode_latent(
                latents_for_pipeline,
                self.infd_decoder,
                self.infd_renderer,
                self.infd_quantizer,
                H_cond,
                W_cond,
                self.dev,
                substance_params=sbsparams,
                classes=classes,
            )
        else:
            img_final = output_from_sampler

        if exists(filename):
            save_image(img_final, self.save_dir(f'{filename}.png'))

        return img_final

    def get_class_embedding(self, dict_or_idx):
        '''
        Returns the class embedding for a noise type, provided by either the noise type index or a dictionary containing the noise type.
        '''

        if isinstance(dict_or_idx, int):
            noise_idx = dict_or_idx
        else:
            noise_idx = cls_idx(dict_or_idx['cls'])
        
        return self.model.model.classes_emb.weight[noise_idx].unsqueeze(0) # (1, C)

    def full_grid(self, H, W, num_samples, filename=None):
        '''
        Generates a large grid of samples from all noise types with spatially uniform parameters (no blending/interpolation).

        H, W:           height and width of each sample
        num_samples:    number of samples to generate for each noise type
        filename:       filename to save the grid to
        '''

        B = 2 # Batch size, TODO: make this a parameter

        noise = torch.randn(num_samples, self.model.channels, H, W).repeat(self.num_types, 1, 1, 1).to(self.dev)

        all_sbsparams = []
        all_cls_idx = []

        # iterate over all noise types `num_samples` times, and sample random parameters
        for i in range(self.num_types):
            for j in range(0, num_samples):
                my_param_names = ntype_to_params[i]
                param_idxs = [param_names.index(p) for p in my_param_names]
                rand_params = torch.rand(len(my_param_names))
                sbsparams = torch.zeros(self.num_params)
                sbsparams[param_idxs] = rand_params
                classes = i
                
                all_sbsparams += [sbsparams]
                all_cls_idx += [classes]
        
        all_sbsparams = torch.stack(all_sbsparams, dim=0).to(self.dev)
        all_cls_idx = torch.tensor(all_cls_idx).to(self.dev)

        imgs = []
        for i in range(0, num_samples * self.num_types, B):
            # Expand parameters and classes to be of shape (B, C, H, W)
            sbsparams = all_sbsparams[i:i+B].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            cls_embs = self.model.model.classes_emb(all_cls_idx[i:i+B]).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            print(sbsparams.shape, cls_embs.shape)
            imgs += [self.generate(sbsparams, None, class_emb=cls_embs, noise=noise[i:i+B])]
        
        imgs = torch.cat(imgs, dim=0)

        if exists(filename):
            imgs = make_grid(imgs, nrow=num_samples)
            save_image(imgs, self.save_dir(f'{filename}.png'))
        
        return imgs

    def slerp_mask(self, mask, dict1=None, dict2=None, H=256, W=256, blending_factor=0.25, filename=None):
        '''
        Generates an interpolation between two noises using a provided blending mask.

        mask:               binary blending mask (file path or (H,W) tensor)
        dict1, dict2:       dictionaries containing noise parameters and noise types
        H, W:               height and width of the output image
        blending_factor:    how gradual the blending should be -- closer to zero makes the blending more abrupt, closer to one makes it more gradual
        filename:           output filepath
        '''
        dist = preproc_mask(mask, blending_factor=blending_factor, H=H, W=W, invert=False) # distance transform (1, H, W)
        print(f"[Inference.slerp_mask] dist tensor min: {dist.min().item():.4f}, max: {dist.max().item():.4f}, mean: {dist.mean().item():.4f}, shape: {dist.shape}") # DEBUG PRINT

        sbsparams1, classes1 = dict2cond(dict1, H, W)
        sbsparams2, classes2 = dict2cond(dict2, H, W)
        sbsparams1 = sbsparams1.cuda()
        sbsparams2 = sbsparams2.cuda()
        classes1 = classes1.cuda()
        classes2 = classes2.cuda()
        
        cls_emb1 = self.get_class_embedding(dict1) # (1,32)
        cls_emb2 = self.get_class_embedding(dict2) # (1,32)

        sbsparams = sbsparams1 * (1 - dist.unsqueeze(1)) + sbsparams2 * dist.unsqueeze(1)
        classes = classes1 * (1 - dist.unsqueeze(1)) + classes2 * dist.unsqueeze(1)
        
        ts = dist.flatten()
        cls_emb_slerp = slerp(ts, cls_emb1, cls_emb2) # (H*W, 32)
        cls_emb_slerp = cls_emb_slerp.reshape(H, W, -1).permute(2, 0, 1).unsqueeze(0)

        img = self.generate(
            sbsparams=sbsparams,
            classes=classes,
            class_emb=cls_emb_slerp,
            filename=filename
        )

        if exists(filename):
            save_image(img, self.save_dir(f'{filename}.png'))

        return img

    #TODO: figure out if anything needs to change here for latent diffusion
    def slerp_horizontal(self, dict1=None, dict2=None, H=256, W=512, filename=None):
        '''
        Generates a horizontal blend between two noise types.
        '''

        # horizontal blending map:
        mask_linear_horiz = torch.linspace(0, 1, W).unsqueeze(0).expand(H, -1).unsqueeze(0).cuda()

        sbsparams1, _ = dict2cond(dict1, H, W)
        sbsparams2, _ = dict2cond(dict2, H, W)
        sbsparams1 = sbsparams1.cuda()
        sbsparams2 = sbsparams2.cuda()
        
        cls_emb1 = self.get_class_embedding(dict1) # (1,32)
        cls_emb2 = self.get_class_embedding(dict2) # (1,32)

        # interpolate noise parameters
        sbsparams = sbsparams1 * (1 - mask_linear_horiz) + sbsparams2 * mask_linear_horiz

        # interpolate class embeddings (spherically)
        ts = torch.linspace(0, 1, W).cuda()
        cls_emb_slerp = slerp(ts, cls_emb1, cls_emb2) # (W, 32)
        cls_emb_slerp = cls_emb_slerp.transpose(0,1).unsqueeze(-2).unsqueeze(0).expand(-1, -1, H, W)

        img = self.generate(sbsparams, None, class_emb=cls_emb_slerp)

        if exists(filename):
            save_image(img, self.save_dir(f'{filename}.png'))
        
        return img
    
    def sample_sphere(self, H=256, W=256, filename=None):
        '''
        Generates an image by sampling a random point on the embedding hypersphere and 
        treating that as the class embedding.

        Note: this doesn't reliably produce nice images (we wouldn't really expect it to),
        but it's fun to play with :)
        '''
        mean_norm = self.model.model.classes_emb.weight.norm(dim=-1, keepdim=True).mean()

        cls_emb = torch.randn(1, 32).to(self.dev)
        cls_emb = (cls_emb / cls_emb.norm(dim=-1, keepdim=True)) * mean_norm
        cls_emb = cls_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        sbsparams = torch.zeros(1, len(param_names), H, W).to(self.dev)

        img = self.generate(sbsparams, None, class_emb=cls_emb, filename=filename)
        return img

    def class_midpoints(self, dict1, dict2, H=256, W=256, filename=None):
        '''
        Generates an image by taking the midpoint of two noise types in the embedding space.
        This should visually look like the (feature-based) average of the two noise types.
        '''
        
        cls_emb1 = self.get_class_embedding(dict1)
        cls_emb2 = self.get_class_embedding(dict2)
        midpoint = slerp(0.5, cls_emb1, cls_emb2)
        midpoint = midpoint.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        mask_linear_horiz = torch.linspace(0, 1, W).unsqueeze(0).expand(H, -1).unsqueeze(0).cuda()
        sbsparams1, _ = dict2cond(dict1, H, W)
        sbsparams2, _ = dict2cond(dict2, H, W)
        sbsparams1 = sbsparams1.cuda()
        sbsparams2 = sbsparams2.cuda()

        sbsparams = sbsparams1 * (1 - mask_linear_horiz) + sbsparams2 * mask_linear_horiz

        img = self.generate(sbsparams, None, class_emb=midpoint, filename=filename)
        return img

    def random_sample(self, H=256, W=256, filename=None):
        '''
        Generates a sample using randomly selected noise type and parameters.
        '''
        ntype = random.choice(noise_types)
        dict = sample_parameters(ntype)
        sbsparams, classes = dict2cond(dict, H, W)
        sbsparams = sbsparams.cuda()
        classes = classes.cuda()
        img = self.generate(sbsparams=sbsparams, classes=classes, filename=filename)
        return img

    def random_class_interpolations(self, H, W, nimg=16, filename=None):
        '''
        Calls `slerp_horizontal` with random noise types and parameters.
        If only one noise type is available, it interpolates between random parameters of that single type.
        '''

        imgs = []
        for i in range(nimg):
            ntype1 = random.choice(noise_types)
            
            if len(noise_types) > 1:
                ntype2 = random.choice(noise_types)
                while ntype1 == ntype2:
                    ntype2 = random.choice(noise_types)
            else:
                ntype2 = ntype1

            imgs += [self.slerp_horizontal(
                sample_parameters(ntype1),
                sample_parameters(ntype2),
                H, W
            )]
        
        imgs = torch.cat(imgs, dim=0)
        if exists(filename):
            grid = make_grid(imgs, nrow=int(math.sqrt(len(imgs))))
            save_image(grid, self.save_dir(f'{filename}.png'))
        
        return imgs
