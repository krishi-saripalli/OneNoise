import os
import sys

#TODO: this is ugly, need a better way to do this
# Add INFD project root to sys.path to allow imports like utils.geometry
# This assumes OneNoise/train.py is located at <project_root>/OneNoise/train.py
_one_noise_train_py_file_path = os.path.abspath(__file__)
_one_noise_dir_path = os.path.dirname(_one_noise_train_py_file_path)
_infd_project_root_path = os.path.dirname(_one_noise_dir_path)

if _infd_project_root_path not in sys.path:
    sys.path.insert(0, _infd_project_root_path)

import torch

from trainer import Trainer
from datetime import datetime
from network.diffusion import create_diffusion_model

def run(config):
    print(config)

    if config.tf32:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True 

    if config.latent_diffusion:
        config.latent_size = 64
        config.image_size = config.latent_size  # "image" size, just to satisfy the diffusion.py assertion
        config.output_size = 256 

    diffusion = create_diffusion_model(config)

    # if we are not resuming from a checkpoint, generate a new experiment name
    if config.exp_name is None and config.milestone is None:
        config.exp_name = f'{config.model_config}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'

    result_dir = os.path.join(config.out_dir, config.exp_name)

    trainer = Trainer(
        diffusion,
        config,
        results_folder=result_dir,
        train_batch_size=config.batch_size,
        train_num_steps=config.train_num_steps,
        gradient_accumulate_every=config.grad_accum,
        ema_decay=config.ema_decay,
        split_batches=False,
        precision=config.precision,
    )

    if config.milestone is not None:
        trainer.load(config.milestone)

    trainer.train()
    print('Training complete.')

if __name__ == '__main__':
    from args import arguments
    
    run(arguments)
