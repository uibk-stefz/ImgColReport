from argparse import ArgumentParser
import wandb
import pytorch_lightning as pl
from dataset import ColorizationDataset, make_dataloaders
from model import ColorDiffusion
from utils import get_device, load_default_configs
from pytorch_lightning.loggers import WandbLogger
from denoising import Unet, Encoder
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log", default=False)
    parser.add_argument("--cpu-only", default=False)
    parser.add_argument("--dataset", default="./img_align_celeba", help="Path to unzipped dataset")
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()
    print(args)

    torch.set_float32_matmul_precision("medium")

    enc_config, unet_config, colordiff_config = load_default_configs()
    train_dl, val_dl = make_dataloaders(args.dataset, colordiff_config, num_workers=4, limit=35000)
    colordiff_config["sample"] = False
    colordiff_config["should_log"] = args.log

    encoder = Encoder(**enc_config)
    unet = Unet(**unet_config)
    
    """
    if args.ckpt is not None:
        print(f"Resuming training from checkpoint: {args.ckpt}")
        model = ColorDiffusion.load_from_checkpoint(
            args.ckpt, 
            strict=True, 
            unet=unet, 
            encoder=encoder, 
            train_dl=train_dl, 
            val_dl=val_dl, 
            **colordiff_config
            )
    else:
        model = ColorDiffusion(unet=unet,
                               encoder=encoder, 
                               train_dl=train_dl,
                               val_dl=val_dl, 
                               **colordiff_config)
    """
                               
    model = ColorDiffusion(unet=unet,
                               encoder=encoder, 
                               train_dl=train_dl,
                               val_dl=val_dl, 
                               **colordiff_config)
                               
    if args.log:
        wandb_logger = WandbLogger(project="Color_diffusion_v2")
        wandb_logger.watch(unet)
        wandb_logger.experiment.config.update(colordiff_config)
        wandb_logger.experiment.config.update(unet_config)
    # ckpt_callback = ModelCheckpoint(every_n_train_steps=300, save_top_k=2, save_last=True, monitor="val_loss")

    trainer = pl.Trainer(accelerator="gpu", 
                        max_epochs=colordiff_config["epochs"],
                        logger=wandb_logger if args.log else None, 
                        num_sanity_val_steps=1,
                        devices= "auto",
                        log_every_n_steps=3,
                        profiler="simple" if args.log else None,
                        accumulate_grad_batches=colordiff_config["accumulate_grad_batches"]
                        )
                        
    if args.ckpt is not None:                    
        trainer.fit(model, train_dl, val_dl, ckpt_path = args.ckpt)
    else:
        trainer.fit(model, train_dl, val_dl)
        