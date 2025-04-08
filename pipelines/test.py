import numpy as np  # sometimes needed to avoid mkl-service error
import sys
import os
import glob
import argparse
import logging
import pytorch_lightning as pl
import glob
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from pathlib import Path

import torch
import wandb
import os
# os.environ["WANDB_MODE"] = "offline"
# print(f"Current PYTHONPATH: {os.environ.get('PYTHONPATH')}")  

from datetime import datetime
import random
import hydra
from omegaconf import DictConfig, OmegaConf

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# Add src in root folder
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir, ".."))

from src.utility.hydra_config import Config
from src.training.module import LNNP
from src.training.logger import CSVLogger, get_latest_ckpt
from src.utility.callbacks import EMA
from src.training.data import DataModule

# import shutup
# shutup.please()

@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def cli(config: DictConfig) -> None:
    schema = OmegaConf.structured(Config)
    config = OmegaConf.merge(schema, config)
    OmegaConf.set_struct(config, False)

    if config.job_id == "auto":
        ct = datetime.now()
        #generate a random word like flushing river
        config.job_id = f'time{ct.year}_{ct.month}_{ct.day}_{ct.hour}_{ct.minute}_{ct.second}'


    print(config)
    config.log_dir = os.path.join(config.log_dir, config.job_id)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
        # Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    with open(config.log_dir+"/config.yaml", 'w') as file:OmegaConf.save(config=config, f=file.name)


    config.batch_size = config.batch_size // config.ngpus if config.ngpus > 1 else config.batch_size
    if config.inference_batch_size is None:
        config.inference_batch_size = config.batch_size

    # save_argparse(config, os.path.join(config.log_dir, "input.yaml"),[])
    main(config)


def main(config):
    print(config)
    
    pl.seed_everything(config.seed, workers=True)
    # initialize data module
    data = DataModule(config)

    # initialize lightning module
    model = LNNP(config)
    # model = torch_geometric.compile(model, dynamic=True)

    callbacks = []
    callbacks.append(EarlyStopping("val_loss", patience=config.early_stopping_patience))
    callbacks.append(ModelCheckpoint(
        dirpath=config.log_dir,
        monitor="val_loss",
        save_top_k=5,  # -1 to save all
        every_n_epochs =config.save_interval,
        filename="{step}-{epoch}-{val_loss}", #{val_loss:.4f}
    ))
    latest_file = get_latest_ckpt(config.log_dir)
    print("latest_file is: ", latest_file)
    
    if config.ema_decay!=1:
        callbacks.append(EMA(decay=config.ema_decay))
        
        
    # logger    
    tb_logger = pl.loggers.TensorBoardLogger(
        config.log_dir, name="tensorbord", version="", default_hp_metric=False
    )
    csv_logger = CSVLogger(config.log_dir, name="", version="")
    # wandb is project/group/name format to save all the log
    wandb_logger = WandbLogger(
                               entity=None,
                               project=config.wandb.wandb_project,
                               group = config.wandb.wandb_group,
                               name=config.job_id, 
                               settings=wandb.Settings(start_method='fork', code_dir="."),
                               )

    # login into wandb
    @rank_zero_only
    def log_code():
        if config.wandb.open:
            wandb.login(key=config.wandb.wandb_api_key, relogin=True)
            wandb_logger.experiment # runs wandb.init, so then code can be logged next
            wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"))
    log_code()

    if config.precision == '32':
        #ENABLE TENSOR CORES
       torch.set_float32_matmul_precision('high') # set from highest to high
    # ddp_plugin = None
    strategy=DDPStrategy(find_unused_parameters=True)
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        max_steps=config.max_steps,
        devices=list(range(config.ngpus)),
        num_nodes=config.num_nodes,
        default_root_dir=config.log_dir,
        callbacks=callbacks,
        logger=[tb_logger, wandb_logger,csv_logger], 
        val_check_interval = config.val_check_interval,
        check_val_every_n_epoch = config.check_val_every_n_epoch,
        precision=config.precision,
        strategy=strategy,
        gradient_clip_val = config.gradient_clip_val,
        use_distributed_sampler = False, # Manual sharding done inside datamodule.
        num_sanity_val_steps = config.num_sanity_val_steps,
        # accelerator=args.distributed_backend,
        # resume_from_checkpoint='/mnt/data/logs/lsr-equiv2-sym-ocploss/step=23697-epoch=81-val_loss=0.0628-test_loss=0.0000-train_per_step=0.0000.ckpt',
        # reload_dataloaders_every_epoch=False,
        # plugins=[ddp_plugin],
    )

    # use previous ckpt if have one
    ckpt_files = glob.glob(os.path.join(config.log_dir, '*.ckpt'))  
    print(os.path.join(config.log_dir, '*.ckpt'))
    # ckpt_files = False
    if ckpt_files:  
        latest_file = max(ckpt_files, key=os.path.getctime)  
        print(f"The latest .ckpt file is: {latest_file}")  
    else:  
        raise ValueError("No .ckpt files found in the folder.")

    # run test set after completing the fit
    latest_file = get_latest_ckpt(config.log_dir)
    print(latest_file,config.log_dir)
    trainer.test(model, data,ckpt_path=latest_file)


if __name__ == "__main__":
     cli()
