
import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from omegaconf import MISSING, DictConfig



@dataclass
class BaseSchema:
    """This is for CLI applications that need to reuse a CLI parameter in multiple places
    in the config file. The subfields of `general` are not fixed and can be anything. It
    also allows configs to be shared between multiple workstreams; e.g. the ks_config can
    be shared between data generation, training (as a callback) and evaluation.

    Examples:
        1) You can set `general.output_dir=foobar` and in other places in the config file
        `output_dir: ${general.output_dir}`.
        2) In the default field of the config, you can set

            ```
            default:
              - general/kohn_sham/default
            ```

        which is useful for composing configs.  This will append the default.yaml file from
        config_path/general/kohn_sham/ to the current config. In other places in the config
        file, you can then call ${general.kohn_sham}.
    """
    

@dataclass
class Config(BaseSchema):
    seed: int = 0
    job_id: str = "auto"
    ckpt_path: str = '/data'
    log_dir: str = "./tmp"
    schedule: Dict[str, Any] = MISSING
    model: Dict[str, Any] = MISSING
    wandb: Dict[str, Any] = MISSING
    #########
    # trainer related config
    model_backbone:str = "SPHNet"
    output_model: str = "EquivariantScalar_viaTP"
    hami_model: Dict[str, Any] = MISSING
    use_sparse_tp: bool = False
    sparsity: float = 0.7
    num_epochs: int = 300 #number of epochs
    max_steps: int = -1 #Maximum number of gradient steps.
    batch_size: int = 32 #batch size
    inference_batch_size: Any = None
    dataloader_num_workers: int = 4
    lr: float = 1e-4
    multi_para_group: bool = False
    weight_decay: float = 0
    enable_hami: bool = False
    enable_symmetry: bool = False
    enable_energy: bool = False
    enable_forces: bool = False
    enable_energy_hami_error: bool = False
    enable_hami_orbital_energy: bool = False
    energy_weight: float = 0 #Weighting factor for energies in the loss function
    forces_weight: float = 0 #Weighting factor for forces in the loss function
    hami_weight: float = 1 #Weighting factor for hami in the loss function
    orbital_energy_weight: float = 0 #Weighting factor for orbital energy in the loss function
    energy_train_loss: str = 'mse'
    forces_train_loss: str = 'mse'
    orbital_energy_train_loss: str = 'mse'
    hami_train_loss: str = 'maemse'
    energy_val_loss: str = 'mae'
    forces_val_loss: str = 'mae'
    hami_val_loss: str = 'mae'
    ed_type: str = 'naive'
    sparse_loss: bool = False
    sparse_loss_coeff: float = 1e-3
    ngpus: int = 1
    num_nodes: int = 1
    gradient_clip_val: Any = None
    early_stopping_patience: int = 30
    val_check_interval: Any = None #follow pytorch lightning
    test_interval: int = 10 #Test interval, one test per n epochs (default = 10)
    save_interval: int = 10 #Save interval, one save per n epochs (default = 10)
    ############: Any
    #: Any data realted config
    basis: str = "def2-svp"  #when predict hamitonian, the basis need to be set
    data_name: str = "QH9"
    dataset_path: Any  = None
    index_path: Any  = None
    dataset_size: int  = -1 #the dataset size is used for debug. -1 is all data")
    train_ratio: Any = 0.8 # Percentage of samples in training set (null to use all remaining samples)
    val_ratio: Any = 0.02 # Percentage of samples in validation set (null to use all remaining samples)
    test_ratio: Any = 0.18 # Percentage of samples in test set (null to use all remaining samples)
    cutoff_lower: Any = 0.0 #Lower cutoff in model
    cutoff_upper: Any = 5.0 #Upper cutoff in model
    used_cache: bool = False
    ema_decay: float = 1.0
    precision: str = "32"
    unit: float = 1
    # nidek related
    activation: str = 'silu'
    remove_init: bool=False
    remove_atomref_energy:bool=False
    debug: bool = False
    test_energy_hami: bool = False
    test_homo_lumo_hami: bool = False
    num_sanity_val_steps: int = 0
    check_val_every_n_epoch: int = 1