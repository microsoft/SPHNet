import os
from os.path import join
from tqdm import tqdm
import torch
from torch.utils.data import Subset,DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_warn
from torch_scatter import scatter


from .utils import make_splits

from ..dataset.dataset_unified import LmdbDataset,get_data_default_config,MdbDataset
from ..dataset.utils import shard_discretizations,InMemoryDataset
from ..dataset.utils import collate_fn_unified
from omegaconf import MISSING, DictConfig
import numpy as np

class DataModule(LightningDataModule):
    def __init__(self, config:DictConfig):
        super(DataModule, self).__init__()
        # if hasattr(hparams, "__dict__"):
        #     self.save_hyperparameters(hparams.__dict__)
        # else:
        #     self.save_hyperparameters(hparams)
            
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()
        self.config = config
        self.data_name = self.config["data_name"]
        self.path = self.config["dataset_path"]
        self.seed = self.config["seed"]
        self.log_dir = self.config["log_dir"]
        self.basis = self.config["basis"]
        self.index_path = self.config["index_path"]
        self.train_dataset = None
        if self.config["inference_batch_size"] is None:
            self.config["inference_batch_size"] = self.config["batch_size"]
    def setup(self, stage):
        is_distributed = torch.distributed.is_initialized()
        num_workers = (
            self.trainer.world_size if is_distributed and self.trainer is not None else 1
            )
        worker_rank = self.trainer.global_rank if is_distributed else 0

        if stage == "fit" or stage is None or self.train_dataset is None:
            if "qh9" in self.data_name.lower() or "md17" in self.data_name.lower() or "custom" in self.data_name.lower():
                dataset = MdbDataset(path = self.path,remove_init=self.config["remove_init"])
                print("len dataset ",len(dataset))
            else:
                dataset = LmdbDataset(self.path,
                                      data_name=self.data_name,
                                        enable_hami = self.config["enable_hami"],
                                        old_blockbuild = False,
                                        basis = self.basis,
                                        remove_atomref_energy = self.config["remove_atomref_energy"],
                                        remove_init=self.config["remove_init"])

            if "qh9" in self.data_name.lower() and "stable_iid" in self.data_name.lower():
                print("qh9 stable iid split")
                index = torch.load(
                    os.path.join(self.index_path, 'processed_QH9Stable_random_12.pt')
                )
                self.idx_train = index[0]
                self.idx_val = index[1]
                self.idx_test = index[2]
            if "qh9" in self.data_name.lower() and "stable_ood" in self.data_name.lower():
                print("qh9 stable ood split")
                index = torch.load(
                    os.path.join(self.index_path, 'processed_QH9Stable_size_ood.pt')
                )
                self.idx_train = index[0]
                self.idx_val = index[1]
                self.idx_test = index[2]
            if "qh9" in self.data_name.lower() and "dynamic_mol" in self.data_name.lower():
                print("qh9 dynamic mol split")
                index = torch.load(
                    os.path.join(self.index_path, 'processed_QH9Dynamic_mol.pt')
                )
                self.idx_train = index[0]
                self.idx_val = index[1]
                self.idx_test = index[2]
            if "qh9" in self.data_name.lower() and "dynamic_geo" in self.data_name.lower():
                print("qh9 dynamic geo split")
                index = torch.load(
                    os.path.join(self.index_path, 'processed_QH9Dynamic_geometry.pt')
                )
                self.idx_train = index[0]
                self.idx_val = index[1]
                self.idx_test = index[2]  

            if "qh9" not in self.data_name.lower():
                len_dataset = min(self.config["dataset_size"],len(dataset)) if self.config["dataset_size"]!=-1 else len(dataset)
            else:
                len_dataset = max(self.idx_train.max(),self.idx_val.max(),self.idx_test.max())+1
            # if "mol" in self.data_name.lower():
            #     len_dataset = len_dataset-180

            if self.config["train_ratio"] == -1:
                _,_,train_ratio,val_ratio,test_ratio = get_data_default_config(self.data_name)
                test_ratio = 1-train_ratio-val_ratio
            elif isinstance(self.config["train_ratio"],int):
                train_ratio = self.config["train_ratio"]*1.0/len_dataset
                val_ratio = self.config["val_ratio"]*1.0/len_dataset
                test_ratio = 1-train_ratio-val_ratio
            else:
                train_ratio = self.config["train_ratio"]
                val_ratio = self.config["val_ratio"]
                test_ratio = 1-train_ratio-val_ratio    

            local_examples = range(len_dataset//num_workers*(worker_rank),len_dataset//num_workers*(worker_rank+1))

            self.dataset = None
            if self.config["used_cache"]:
                self.dataset = InMemoryDataset(dataset,indices=local_examples) if "qh9" not in self.data_name.lower() else InMemoryDataset(dataset)
            else:
                self.dataset = Subset(dataset,indices=local_examples) if "qh9" not in self.data_name.lower() else dataset
                
            if "qh9" not in self.data_name.lower():
                self.idx_train, self.idx_val, self.idx_test = make_splits(len(self.dataset),
                    train_ratio,val_ratio,test_ratio,
                    self.seed,
                    splits = None,
                ) 
            else:
                len_train = self.idx_train.shape[0]//num_workers
                len_val = self.idx_val.shape[0]//num_workers
                len_test = self.idx_test.shape[0]//num_workers
                self.idx_train = self.idx_train[range(len_train*(worker_rank),len_train*(worker_rank+1))]
                self.idx_val = self.idx_val[range(len_val*(worker_rank),len_val*(worker_rank+1))]
                self.idx_test = self.idx_test[range(len_test*(worker_rank),len_test*(worker_rank+1))]

            print(
                f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}"
            )

            self.train_dataset = Subset(self.dataset, self.idx_train)
            self.val_dataset = Subset(self.dataset, self.idx_val)
            self.test_dataset = Subset(self.dataset, self.idx_test)
            down_sample_size = int(self.idx_val.shape[0]/self.config["inference_batch_size"])
            self.val_dataset_realworld = Subset(self.dataset, self.idx_val[::down_sample_size])


    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        loaders = [self._get_dataloader(self.val_dataset, "val"),self._get_dataloader(self.val_dataset_realworld, "val")]
        return loaders

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    @property
    def atomref(self):
        if hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std
    
    def _get_num_batches(self, length: int, batch_size: int):
        is_distributed = torch.distributed.is_initialized()
        world_size = self.trainer.world_size if is_distributed and self.trainer is not None else 1
        return length // world_size // batch_size

        
        
    def _get_dataloader(self, dataset, stage): 
        
        if stage == "train":
            batch_size = self.config["batch_size"]
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.config["inference_batch_size"]
            shuffle = False
            
        num_batches = self._get_num_batches(len(dataset),batch_size)
        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config["dataloader_num_workers"],
            collate_fn = collate_fn_unified(long_cutoff_upper = 9,unit = self.config["unit"]),
            # max_num_batches=num_batches,
            drop_last=True,
            pin_memory=True,
        )

        return dl

    def _standardize(self):
        def get_energy(batch, atomref):
            if batch.y is None:
                raise ValueError("sorry, the batch.y does not exist ")

            if atomref is None:
                return batch.y.clone()

            # remove atomref energies from the target energy
            atomref_energy = scatter(atomref[batch.z], batch.batch, dim=0)
            return (batch.y.squeeze() - atomref_energy.squeeze()).clone()

        data = tqdm(
            self._get_dataloader(self.train_dataset, "val", store_dataloader=False),
            desc="computing mean and std",
        )
        try:
            # only remove atomref energies if the atomref prior is used
            atomref = self.atomref if self.config["prior_model"] == "Atomref" else None
            # extract energies from the data
            ys = torch.cat([get_energy(batch, atomref) for batch in data])
        except ValueError:
            rank_zero_warn(
                "Standardize is true but failed to compute dataset mean and "
                "standard deviation. Maybe the dataset only contains forces."
            )
            return

        # compute mean and standard deviation
        self._mean = ys.mean(dim=0)
        self._std = ys.std(dim=0)
