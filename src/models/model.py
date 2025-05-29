import re
from typing import Optional, List, Tuple
import torch
from torch.autograd import grad
from torch import nn
from torch_scatter import scatter
from pytorch_lightning.utilities import rank_zero_warn


import warnings
import traceback
import os
from .SPHNet import SPHNet,Pair_construction_layer

from .lsrm.lsrm_modules import Visnorm_shared_LSRMNorm2_2branchSerial

from .utils import construct_o3irrps, construct_o3irrps_base


# from torchmdnet.models.torchmd_norm import TorchMD_Norm

def create_model(config, prior_model=None, mean=None, std=None):
    bottle_hidden_size = 32 # fii, fij dimension, 32*0e+1e*1e+...+32*order e
    basis = config["basis"]

    enable_hami = config["enable_hami"]
    enable_energy = config["enable_energy"]
    enable_forces = config["enable_forces"]
    enable_symmetry = config["enable_symmetry"]
    model_backbone_cfg = config["model"]
    
    
    model_backbone_name = config["model_backbone"]
    outputmodel_name = config["output_model"]
    
    cut_off = config["cutoff_upper"]
    model_backbone_cfg['use_sparse_tp']=config['use_sparse_tp']
    
    model_type = None
    if model_backbone_name.startswith(
        ("SPHNet")
         ):
        model_type = "TP"
        representation_model = eval(model_backbone_name)(**model_backbone_cfg)
    else:
        raise ValueError(f'Unknown architecture: {model_backbone_name}')

    hami_model = None
    if enable_hami:
        hami_func = None
        order = 6 if basis=="def2-tzvp" else 4
        hami_config = { "irrep_in_node":representation_model.irreps_node_embedding,
                        "order":order,
                        "num_layer":config["hami_model"]["num_layer"],
                        "pyscf_basis_name":basis,
                        "max_radius_cutoff":config["hami_model"]["max_radius_cutoff"],
                        "use_sparse_tp":config['use_sparse_tp'],
                        "radius_embed_dim": config["hami_model"]['radius_embed_dim'],
                        "bottle_hidden_size": config["hami_model"]['bottle_hidden_size'],
                        "ckpt_path": os.path.join(config['ckpt_path'],config['job_id'])}

        head_name = 'Pair_construction_layer'
        hami_func = eval(head_name)
        if config["hami_model"]["irreps_edge_embedding"]:
            hami_config.update({"irreps_edge_embedding":config["hami_model"]["irreps_edge_embedding"]
                    })
        else:
            hami_config.update({"irreps_edge_embedding":
                construct_o3irrps_base(bottle_hidden_size,order)})
        hami_config.update({"sparsity":config["sparsity"]})

        hami_config['steps_1epoch']=int(config['dataset_size']*config['train_ratio']/config['batch_size'])
        hami_model = hami_func(**hami_config)
        
        
    # combine representation and output network
    model = UnifiedModel(
        representation_model,
        hami_model = hami_model,
        enable_hami=enable_hami,
        enable_energy=enable_energy,
        mean=mean,
        std=std,
        enable_forces=enable_forces,
        output_model_noise=None,
        # position_noise_scale=args['position_noise_scale'],

        
    )
    # print(model)
    return model


def load_model(filepath, args=None, device="cpu", mean=None, std=None, **kwargs):
    ckpt = torch.load(filepath, map_location="cpu")
    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if not key in args:
            warnings.warn(f'Unknown hyperparameter: {key}={value}')
        args[key] = value

    model = create_model(args)

    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
    loading_return = model.load_state_dict(state_dict, strict=False)
    
    if len(loading_return.unexpected_keys) > 0:
        # Should only happen if not applying denoising during fine-tuning.
        assert all(("output_model_noise" in k or "pos_normalizer" in k) for k in loading_return.unexpected_keys)
    assert len(loading_return.missing_keys) == 0, f"Missing keys: {loading_return.missing_keys}"

    if mean:
        model.mean = mean
    if std:
        model.std = std

    return model.to(device)


class UnifiedModel(nn.Module):
    def __init__(
        self,
        representation_model,
        output_model = None,
        hami_model = None,
        prior_model=None,
        mean=None,
        std=None,
        output_model_noise=None,
        position_noise_scale=0.,
        enable_energy=False,
        enable_forces = False,
        enable_hami = False
    ):
        super(UnifiedModel, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model

        self.prior_model = prior_model
        # if not output_model.allow_prior_model and prior_model is not None:
        #     self.prior_model = None
        #     rank_zero_warn(
        #         (
        #             "Prior model was given but the output model does "
        #             "not allow prior models. Dropping the prior model."
        #         )
        #     )
        self.hami_model = hami_model
        self.output_model = output_model
        self.enable_energy = enable_energy
        self.enable_forces = enable_forces
        self.enable_hami = enable_hami
        self.output_model_noise = output_model_noise        
        self.position_noise_scale = position_noise_scale

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)

        if self.position_noise_scale > 0:
            self.pos_normalizer = AccumulatedNormalization(accumulator_shape=(3,))

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        if self.output_model is not None:
            self.output_model.reset_parameters()
        if self.hami_model is not None:
            self.hami_model.reset_parameters()
            
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(self, batch_data):
        pos = batch_data["pos"]
        batch = batch_data["batch"]
        batch_data["atomic_numbers"] = batch_data["atomic_numbers"].reshape(-1)
        z = batch_data["atomic_numbers"]
        assert z.dim() == 1 and (z.dtype == torch.long or z.dtype == torch.int32)

        if self.enable_forces:
            pos.requires_grad_(True)

        # run the potentially wrapped representation model
        batch_data = self.representation_model(batch_data)
        # data["node_embedding"]
        # data["node_vec"]
        # data["fii"] for Hamiltanian
        # data["fij"] for Hamiltanian


        # this means that the model will only be used to predict hamiltonian matrix
        if self.enable_hami:
            batch_data = self.hami_model(batch_data)
        # # predict noise
        # noise_pred = None
        # if self.output_model_noise is not None:
        #     noise_pred = self.output_model_noise.pre_reduce(x, v, z, pos, batch) 
        #     data["pred_noise"] = noise_pred
        
        if not self.output_model:
            return batch_data 
        
        if self.enable_energy or self.enable_forces:
            # apply the output network
            batch_data = self.output_model.pre_reduce(batch_data)

            pred_energy = batch_data["pred_energy"]
            # # apply prior model
            # if self.prior_model is not None:
            #     pred_energy = self.prior_model(pred_energy, z, pos, batch)
            # aggregate atoms
            pred_energy = scatter(pred_energy, batch, dim=0, reduce="add")
            # shift by data mean
            # scale by data standard deviation
            if self.std is not None:pred_energy = pred_energy * self.std
            if self.mean is not None:pred_energy = pred_energy + self.mean
            batch_data["pred_energy"] = pred_energy

            # # apply output model after reduction
            # out = self.output_model.post_reduce(out)

        # compute gradients with respect to coordinates
        if self.enable_forces:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(pred_energy)]
            dy = -1 * grad(
                [pred_energy],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if dy is None:
                raise RuntimeError("Autograd returned None for the force prediction.")
            batch_data["pred_forces"] = dy
        return batch_data

class AccumulatedNormalization(nn.Module):
    """Running normalization of a tensor."""
    def __init__(self, accumulator_shape: Tuple[int, ...], epsilon: float = 1e-8):
        super(AccumulatedNormalization, self).__init__()

        self._epsilon = epsilon
        self.register_buffer("acc_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_squared_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_count", torch.zeros((1,)))
        self.register_buffer("num_accumulations", torch.zeros((1,)))

    def update_statistics(self, batch: torch.Tensor):
        batch_size = batch.shape[0]
        self.acc_sum += batch.sum(dim=0)
        self.acc_squared_sum += batch.pow(2).sum(dim=0)
        self.acc_count += batch_size
        self.num_accumulations += 1

    @property
    def acc_count_safe(self):
        return self.acc_count.clamp(min=1)

    @property
    def mean(self):
        return self.acc_sum / self.acc_count_safe

    @property
    def std(self):
        return torch.sqrt(
            (self.acc_squared_sum / self.acc_count_safe) - self.mean.pow(2)
        ).clamp(min=self._epsilon)

    def forward(self, batch: torch.Tensor):
        if self.training:
            self.update_statistics(batch)
        return ((batch - self.mean) / self.std)

