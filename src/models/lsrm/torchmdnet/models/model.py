import re
from typing import Optional, List, Tuple
import torch
from torch.autograd import grad
from torch import nn, Tensor
from torch_scatter import scatter
from pytorch_lightning.utilities import rank_zero_warn
from . import output_modules
from .wrappers import AtomFilter
# from . import priors
import warnings


def create_model(config, prior_model=None, mean=None, std=None,atomref = None):
    shared_args = dict(
        hidden_channels=config["hidden_channels"],
        num_layers=config["num_interactions"],
        num_rbf=32,
        rbf_type="expnorm",
        trainable_rbf=False,
        activation="silu",
        neighbor_embedding=True,
        cutoff_lower=0,
        cutoff_upper=config["otfcutoff"], # default:5
        max_z=100,
        max_num_neighbors=32,
        otf_graph = config['otf_graph']
    )
    if config["model"] == "TorchMD_NeurIPS_Max":
        from .torchmd_maxneigh import TorchMD_NeurIPS_Max
        representation_model = TorchMD_NeurIPS_Max(
                            attn_activation="silu",
                            num_heads=8,
                            distance_influence="both",
                            **shared_args,
                        )
    elif config["model"] == "TorchMD_Norm":
        from .torchmd_norm import TorchMD_Norm
        representation_model = TorchMD_Norm(
            attn_activation="silu",
            num_heads=8,
            distance_influence="both",
            **shared_args,
        )
    elif config["model"] == "TorchMD_NeurIPS":
        from .torchmd_nips import TorchMD_NeurIPS
        representation_model = TorchMD_NeurIPS(
            attn_activation="silu",
            num_heads=8,
            distance_influence="both",
            **shared_args,
        )
    elif config["model"] == "TorchMD_Norm_VN":
        from .torchmd_norm_revised import TorchMD_Norm_VN
        representation_model = TorchMD_Norm_VN(
            attn_activation="silu",
            num_heads=8,
            distance_influence="both",
            **shared_args,
        )
    elif config["model"] == "ViSNetBlock":
        from .torchmd_sphere import ViSNetBlock
        representation_model = ViSNetBlock(
            attn_activation="silu",
            num_heads=8,
            distance_influence="both",
            **shared_args,
        )
    elif config["model"] == 'TorchMD_ET':
        from .torchmd_et import TorchMD_ET 
        representation_model = TorchMD_ET(
        attn_activation="silu",
        num_heads=8,
        distance_influence="both",
        **shared_args,
    )
    elif config['model'] == 'PaiNN':
        from .painn import PaiNN 
        representation_model = PaiNN(
        attn_activation="silu",
        num_heads=8,
        distance_influence="both",
        **shared_args,
    )
        
    ## EquivariantScalar and Scalar has the similar result.    
    # output_model = output_modules.Scalar(config["hidden_channels"], "silu")
    output_model = output_modules.EquivariantScalar(config["hidden_channels"], "silu")

    # combine representation and output network
    model = TorchMD_Net(
        representation_model,
        output_model,
        prior_model=prior_model,
        reduce_op="add",
        meanstd_level = config["meanstd_level"],
        mean=mean,
        std=std,
        atomref =atomref,
        derivative=config["regress_forces"],
    )
    
    return model


def load_model(filepath, args=None, device="cpu", **kwargs):
    ckpt = torch.load(filepath, map_location="cpu")
    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if not key in args:
            warnings.warn(f"Unknown hyperparameter: {key}={value}")
        args[key] = value

    model = create_model(args)

    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    return model.to(device)


class TorchMD_Net(nn.Module):
    def __init__(
        self,
        representation_model,
        output_model,
        prior_model=None,
        reduce_op="add",
        meanstd_level = "atom_level",
        mean=None,
        std=None,
        atomref = None,
        derivative=False,
    ):
        super(TorchMD_Net, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model
        self.meanstd_level = meanstd_level
        
        self.prior_model = prior_model
        if not output_model.allow_prior_model and prior_model is not None:
            self.prior_model = None
            rank_zero_warn(
                (
                    "Prior model was given but the output model does "
                    "not allow prior models. Dropping the prior model."
                )
            )

        self.reduce_op = reduce_op
        self.derivative = derivative
        
        self.atomref = None
        if atomref is not None:
            self.atomref = nn.Embedding(atomref.shape[0], 1)
            self.atomref.weight.data.copy_(atomref.reshape(-1,1))
        else:
            self.atomref = None
                    
        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(self,
                data,
                **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        z = data.atomic_numbers.long()
        if z.dim() == 2: # if z of shape num_atoms x 1
            z = z.squeeze() # squeeze to num_atoms
        pos = data.pos
        batch = data.batch
        if self.derivative:
            pos.requires_grad_(True)
        # run the potentially wrapped representation model
        x, v, z, pos, batch = self.representation_model(data)

        # apply the output network
        x = self.output_model.pre_reduce(x, v, z, pos, batch)

        if self.meanstd_level == "atom_level":
            # scale by data standard deviation
            if self.std is not None:
                x = x * self.std
            # shift by data mean
            if self.mean is not None:
                x = x + self.mean
            

        # apply prior model
        if self.prior_model is not None:
            x = self.prior_model(x, z, pos, batch)

        # aggregate atoms
        out = scatter(x, batch, dim=0, reduce=self.reduce_op)
        
        if self.meanstd_level == "molecule_level":
            # scale by data standard deviation
            if self.std is not None:
                out = out * self.std
            # shift by data mean
            if self.mean is not None:
                out = out + self.mean
        # apply output model after reduction
        out = self.output_model.post_reduce(out)
        if self.atomref is not None:
            out = out + scatter(self.atomref(z),batch, dim=0, reduce=self.reduce_op)
        
        # compute gradients with respect to coordinates
        if self.derivative:
            grad_outputs: List[Optional[torch.Tensor]] = torch.ones_like(out)
            dy = -grad(
                out,
                pos,
                grad_outputs=grad_outputs,
                create_graph=True if self.training else False,
                retain_graph=True if self.training else False
            )[0]
            if dy is None:
                raise RuntimeError("Autograd returned None for the force prediction.")
            return {"energy": out, "forces": dy}
        else:
            return {'energy': out}

