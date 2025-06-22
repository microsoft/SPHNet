# This file is adapted from QHNet_backbone.py in the QHNet repository.
# Also contains code from equiformer_v2.py in the equiformer_v2 repository.
# The original code is licensed under the MIT License.
from ..utils import construct_o3irrps_base,construct_o3irrps
import time
import torch.nn.functional as F

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius_graph
from e3nn import o3
from torch_scatter import scatter
from ..equiformer_v2.equiformer_v2_oc20 import  EquiformerV2_OC20_2output_Backbone
import numpy as np
import warnings

from ..utils import construct_o3irrps_base,construct_o3irrps

def prod(x):
    """Compute the product of a sequence."""
    out = 1
    for a in x:
        out *= a
    return out


def ShiftedSoftPlus(x):
    return torch.nn.functional.softplus(x) - math.log(2.0)


def softplus_inverse(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x + torch.log(-torch.expm1(-x))


def get_nonlinear(nonlinear: str):
    if nonlinear.lower() == 'ssp':
        return ShiftedSoftPlus
    elif nonlinear.lower() == 'silu':
        return F.silu
    elif nonlinear.lower() == 'tanh':
        return F.tanh
    elif nonlinear.lower() == 'abs':
        return torch.abs
    else:
        raise NotImplementedError


def get_feasible_irrep(irrep_in1, irrep_in2, cutoff_irrep_out, tp_mode="uvu"):
    """
    Get the feasible irreps based on the input irreps and cutoff irreps.

    Args:
        irrep_in1 (list): List of tuples representing the input irreps for the first input.
        irrep_in2 (list): List of tuples representing the input irreps for the second input.
        cutoff_irrep_out (list): List of irreps to be considered as cutoff irreps.
        tp_mode (str, optional): Tensor product mode. Defaults to "uvu".

    Returns:
        tuple: A tuple containing the feasible irreps and the corresponding instructions.
    """

    irrep_mid = []
    instructions = []

    for i, (_, ir_in) in enumerate(irrep_in1):
        for j, (_, ir_edge) in enumerate(irrep_in2):
            for ir_out in ir_in * ir_edge:
                if ir_out in cutoff_irrep_out:
                    if (cutoff_irrep_out.count(ir_out), ir_out) not in irrep_mid:
                        k = len(irrep_mid)
                        irrep_mid.append((cutoff_irrep_out.count(ir_out), ir_out))
                    else:
                        k = irrep_mid.index((cutoff_irrep_out.count(ir_out), ir_out))
                    instructions.append((i, j, k, tp_mode, True))

    irrep_mid = o3.Irreps(irrep_mid)
    normalization_coefficients = []
    for ins in instructions:
        ins_dict = {
            'uvw': (irrep_in1[ins[0]].mul * irrep_in2[ins[1]].mul),
            'uvu': irrep_in2[ins[1]].mul,
            'uvv': irrep_in1[ins[0]].mul,
            'uuw': irrep_in1[ins[0]].mul,
            'uuu': 1,
            'uvuv': 1,
            'uvu<v': 1,
            'u<vw': irrep_in1[ins[0]].mul * (irrep_in2[ins[1]].mul - 1) // 2,
        }
        alpha = irrep_mid[ins[2]].ir.dim
        x = sum([ins_dict[ins[3]] for ins in instructions])
        if x > 0.0:
            alpha /= x
        normalization_coefficients += [math.sqrt(alpha)]

    irrep_mid, p, _ = irrep_mid.sort()
    instructions = [
        (i_in1, i_in2, p[i_out], mode, train, alpha)
        for (i_in1, i_in2, i_out, mode, train), alpha
        in zip(instructions, normalization_coefficients)
    ]
    return irrep_mid, instructions


def cutoff_function(x, cutoff):
    zeros = torch.zeros_like(x)
    x_ = torch.where(x < cutoff, x, zeros)
    return torch.where(x < cutoff, torch.exp(-x_**2/((cutoff-x_)*(cutoff+x_))), zeros)

class ExponentialBernsteinRadialBasisFunctions(nn.Module):
    def __init__(self, num_basis_functions, cutoff, ini_alpha=0.5):
        super(ExponentialBernsteinRadialBasisFunctions, self).__init__()
        self.num_basis_functions = num_basis_functions
        self.ini_alpha = ini_alpha
        # compute values to initialize buffers
        logfactorial = np.zeros((num_basis_functions))
        for i in range(2,num_basis_functions):
            logfactorial[i] = logfactorial[i-1] + np.log(i)
        v = np.arange(0,num_basis_functions)
        n = (num_basis_functions-1)-v
        logbinomial = logfactorial[-1]-logfactorial[v]-logfactorial[n]
        #register buffers and parameters
        self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float32))
        self.register_buffer('logc', torch.tensor(logbinomial, dtype=torch.float32))
        self.register_buffer('n', torch.tensor(n, dtype=torch.float32))
        self.register_buffer('v', torch.tensor(v, dtype=torch.float32))
        self.register_parameter('_alpha', nn.Parameter(torch.tensor(1.0, dtype=torch.float32)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self._alpha,  softplus_inverse(self.ini_alpha))

    def forward(self, r):
        alpha = F.softplus(self._alpha)
        x = - alpha * r
        x = self.logc + self.n * x + self.v * torch.log(- torch.expm1(x) )
        rbf = cutoff_function(r, self.cutoff) * torch.exp(x)
        return rbf





    


class WANet_backbone(nn.Module):
    def __init__(self,
                 order = 4,
                 embedding_dimension=128,
                 bottle_hidden_size=32,
                 num_gnn_layers=5,
                 max_radius=15,
                 num_nodes=20,
                 radius_embed_dim=32,
                 use_equi_norm=False,
                 start_layer = 3,
                 load_pretrain = '',
                 **kwargs):  # maximum nuclear charge (+1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        """
        Initialize the QHNet_backbone model.

        Args:
            order (int): The order of the spherical harmonics.
            embedding_dimension (int): The size of the hidden layer.
            bottle_hidden_size (int): The size of the bottleneck hidden layer.
            num_gnn_layers (int): The number of GNN layers.
            max_radius (int): The maximum radius cutoff.
            num_nodes (int): The number of nodes.
            radius_embed_dim (int): The dimension of the radius embedding.
        """
        
        super().__init__()
        self.order = order
        self.sh_irrep = o3.Irreps.spherical_harmonics(lmax=self.order)
        self.hs = embedding_dimension
        self.hbs = bottle_hidden_size
        self.radius_embed_dim = radius_embed_dim
        self.max_radius = max_radius
        self.num_gnn_layers = num_gnn_layers
        self.node_embedding = nn.Embedding(num_nodes, self.hs)
        
        self.init_sph_irrep = o3.Irreps(construct_o3irrps(1, order=order))
        
        self.irreps_node_embedding = construct_o3irrps_base(self.hs, order=order)
        self.hidden_irrep = o3.Irreps(construct_o3irrps(self.hs, order=order))
        self.hidden_irrep_base = o3.Irreps(self.irreps_node_embedding)
        self.hidden_bottle_irrep = o3.Irreps(construct_o3irrps(self.hbs, order=order))
        self.hidden_bottle_irrep_base = o3.Irreps(construct_o3irrps_base(self.hbs, order=order))
        
        self.input_irrep = o3.Irreps(f'{self.hs}x0e')
        self.radial_basis_functions = ExponentialBernsteinRadialBasisFunctions(self.radius_embed_dim, self.max_radius)
        self.nonlinear_scalars = {1: "ssp", -1: "tanh"}
        self.nonlinear_gates = {1: "ssp", -1: "abs"}
        self.num_fc_layer = 1

        # prevent double kwargs
        [kwargs.pop(x, None) for x in ["use_pbc", "regress_forces", "max_raius", "otf_graph", "num_layers", "sphere_channels", "lmax_list"]]
        self.node_attr_encoder = EquiformerV2_OC20_2output_Backbone(None, None, None, max_radius = max_radius, lmax_list=[order], 
                                                           sphere_channels=embedding_dimension, 
                                                           num_layers = num_gnn_layers, use_pbc=False, 
                                                           regress_forces=False, otf_graph=False, **kwargs)
        if load_pretrain != '':
            loaded_state_dict = torch.load(load_pretrain)['state_dict']
            state_dict = {k.replace('module.module.', ''): v for k, v in loaded_state_dict.items()}
            self.node_attr_encoder.load_state_dict(state_dict, strict=False)

    def reset_parameters(self):
        warnings.warn("reset parameter is not init in wanet backbone model")
    
    
    def forward(self, batch_data):
        batch_data['ptr'] = torch.cat([torch.Tensor([0]).to(batch_data["molecule_size"].device).int(),
                              torch.cumsum(batch_data["molecule_size"],dim = 0)],dim = 0)
        
        batch_data['natoms'] = scatter(torch.ones_like(batch_data.batch), batch_data.batch, dim=0, reduce='sum')
        batch_data.atomic_numbers = batch_data.atomic_numbers.squeeze()
        batch_data['node_attr'] = self.node_embedding(batch_data.atomic_numbers)
        edge_index = radius_graph(batch_data.pos, self.max_radius, batch_data.batch)
        batch_data.edge_index = edge_index

        node_vec = self.node_attr_encoder(batch_data)
        batch_data["node_embedding"] = batch_data.node_attr
        batch_data["node_vec"] = node_vec  

        return batch_data


