import torch.nn.functional as F
import os

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius_graph
from e3nn import o3
from torch_scatter import scatter
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Linear, TensorProduct
from .lsrm.normalize import EquivariantLayerNormArraySphericalHarmonics
import numpy as np
import warnings
from .sparse_tp.expansion_fx import ExpansionFX as Expansion
from .sparse_tp.sparse_tp import Sparse_TensorProduct

from .utils import construct_o3irrps_base, construct_o3irrps, get_full_graph, get_conv_variable, block2matrix, get_transpose_index

from .lsrm.lsrm_modules import Visnorm_shared_LSRMNorm2_2branchSerial

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


class NormGate(torch.nn.Module):
    def __init__(self, irrep):
        super(NormGate, self).__init__()
        self.irrep = irrep
        self.norm = o3.Norm(self.irrep)
    
        num_mul, num_mul_wo_0 = 0, 0
        for mul, ir in self.irrep:
            num_mul += mul
            if ir.l != 0:
                num_mul_wo_0 += mul

        self.mul = o3.ElementwiseTensorProduct(
            self.irrep[1:], o3.Irreps(f"{num_mul_wo_0}x0e"))
        self.fc = nn.Sequential(
            nn.Linear(num_mul, num_mul),
            nn.SiLU(),
            nn.Linear(num_mul, num_mul))

        self.num_mul = num_mul
        self.num_mul_wo_0 = num_mul_wo_0

    def forward(self, x):
        norm_x = self.norm(x)[:, self.irrep.slices()[0].stop:]
        f0 = torch.cat([x[:, self.irrep.slices()[0]], norm_x], dim=-1)
        gates = self.fc(f0)
        gated = self.mul(x[:, self.irrep.slices()[0].stop:], gates[:, self.irrep.slices()[0].stop:])
        x = torch.cat([gates[:, self.irrep.slices()[0]], gated], dim=-1)
        return x

class NonDiagLayer(torch.nn.Module):
    def __init__(self,
                 irrep_in_node,
                 irrep_bottle_hidden,
                 irrep_out,
                 sh_irrep,
                 edge_attr_dim,
                 node_attr_dim,
                 resnet = True,
                 invariant_layers=1,
                 invariant_neurons=8,
                 tp_mode = "uuu",
                 nonlinear='ssp',
                 use_sparse_tp=False,
                 use_pair_sparse=False,
                 sparsity = 0.7,
                 id=1,
                 ckpt_path=None,
                 steps_1epoch=1700):
        super().__init__()
        self.id = id
        self.sparse = use_sparse_tp
        self.pair_sparsity = sparsity # pair sparsity
        self.use_pair_sparse = use_pair_sparse
        self.nonlinear_scalars = {1: "ssp", -1: "tanh"}
        self.nonlinear_gates = {1: "ssp", -1: "abs"}
        self.invariant_layers = invariant_layers
        self.invariant_neurons = invariant_neurons
        self.irrep_in_node = irrep_in_node if isinstance(irrep_in_node, o3.Irreps) else o3.Irreps(irrep_in_node)
        self.irrep_bottle_hidden = irrep_bottle_hidden \
            if isinstance(irrep_bottle_hidden, o3.Irreps) else o3.Irreps(irrep_bottle_hidden)
        self.irrep_out = irrep_out if isinstance(irrep_out, o3.Irreps) else o3.Irreps(irrep_out)
        self.sh_irrep = sh_irrep if isinstance(sh_irrep, o3.Irreps) else o3.Irreps(sh_irrep)

        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.nonlinear_layer = get_nonlinear(nonlinear)

        self.irrep_tp_in_node, _ = get_feasible_irrep(self.irrep_in_node, o3.Irreps("0e"), self.irrep_bottle_hidden)

        self.norm_gate_pre = NormGate(self.irrep_in_node)
        self.linear_node_pair_input = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_tp_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )
        
        self.irrep_tp_out_node_pair, instruction_node_pair = get_feasible_irrep(
        self.irrep_tp_in_node, self.irrep_tp_in_node, self.irrep_bottle_hidden, tp_mode=tp_mode)


        self.linear_node_pair_inner = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )
        self.inner_product = InnerProduct(self.irrep_in_node)

        # tensor product for node pair : left and right


        self.tp_node_pair_sp = Sparse_TensorProduct(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_tp_out_node_pair,
            instruction_node_pair,
            shared_weights=False,
            internal_weights=False,
            sparsity=sparsity,
            steps_1epoch=steps_1epoch,
        )

        self.tp_node_pair_fix = TensorProduct(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_tp_out_node_pair,
            instruction_node_pair,
            shared_weights=False,
            internal_weights=False,
        )      

        self.fc_node_pair = FullyConnectedNet(
            [self.edge_attr_dim] + invariant_layers * [invariant_neurons] + [self.tp_node_pair_sp.weight_numel],
            self.nonlinear_layer
        )
        self.indice = list(range(self.tp_node_pair_sp.weight_numel))

        if self.irrep_in_node == self.irrep_out and resnet:
            self.resnet = True
        else:
            self.resnet = False

        self.node_residual = Linear(
            irreps_in=self.irrep_tp_out_node_pair,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )
        
        self.norm_gate = NormGate(self.irrep_tp_out_node_pair)
        num_mul = 0
        for mul, ir in self.irrep_in_node:
            num_mul = num_mul + mul

        self.fc = nn.Sequential(
            nn.Linear(num_mul, self.irrep_in_node[0][0]),
            nn.SiLU(),
            nn.Linear(self.irrep_in_node[0][0], self.tp_node_pair_sp.weight_numel))
        
        self.pair_select_layer = nn.Linear(self.irrep_in_node[0][0]*len(self.irrep_in_node),1)
        
        self.ckpt_path = ckpt_path
        if os.path.isfile(f'{ckpt_path}/nondiag_ins_{self.id}.pt') and os.path.isfile(f'{ckpt_path}/nondiag_indice_{self.id}.pt'):
            print(f'load existing sparse tp, non diag {self.id}')
            instruction_saved = torch.load(f'{ckpt_path}/nondiag_ins_{self.id}.pt')
            self.indice = torch.load(f'{ckpt_path}/nondiag_indice_{self.id}.pt')
            self.reset_tp(instruction_saved)
            self.use_fix_tp = True
        else:
            self.use_fix_tp = use_sparse_tp

    @property
    def device(self):
        return next(self.parameters()).device

    def reset_tp(self, instructions):
        instruction_nodes = []
        for ins in instructions:
            instruction_nodes.append((ins.i_in1, ins.i_in2, ins.i_out, ins.connection_mode, ins.has_weight, ins.path_weight**2))
        self.tp_node_pair_fix = TensorProduct(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_tp_out_node_pair,
            instruction_nodes,
            shared_weights=False,
            internal_weights=False,
            path_normalization="none",
            irrep_normalization="none",
        ).to(self.device)

    def forward(self, data, node_attr, node_pair_attr=None):
        dst, src = data['full_edge_index']
        node_attr_0 = self.linear_node_pair_inner(node_attr)
        s0 = self.inner_product(node_attr_0[dst], node_attr_0[src])[:, self.irrep_in_node.slices()[0].stop:]
        s0 = torch.cat([0.5*node_attr_0[dst][:, self.irrep_in_node.slices()[0]]+
                        0.5*node_attr_0[src][:, self.irrep_in_node.slices()[0]], s0], dim=-1)

        node_attr = self.norm_gate_pre(node_attr)
        node_attr = self.linear_node_pair_input(node_attr)

        # Sparse pair gate
        if self.use_pair_sparse:
            pair_weight = self.pair_select_layer(s0)
            quantile = torch.quantile(pair_weight, q=self.pair_sparsity)
            pair_weight = torch.sigmoid(pair_weight-quantile)
            pair_weight = pair_weight * (pair_weight >= 0.5)
            path_mask = torch.where(pair_weight!=0)[0]
            src = src[path_mask]
            dst = dst[path_mask]
            s0 = (s0*pair_weight)[path_mask]
            data['full_edge_attr'] = data['full_edge_attr'][path_mask]
            
        if self.use_fix_tp:
            node_pair = self.tp_node_pair_fix(node_attr[src], node_attr[dst],
                (self.fc_node_pair(data['full_edge_attr']) * self.fc(s0))[:,self.indice])
        else:
            node_pair = self.tp_node_pair_sp(node_attr[src], node_attr[dst],
                (self.fc_node_pair(data['full_edge_attr']) * self.fc(s0))[:,self.indice])

        if isinstance(node_pair, tuple):
            print('dynamic sparse tensor product finished, returning to fixed graph tensor product')
            self.reset_tp(node_pair[0])
            torch.save(node_pair[0],f'{self.ckpt_path}/nondiag_ins_{self.id}.pt')
            torch.save(node_pair[2],f'{self.ckpt_path}/nondiag_indice_{self.id}.pt')
            self.indice = node_pair[2]
            node_pair = node_pair[1]
            self.use_fix_tp = True

        node_pair = self.norm_gate(node_pair)
        node_pair = self.node_residual(node_pair)

        if self.resnet and node_pair_attr is not None:
            if not self.use_pair_sparse:
                node_pair_attr = node_pair + node_pair_attr  
            else: 
                node_pair_attr[path_mask] = node_pair + node_pair_attr[path_mask]
            node_pair = node_pair_attr
        return node_pair

class DiagLayer(torch.nn.Module):
    def __init__(self,
                 irrep_in_node,
                 irrep_bottle_hidden,
                 irrep_out,
                 sh_irrep,
                 edge_attr_dim,
                 node_attr_dim,
                 resnet = True,
                 tp_mode = "uuu",
                 nonlinear='ssp',
                 use_sparse_tp=False,
                 sparsity=0.7,
                 id=1,
                 ckpt_path=None,
                 steps_1epoch=1700):
        super(DiagLayer, self).__init__()
        self.id = str(id)
        self.nonlinear_scalars = {1: "ssp", -1: "tanh"}
        self.nonlinear_gates = {1: "ssp", -1: "abs"}
        self.sh_irrep = sh_irrep
        self.irrep_in_node = irrep_in_node if isinstance(irrep_in_node, o3.Irreps) else o3.Irreps(irrep_in_node)
        self.irrep_bottle_hidden = irrep_bottle_hidden \
            if isinstance(irrep_bottle_hidden, o3.Irreps) else o3.Irreps(irrep_bottle_hidden)
        self.irrep_out = irrep_out if isinstance(irrep_out, o3.Irreps) else o3.Irreps(irrep_out)

        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.resnet = resnet
        self.nonlinear_layer = get_nonlinear(nonlinear)

        self.irrep_tp_in_node, _ = get_feasible_irrep(self.irrep_in_node, o3.Irreps("0e"), self.irrep_bottle_hidden)
        self.irrep_tp_out_node, instruction_node = get_feasible_irrep(
            self.irrep_tp_in_node, self.irrep_tp_in_node, self.irrep_bottle_hidden, tp_mode=tp_mode)

        # - Build modules -
        self.linear_node_1 = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_tp_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        self.linear_node_2 = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_tp_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        self.tp_sp = Sparse_TensorProduct(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_tp_out_node,
            instruction_node,
            shared_weights=True,
            internal_weights=True,
            sparsity=sparsity,
            steps_1epoch=steps_1epoch,
        )
        self.tp_fix = TensorProduct(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_tp_out_node,
            instruction_node,
            shared_weights=True,
            internal_weights=True
        )

        self.norm_gate = NormGate(self.irrep_out)
        self.norm_gate_1 = NormGate(self.irrep_in_node)
        self.norm_gate_2 = NormGate(self.irrep_in_node)
        self.linear_node_3 = Linear(
            irreps_in=self.irrep_tp_out_node,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        self.ckpt_path = ckpt_path
        if os.path.isfile(f'{ckpt_path}/diag_ins_{self.id}.pt') and os.path.isfile(f'{ckpt_path}/diag_weight_{self.id}.pt'):
            print(f'load existing sparse tp, diag {self.id}')
            instruction_saved = torch.load(f'{ckpt_path}/diag_ins_{self.id}.pt')
            instruction_weight = torch.load(f'{ckpt_path}/diag_weight_{self.id}.pt')
            self.reset_tp(instruction_saved,instruction_weight)
            self.use_fix_tp = True
        else:
            self.use_fix_tp = use_sparse_tp
    
    def reset_tp(self, instructions, weight):
        instruction_nodes = []
        for ins in instructions:
            instruction_nodes.append((ins.i_in1, ins.i_in2, ins.i_out, ins.connection_mode, ins.has_weight, ins.path_weight**2))
        self.tp_fix = TensorProduct(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_tp_out_node,
            instruction_nodes,
            shared_weights=True,
            internal_weights=True,
            path_normalization="none",
            irrep_normalization="none",
        ).to(self.device)
        self.tp_fix.weight.data = weight

    def forward(self, data, x, old_fii):
        old_x = x
        xl = self.norm_gate_1(x)
        xl = self.linear_node_1(xl)
        xr = self.norm_gate_2(x)
        xr = self.linear_node_2(xr)

        if self.use_fix_tp:
            x = self.tp_fix(xl, xr)
        else:
            x = self.tp_sp(xl, xr)

        if isinstance(x, tuple):
            print('dynamic sparse tensor product finished, returning to fixed graph tensor product')
            self.reset_tp(x[0],x[3])
            torch.save(x[0],f'{self.ckpt_path}/diag_ins_{self.id}.pt')
            torch.save(x[3],f'{self.ckpt_path}/diag_weight_{self.id}.pt')
            x = x[1]
            self.use_fix_tp = True

        if self.resnet and x.shape[-1] == old_x.shape[-1]:   # can't add if raise order
            x = x + old_x
        x = self.norm_gate(x)
        x = self.linear_node_3(x)
        if self.resnet and old_fii is not None:
            x = old_fii + x
        return x

    @property
    def device(self):
        return next(self.parameters()).device


class ConvLayer(torch.nn.Module):
    def __init__(
            self,
            irrep_in_node,
            irrep_hidden,
            irrep_out,
            sh_irrep,
            edge_attr_dim,
            node_attr_dim,
            invariant_layers=1,
            invariant_neurons=32,
            avg_num_neighbors=None,
            nonlinear='ssp',
            use_norm_gate=True,
            edge_wise=False,
            use_equi_norm=False,
            use_sparse_tp=False,
            use_pair_sparse=False,
            sparsity=0.7,
    ):
        """
        Initialize the ConvLayer.

        Args:
            irrep_in_node (o3.Irreps or str): The irreps of the input nodes.
            irrep_hidden (o3.Irreps or str): The irreps of the hidden layers.
            irrep_out (o3.Irreps or str): The irreps of the output nodes.
            sh_irrep (o3.Irreps or str): The irreps of the spherical harmonics.
            edge_attr_dim (int): The dimension of the edge attributes.
            node_attr_dim (int): The dimension of the node attributes.
            invariant_layers (int, optional): The number of invariant layers. Defaults to 1.
            invariant_neurons (int, optional): The number of neurons in each invariant layer. Defaults to 32.
            avg_num_neighbors (int, optional): The average number of neighbors. Defaults to None.
            nonlinear (str, optional): The type of nonlinearity. Defaults to 'ssp'.
            use_norm_gate (bool, optional): Whether to use the normalization gate. Defaults to True.
            edge_wise (bool, optional): Whether to use edge-wise operations. Defaults to False.
        """    
        super(ConvLayer, self).__init__()
        self.use_pair_sparse = use_pair_sparse
        self.pair_sparsity = sparsity
        self.avg_num_neighbors = avg_num_neighbors
        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.edge_wise = edge_wise

        self.irrep_in_node = irrep_in_node if isinstance(irrep_in_node, o3.Irreps) else o3.Irreps(irrep_in_node)
        self.irrep_hidden = irrep_hidden \
            if isinstance(irrep_hidden, o3.Irreps) else o3.Irreps(irrep_hidden)
        self.irrep_out = irrep_out if isinstance(irrep_out, o3.Irreps) else o3.Irreps(irrep_out)
        self.sh_irrep = sh_irrep if isinstance(sh_irrep, o3.Irreps) else o3.Irreps(sh_irrep)
        self.nonlinear_layer = get_nonlinear(nonlinear)

        self.irrep_tp_out_node, instruction_node = get_feasible_irrep(
            self.irrep_in_node, self.sh_irrep, self.irrep_hidden, tp_mode='uvu')
        

        if use_sparse_tp:
            self.tp_node = Sparse_TensorProduct(
                self.irrep_in_node,
                self.sh_irrep,
                self.irrep_tp_out_node,
                instruction_node,
                shared_weights=False,
                internal_weights=False,
                sparsity=sparsity,
            )
        else:
            self.tp_node = TensorProduct(
                self.irrep_in_node,
                self.sh_irrep,
                self.irrep_tp_out_node,
                instruction_node,
                shared_weights=False,
                internal_weights=False,
            )

        self.fc_node = FullyConnectedNet(
            [self.edge_attr_dim] + invariant_layers * [invariant_neurons] + [self.tp_node.weight_numel],
            self.nonlinear_layer
        )
        self.indice = list(range(self.tp_node.weight_numel))

        num_mul = 0
        for mul, ir in self.irrep_in_node:
            num_mul = num_mul + mul

        self.layer_l0 = FullyConnectedNet(
            [num_mul + self.irrep_in_node[0][0]] + invariant_layers * [invariant_neurons] + [self.tp_node.weight_numel],
            self.nonlinear_layer
        )

        self.linear_out = Linear(
            irreps_in=self.irrep_tp_out_node,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        self.use_norm_gate = use_norm_gate
        self.norm_gate = NormGate(self.irrep_in_node)
        self.irrep_linear_out, instruction_node = get_feasible_irrep(
            self.irrep_in_node, o3.Irreps("0e"), self.irrep_in_node)
        self.linear_node = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_linear_out,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )
        self.linear_node_pre = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_linear_out,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )
        self.inner_product = InnerProduct(self.irrep_in_node)
        if use_pair_sparse:
            self.pair_select_layer = nn.Linear(self.irrep_in_node[0][0]*(len(self.irrep_in_node)+1),1)

        self.use_equi_norm = use_equi_norm
        if self.use_equi_norm:
            self.lmax = len(self.irrep_tp_out_node)-1
            self.norm = EquivariantLayerNormArraySphericalHarmonics(self.lmax,self.irrep_tp_out_node[0][0])
            
    @property
    def device(self):
        return next(self.parameters()).device
    
    def reset_tp(self, instructions):
        instruction_nodes = []
        for ins in instructions:
            instruction_nodes.append((ins.i_in1, ins.i_in2, ins.i_out, ins.connection_mode, ins.has_weight, ins.path_weight))
        self.tp_node = TensorProduct(
            self.irrep_in_node,
            self.sh_irrep,
            self.irrep_tp_out_node,
            instruction_nodes,
            shared_weights=False,
            internal_weights=False,
        ).to(self.device)

    def forward(self, data, x):
        edge_dst, edge_src = data.edge_index[0], data.edge_index[1]

        if self.use_norm_gate:
            pre_x = self.linear_node_pre(x)
            s0 = self.inner_product(pre_x[edge_dst], pre_x[edge_src])[:, self.irrep_in_node.slices()[0].stop:]
            s0 = torch.cat([pre_x[edge_dst][:, self.irrep_in_node.slices()[0]],
                            pre_x[edge_src][:, self.irrep_in_node.slices()[0]], s0], dim=-1)
            x = self.norm_gate(x)
            x = self.linear_node(x)
        else:
            s0 = self.inner_product(x[edge_dst], x[edge_src])[:, self.irrep_in_node.slices()[0].stop:]
            s0 = torch.cat([x[edge_dst][:, self.irrep_in_node.slices()[0]],
                            x[edge_src][:, self.irrep_in_node.slices()[0]], s0], dim=-1)

        self_x = x

        # Sparse pair gate
        if self.use_pair_sparse:
            pair_weight = self.pair_select_layer(s0)
            quantile = torch.quantile(pair_weight, q=self.pair_sparsity)
            pair_weight = torch.sigmoid(pair_weight-quantile)
            pair_weight = torch.where(pair_weight < 0.5, torch.zeros_like(pair_weight), pair_weight)
            path_mask = torch.where(pair_weight!=0)[0]
            edge_src = edge_src[path_mask]
            edge_dst = edge_dst[path_mask]
            s0 = (s0*pair_weight)[path_mask]
            data.edge_sh = data.edge_sh[path_mask]
            data.edge_attr = data.edge_attr[path_mask]

        edge_features = self.tp_node(
            x[edge_src], data.edge_sh, (self.fc_node(data.edge_attr) * self.layer_l0(s0))[:,self.indice])#, torch.randint(0,1,(len(self.tp_node.instructions),)))

        if isinstance(edge_features, tuple):
            print('dynamic sparse tensor product finished, returning to fixed graph tensor product')
            self.reset_tp(edge_features[0])
            self.indice = edge_features[2]
            edge_features = edge_features[1]

        if self.edge_wise:
            out = edge_features
        else:
            out = scatter(edge_features, edge_dst, dim=0, dim_size=len(x))

        if self.use_equi_norm: out = self.norm(out.view(out.shape[0],(self.lmax+1)**2, -1)).view(out.shape[0],-1) 
        
        if self.irrep_in_node == self.irrep_out:
            out = out + self_x

        out = self.linear_out(out)
        return out


class InnerProduct(torch.nn.Module):
    def __init__(self, irrep_in):
        super(InnerProduct, self).__init__()
        self.irrep_in = o3.Irreps(irrep_in).simplify()
        irrep_out = o3.Irreps([(mul, "0e") for mul, _ in self.irrep_in])
        instr = [(i, i, i, "uuu", False, 1/ir.dim) for i, (mul, ir) in enumerate(self.irrep_in)]
        
        self.tp = o3.TensorProduct(self.irrep_in, self.irrep_in, irrep_out, instr, irrep_normalization="component")
        self.irrep_out = irrep_out.simplify()

    def forward(self, features_1, features_2):
        out = self.tp(features_1, features_2)
        return out


class ConvNetLayer(torch.nn.Module):
    def __init__(
            self,
            irrep_in_node,
            irrep_hidden,
            irrep_out,
            sh_irrep,
            edge_attr_dim,
            node_attr_dim,
            resnet = True,
            use_norm_gate=True,
            edge_wise=False,
            use_equi_norm=False,
            use_sparse_tp=False,
            use_pair_sparse=False,
            sparsity=0.7,
    ):
        """
        Initializes the tensor product ConvNetLayer.

        Args:
            irrep_in_node (o3.Irreps or str): The input irreps for each node.
            irrep_hidden (o3.Irreps or str): The irreps for the hidden layers.
            irrep_out (o3.Irreps or str): The output irreps.
            sh_irrep (o3.Irreps or str): The irreps for the spherical harmonics.
            edge_attr_dim (int): The dimension of the edge attributes.
            node_attr_dim (int): The dimension of the node attributes.
            resnet (bool, optional): Whether to use residual connections. Defaults to True.
            use_norm_gate (bool, optional): Whether to use normalization gates. Defaults to True.
            edge_wise (bool, optional): Whether to process edges independently. Defaults to False.
        """
        super(ConvNetLayer, self).__init__()
        self.nonlinear_scalars = {1: "ssp", -1: "tanh"}
        self.nonlinear_gates = {1: "ssp", -1: "abs"}

        self.irrep_in_node = irrep_in_node if isinstance(irrep_in_node, o3.Irreps) else o3.Irreps(irrep_in_node)
        self.irrep_hidden = irrep_hidden if isinstance(irrep_hidden, o3.Irreps) \
            else o3.Irreps(irrep_hidden)
        self.irrep_out = irrep_out if isinstance(irrep_out, o3.Irreps) else o3.Irreps(irrep_out)
        self.sh_irrep = sh_irrep if isinstance(sh_irrep, o3.Irreps) else o3.Irreps(sh_irrep)

        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.resnet = resnet and self.irrep_in_node == self.irrep_out

        self.conv = ConvLayer(
            irrep_in_node=self.irrep_in_node,
            irrep_hidden=self.irrep_hidden,
            sh_irrep=self.sh_irrep,
            irrep_out=self.irrep_out,
            edge_attr_dim=self.edge_attr_dim,
            node_attr_dim=self.node_attr_dim,
            invariant_layers=1,
            invariant_neurons=32,
            avg_num_neighbors=None,
            nonlinear='ssp',
            use_norm_gate=use_norm_gate,
            edge_wise=edge_wise,
            use_equi_norm=use_equi_norm,
            use_sparse_tp=use_sparse_tp,
            use_pair_sparse=use_pair_sparse,
            sparsity=sparsity,
        )

    def forward(self, data, x):
        old_x = x
        x = self.conv(data, x)
        if self.resnet:
            x = old_x + x
        return x

class Pair_construction_layer(nn.Module):
    def __init__(self,
                 irrep_in_node =  "128x0e + 128x1e + 128x2e + 128x3e + 128x4e",
                 irreps_edge_embedding = "32x0e + 32x1e + 32x2e + 32x3e + 32x4e",
                 order = 4,
                 pyscf_basis_name = "def2-svp",
                 radius_embed_dim=64,
                 max_radius_cutoff=15,
                 bottle_hidden_size=64,
                 num_layer=2,
                 use_sparse_tp=False,
                 sparsity=0.7,
                 ckpt_path=None,
                 steps_1epoch=1700,
                 **kwargs):
        """
        This is the implement of Pair Construction Blocks.
        irrep_in_node: node feature shape.
        irreps_edge_embedding: pair feature shape.
        order: maximum order of irreps.
        pyscf_basis_name: basis.
        radius_embed_dim: feature dimension of the output of radius basis function.
        max_radius_cutoff: max radius cutoff of RBF.
        bottle_hidden_size: hidden dimension of pair features.
        num_layer: number of pair construction blocks.
        use_sparse_tp: wether use sparse gate.
        sparsity: sparse rate of the sparse gates (both pair and tp gates).    
        ckpt_path: path to save model.
        steps_1epoch: the steps to finish 1 epoch.
        """
        super().__init__()

        self.hs = o3.Irreps(irrep_in_node)[0][0]
        self.use_sparse_tp = use_sparse_tp
        self.pyscf_basis_name = pyscf_basis_name
        if pyscf_basis_name == 'def2-svp':
            exp_irrp = o3.Irreps("3x0e + 2x1e + 1x2e")
        elif pyscf_basis_name == 'def2-tzvp':
            exp_irrp = o3.Irreps("5x0e + 5x1e + 2x2e + 1x3e")
        else:
            raise ValueError('invalid base')
        
        self.conv,_,self.mask_lin,_ = get_conv_variable(pyscf_basis_name)
        self.order = order
        self.radial_basis_functions = None
        self.sh_irrep = o3.Irreps.spherical_harmonics(lmax=self.order)
        self.expand_ii, self.expand_ij, self.fc_ii, self.fc_ij, self.fc_ii_bias, self.fc_ij_bias = \
            nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
        
        for name in {"hamiltonian"}:
            self.expand_ii[name] = Expansion(
                o3.Irreps(irreps_edge_embedding),
                exp_irrp,
                exp_irrp
            )
            self.fc_ii[name] = torch.nn.Sequential(
                nn.Linear(self.hs, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ii[name].num_path_weight)
            )
            self.fc_ii_bias[name] = torch.nn.Sequential(
                nn.Linear(self.hs, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ii[name].num_bias)
            )

            self.expand_ij[name] = Expansion(
                o3.Irreps(irreps_edge_embedding),
                exp_irrp,
                exp_irrp
            )

            self.fc_ij[name] = torch.nn.Sequential(
                nn.Linear(self.hs , self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_path_weight)
            )

            self.fc_ij_bias[name] = torch.nn.Sequential(
                nn.Linear(self.hs , self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_bias)
            )

        self.num_layer = num_layer
        self.sh_irrep = o3.Irreps.spherical_harmonics(lmax=self.order)
        self.radius_embed_dim = radius_embed_dim
        self.rbf = ExponentialBernsteinRadialBasisFunctions(self.radius_embed_dim, max_radius_cutoff)

        self.hbs = bottle_hidden_size
        self.hidden_irrep = o3.Irreps(construct_o3irrps(self.hs, order=self.order))
        self.hidden_irrep_base = o3.Irreps(construct_o3irrps_base(self.hs, order=self.order))
        self.hidden_bottle_irrep = o3.Irreps(construct_o3irrps(self.hbs, order=self.order))

        self.e3_gnn_node_pair_layer = nn.ModuleList()
        self.e3_gnn_node_layer = nn.ModuleList()
        irrep_in_node = o3.Irreps(str(irrep_in_node).replace('o', 'e'))
        for l in range(self.num_layer):
            if l == 0:
                self.e3_gnn_node_layer.append(DiagLayer(
                    irrep_in_node=irrep_in_node, # o3.Irreps(str("128x0e + 128x1e + 128x2e")),
                    irrep_bottle_hidden=self.hidden_irrep_base,
                    irrep_out=self.hidden_irrep_base,
                    sh_irrep=self.sh_irrep,
                    edge_attr_dim=self.radius_embed_dim,
                    node_attr_dim=self.hs,
                    use_sparse_tp=use_sparse_tp,
                    sparsity = sparsity,
                    resnet=False,
                    id=l+1,
                    ckpt_path=ckpt_path,
                    steps_1epoch=steps_1epoch,
                ))
                self.e3_gnn_node_pair_layer.append(NonDiagLayer(
                    irrep_in_node=irrep_in_node, # o3.Irreps(str("128x0e + 128x1e + 128x2e")),
                    irrep_bottle_hidden=self.hidden_irrep_base,
                    irrep_out=self.hidden_irrep_base,
                    sh_irrep=self.sh_irrep,
                    edge_attr_dim=self.radius_embed_dim,
                    node_attr_dim=self.hs,
                    invariant_layers=1,
                    invariant_neurons=self.hs,
                    use_sparse_tp=use_sparse_tp,
                    sparsity = sparsity,
                    resnet=False,
                    id=l+1,
                    ckpt_path=ckpt_path,
                    steps_1epoch=steps_1epoch,
                ))
            else:
                self.e3_gnn_node_layer.append(DiagLayer(
                    irrep_in_node=irrep_in_node,
                    irrep_bottle_hidden=self.hidden_irrep_base,
                    irrep_out=self.hidden_irrep_base,
                    sh_irrep=self.sh_irrep,
                    edge_attr_dim=self.radius_embed_dim,
                    node_attr_dim=self.hs,
                    use_sparse_tp=use_sparse_tp,
                    sparsity = sparsity,
                    resnet=True,
                    id=l+1,
                    ckpt_path=ckpt_path,
                    steps_1epoch=steps_1epoch,
                ))
                self.e3_gnn_node_pair_layer.append(NonDiagLayer(
                    irrep_in_node=irrep_in_node,
                    irrep_bottle_hidden=self.hidden_irrep_base,
                    irrep_out=self.hidden_irrep_base,
                    sh_irrep=self.sh_irrep,
                    edge_attr_dim=self.radius_embed_dim,
                    node_attr_dim=self.hs,
                    invariant_layers=1,
                    invariant_neurons=self.hs,
                    use_sparse_tp=use_sparse_tp,
                    use_pair_sparse=True,
                    sparsity = sparsity,
                    resnet=True, 
                    id=l+1,   
                    ckpt_path=ckpt_path,
                    steps_1epoch=steps_1epoch,
                ))            

        self.output_ii = o3.Linear(self.hidden_irrep, self.hidden_bottle_irrep)
        self.output_ij = o3.Linear(self.hidden_irrep, self.hidden_bottle_irrep)

    def reset_parameters(self):
        warnings.warn("reset parameter is not init in hami head model")

    # Reconstruct the full hamiltonian matrix with hii and hij
    def build_final_matrix(self,batch_data):
        atom_start = 0
        atom_pair_start = 0
        rebuildfocks = []
        gt_focks = []
        for idx,n_atom in enumerate(batch_data.molecule_size.reshape(-1)):
            n_atom = n_atom.item()
            Z = batch_data.atomic_numbers[atom_start:atom_start+n_atom]
            diag = batch_data["pred_hamiltonian_diagonal_blocks"][atom_start:atom_start+n_atom]
            non_diag = batch_data["pred_hamiltonian_non_diagonal_blocks"][atom_pair_start:atom_pair_start+n_atom*(n_atom-1)//2]
            rebuildfock = block2matrix(Z,diag,non_diag,self.mask_lin,self.conv.max_block_size,sym = True)
            
            diag = batch_data["diag_hamiltonian"][atom_start:atom_start+n_atom]
            non_diag = batch_data["non_diag_hamiltonian"][atom_pair_start:atom_pair_start+n_atom*(n_atom-1)//2]
            fock = block2matrix(Z,diag,non_diag,self.mask_lin,self.conv.max_block_size,sym = True)
            
            
            atom_start += n_atom
            atom_pair_start += n_atom*(n_atom-1)//2
            rebuildfocks.append(rebuildfock)
            gt_focks.append(fock)
        batch_data["pred_hamiltonian"] = rebuildfocks
        batch_data["hamiltonian"] = gt_focks
            
        return rebuildfocks
        
    def forward(self, data):
        if 'fii' not in data.keys() or "fij" not in data.keys():
            full_edge_index = get_full_graph(data)
            # Symmetry
            data["non_diag_hamiltonian"] = data["non_diag_hamiltonian"][full_edge_index[0]>full_edge_index[1]]
            data['non_diag_mask'] = data["non_diag_mask"][full_edge_index[0]>full_edge_index[1]]
            full_edge_index = full_edge_index[:,full_edge_index[0]>full_edge_index[1]]
            data["full_edge_index"] = full_edge_index
            
            full_edge_vec = data.pos[full_edge_index[0].long()] - data.pos[full_edge_index[1].long()]
            data.full_edge_attr = self.rbf(full_edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type(data.pos.type())
            data.full_edge_sh = o3.spherical_harmonics(
                    self.sh_irrep, full_edge_vec[:, [1, 2, 0]],
                    normalize=True, normalization='component').type(data.pos.type())

            node_features = data['node_vec2']
            fii = None
            fij = None

            # diagonal and non-diagonal layers
            for layer_idx in range(self.num_layer):
                if layer_idx == 0:
                    fii = self.e3_gnn_node_layer[layer_idx](data, node_features, None)
                    fij = self.e3_gnn_node_pair_layer[layer_idx](data, node_features, None)
                else:
                    fii = self.e3_gnn_node_layer[layer_idx](data, data['node_vec'], fii)
                    fij = self.e3_gnn_node_pair_layer[layer_idx](data, data['node_vec'], fij)

            fii = self.output_ii(fii)
            fij = self.output_ij(fij)

            data['fii'], data['fij'] = fii, fij

        fii = data["fii"]
        fij = data["fij"]

        node_attr  = data["node_attr"]

        data['ptr'] = torch.cat([torch.Tensor([0]).to(data["molecule_size"].device).int(),
                              torch.cumsum(data["molecule_size"],dim = 0)],dim = 0)
        full_dst, full_src = data.full_edge_index

        hamiltonian_diagonal_matrix = self.expand_ii['hamiltonian'](
            fii, self.fc_ii['hamiltonian'](node_attr), self.fc_ii_bias['hamiltonian'](node_attr))
        node_pair_embedding = node_attr[full_dst] + node_attr[full_src]
        hamiltonian_non_diagonal_matrix = self.expand_ij['hamiltonian'](
            fij, self.fc_ij['hamiltonian'](node_pair_embedding),
            self.fc_ij_bias['hamiltonian'](node_pair_embedding),use_sparse_tp=self.use_sparse_tp)
        data['pred_hamiltonian_diagonal_blocks'] = hamiltonian_diagonal_matrix
        data['pred_hamiltonian_non_diagonal_blocks'] = hamiltonian_non_diagonal_matrix

        return data


class SPHNet(nn.Module):
    def __init__(self,
                 order = 4,
                 embedding_dimension=128,
                 bottle_hidden_size=32,
                 max_radius=15,
                 radius_embed_dim=32,
                 use_equi_norm = False,
                 use_sparse_tp=False,
                 sparsity=0.7,
                 short_cutoff_upper=4,
                 long_cutoff_upper=9,
                 num_scale_atom_layers=3,
                 num_long_range_layers=3,
                 **kwargs):  # maximum nuclear charge (+1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        """
        Initialize the SPHNet model. Note that this class only contains the Vectorial Node Interaction Blocks and the Spherical Node Interaction Blocks.
        For the Pair Construction Blocks, see Pair_construction_layer.

        Args:
            order (int): The order of the spherical harmonics.
            embedding_dimension (int): The size of the hidden layer.
            bottle_hidden_size (int): The size of the bottleneck hidden layer.
            max_radius (int): The maximum radius cutoff.
            use_equi_norm (bool): Wether use equivariant normalization.
            short_cutoff_upper (float): the upper cutoff of short range interaction (Ångstrom).
            long_cutoff_upper: the upper cutoff of long range interaction (Ångstrom).
            num_scale_atom_layers: number of short range interaction module.
            num_long_range_layers: number of long range interaction module.
            radius_embed_dim (int): The dimension of the radius embedding.
            use_sparse_tp: wether use sparse gate.
            sparsity: sparse rate of the sparse gates (both pair and tp gates).
        """
        
        super(SPHNet, self).__init__()
        self.order = order
        self.input_order = 0 
        self.sh_irrep = o3.Irreps.spherical_harmonics(lmax=self.order)
        self.hs = embedding_dimension
        self.hbs = bottle_hidden_size
        self.radius_embed_dim = radius_embed_dim
        self.max_radius = max_radius
        
        self.init_sph_irrep = o3.Irreps(construct_o3irrps(1, order=order))
        
        self.irreps_node_embedding = construct_o3irrps_base(self.hs, order=order)
        self.hidden_irrep = o3.Irreps(construct_o3irrps(self.hs, order=order))
        self.hidden_irrep_mid = o3.Irreps(construct_o3irrps(self.hs, order=int(order/2)))
        self.hidden_irrep_base = o3.Irreps(self.irreps_node_embedding)
        self.hidden_bottle_irrep = o3.Irreps(construct_o3irrps(self.hbs, order=order))
        self.hidden_bottle_irrep_base = o3.Irreps(construct_o3irrps_base(self.hbs, order=order))
        
        self.input_irrep = o3.Irreps(construct_o3irrps(self.hs, order=self.input_order))
        self.radial_basis_functions = ExponentialBernsteinRadialBasisFunctions(self.radius_embed_dim, self.max_radius)
        self.nonlinear_scalars = {1: "ssp", -1: "tanh"}
        self.nonlinear_gates = {1: "ssp", -1: "abs"}
        self.num_fc_layer = 1

        self.Spherical_interaction = nn.ModuleList()
        print('use_equi_norm: ',use_equi_norm)
        for i in range(2):
            input_irrep = self.input_irrep if i == 0 else self.hidden_irrep
            self.Spherical_interaction.append(ConvNetLayer(
                irrep_in_node=input_irrep,
                irrep_hidden=self.hidden_irrep,
                irrep_out=self.hidden_irrep,
                edge_attr_dim=self.radius_embed_dim,
                node_attr_dim=self.hs,
                sh_irrep=self.sh_irrep,
                resnet=True,
                use_norm_gate=True if i != 0 else False,
                use_equi_norm=use_equi_norm,
                use_pair_sparse=use_sparse_tp if i != 0 else False,  # apply sparse pair gate for the second spherical layer
                use_sparse_tp=False,
                sparsity=sparsity,
            ))

        self.Vectorial_interaction = Visnorm_shared_LSRMNorm2_2branchSerial(
            hidden_channels=embedding_dimension,
            num_layers=num_scale_atom_layers,
            long_num_layers=num_long_range_layers,
            short_cutoff_upper=short_cutoff_upper,
            long_cutoff_lower=0,
            long_cutoff_upper=long_cutoff_upper,
        )
        
        
    def reset_parameters(self):
        warnings.warn("reset parameter is not init in SPHNet backbone model")

    def forward(self, batch_data):
        batch_data['ptr'] = torch.cat([torch.Tensor([0]).to(batch_data["molecule_size"].device).int(),
                              torch.cumsum(batch_data["molecule_size"],dim = 0)],dim = 0)

        # vectorial layers
        batch_data['natoms'] = scatter(torch.ones_like(batch_data.batch), batch_data.batch, dim=0, reduce='sum')
        batch_data.atomic_numbers = batch_data.atomic_numbers.squeeze()
        
        node_attr = self.Vectorial_interaction(batch_data)

        # spherical layers
        edge_index = radius_graph(batch_data.pos, self.max_radius, batch_data.batch)#, max_num_neighbors=100)

        edge_vec = batch_data.pos[edge_index[0].long()] - batch_data.pos[edge_index[1].long()]
        rbf_new = self.radial_basis_functions(edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type(batch_data.pos.type())
        edge_sh = o3.spherical_harmonics(
            self.sh_irrep, edge_vec[:, [1, 2, 0]],
            normalize=True, normalization='component').type(batch_data.pos.type())

        batch_data.node_attr, batch_data.edge_index, batch_data.edge_attr, batch_data.edge_sh = \
            node_attr, edge_index, rbf_new, edge_sh

        node_vec2 = self.Spherical_interaction[-2](batch_data, node_attr)
        node_vec1 = self.Spherical_interaction[-1](batch_data, node_vec2)

        batch_data["node_embedding"] = batch_data.node_attr
        batch_data["node_vec"] = node_vec1 
        batch_data["node_vec2"] = node_vec2  
        
        # return the atom features
        return batch_data
