import time

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
from .sparse_tp import ExpansionFX as Expansion
from .sparse_tp import Sparse_TensorProduct

import random

from .utils import construct_o3irrps_base, construct_o3irrps, get_full_graph, get_conv_variable, block2matrix, get_transpose_index


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

class HamiHead_QHNet(nn.Module):
    def __init__(self,
                 irrep_in_node =  "128x0e + 128x1e + 128x2e + 128x3e + 128x4e",
                 irreps_edge_embedding = "32x0e + 32x1e + 32x2e + 32x3e + 32x4e",
                 order = 4,
                 pyscf_basis_name = "def2-svp",
                 radius_embed_dim=64,
                 max_radius_cutoff=15,
                 bottle_hidden_size=64,
                 num_layer=2,
                 num_nodes = 20,
                 use_sparse_tp=False,
                 ckpt_path=None,
                 sparsity=0.7,
                 steps_1epoch=1700,
                 ):
        super().__init__()

        # #convert the model to the correct dtype
        # self.model.to(torch.float32)
        
        self.hs = o3.Irreps(irrep_in_node)[0][0]

        if pyscf_basis_name == 'def2-svp':
            exp_irrp = o3.Irreps("3x0e + 2x1e + 1x2e")
        elif pyscf_basis_name == 'def2-tzvp':
            exp_irrp = o3.Irreps("5x0e + 5x1e + 2x2e + 1x3e")
        else:
            raise ValueError('invalid base')
        self.conv,_,self.mask_lin,_ = get_conv_variable(pyscf_basis_name)
        self.node_embedding = nn.Embedding(num_nodes, self.hs)

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
                nn.Linear(self.hs * 2, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_path_weight)
            )

            self.fc_ij_bias[name] = torch.nn.Sequential(
                nn.Linear(self.hs * 2, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_bias)
            )

        self.num_layer = num_layer
        self.sh_irrep = o3.Irreps.spherical_harmonics(lmax=self.order)
        self.radius_embed_dim = radius_embed_dim

        self.rbf = ExponentialBernsteinRadialBasisFunctions(self.radius_embed_dim, max_radius_cutoff)
        self.hbs = bottle_hidden_size # 
        self.hidden_irrep = o3.Irreps(construct_o3irrps(self.hs, order=self.order))
        self.hidden_irrep_base = o3.Irreps(construct_o3irrps_base(self.hs, order=self.order))
        self.hidden_bottle_irrep = o3.Irreps(construct_o3irrps(self.hbs, order=self.order))

        self.e3_gnn_node_pair_layer = nn.ModuleList()
        self.e3_gnn_node_layer = nn.ModuleList()
        irrep_in_node = o3.Irreps(str(irrep_in_node).replace('o', 'e'))
        for l in range(self.num_layer):
            if l == 0:
                self.e3_gnn_node_layer.append(SelfNetLayer(
                    irrep_in_node=irrep_in_node,
                    irrep_bottle_hidden=self.hidden_irrep_base,
                    irrep_out=self.hidden_irrep_base,
                    sh_irrep=self.sh_irrep,
                    edge_attr_dim=self.radius_embed_dim,
                    node_attr_dim=self.hs,
                    # tp_mode = "uuw",
                    sparsity=sparsity,
                    use_sparse_tp=use_sparse_tp,
                    resnet=False,
                ))
                self.e3_gnn_node_pair_layer.append(PairNetLayer(
                    irrep_in_node=irrep_in_node,
                    irrep_bottle_hidden=self.hidden_irrep_base,
                    irrep_out=self.hidden_irrep_base,
                    sh_irrep=self.sh_irrep,
                    edge_attr_dim=self.radius_embed_dim,
                    node_attr_dim=self.hs,
                    invariant_layers=1,
                    invariant_neurons=self.hs,
                    # tp_mode = "uuw",
                    sparsity=sparsity,
                    use_sparse_tp=use_sparse_tp,
                    resnet=False,
                ))
            else:
                self.e3_gnn_node_layer.append(SelfNetLayer(
                    irrep_in_node=irrep_in_node,
                    irrep_bottle_hidden=self.hidden_irrep_base,
                    irrep_out=self.hidden_irrep_base,
                    sh_irrep=self.sh_irrep,
                    edge_attr_dim=self.radius_embed_dim,
                    node_attr_dim=self.hs,
                    # tp_mode = "uuw",
                    use_sparse_tp=use_sparse_tp,
                    resnet=True,
                ))
                self.e3_gnn_node_pair_layer.append(PairNetLayer(
                    irrep_in_node=irrep_in_node,
                    irrep_bottle_hidden=self.hidden_irrep_base,
                    irrep_out=self.hidden_irrep_base,
                    sh_irrep=self.sh_irrep,
                    edge_attr_dim=self.radius_embed_dim,
                    node_attr_dim=self.hs,
                    invariant_layers=1,
                    invariant_neurons=self.hs,
                    # tp_mode = "uuw",
                    use_sparse_tp=use_sparse_tp,
                    resnet=True,    
                ))        

        self.output_ii = o3.Linear(self.hidden_irrep, self.hidden_bottle_irrep)
        self.output_ij = o3.Linear(self.hidden_irrep, self.hidden_bottle_irrep)

    def reset_parameters(self):
        warnings.warn("reset parameter is not init in hami head model")
    
    def to(self, device):
        super().to(device)

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

    def build_graph(self, data, max_radius):
        node_attr = data.atomic_numbers.squeeze()
        radius_edges = radius_graph(data.pos, max_radius, data.batch, max_num_neighbors=1000)

        dst, src = radius_edges
        edge_vec = data.pos[dst.long()] - data.pos[src.long()]
        rbf = self.rbf(edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type(data.pos.type())

        edge_sh = o3.spherical_harmonics(
            self.sh_irrep, edge_vec[:, [1, 2, 0]],
            normalize=True, normalization='component').type(data.pos.type())

        start_edge_index = 0
        all_transpose_index = []
        for graph_idx in range(data.ptr.shape[0] - 1):
            num_nodes = data.ptr[graph_idx +1] - data.ptr[graph_idx]
            graph_edge_index = radius_edges[:, start_edge_index:start_edge_index+num_nodes*(num_nodes-1)]
            sub_graph_edge_index = graph_edge_index - data.ptr[graph_idx]
            bias = (sub_graph_edge_index[0] < sub_graph_edge_index[1]).type(torch.int)
            transpose_index = sub_graph_edge_index[0] * (num_nodes - 1) + sub_graph_edge_index[1] - bias
            transpose_index = transpose_index + start_edge_index
            all_transpose_index.append(transpose_index)
            start_edge_index = start_edge_index + num_nodes*(num_nodes-1)

        return node_attr, radius_edges, rbf, edge_sh, torch.cat(all_transpose_index, dim=-1)
    

    def forward(self, data):
        if 'fii' not in data.keys():
            full_edge_index = get_full_graph(data)
            data["full_edge_index"] = full_edge_index
            
            full_edge_vec = data.pos[full_edge_index[0].long()] - data.pos[full_edge_index[1].long()]
            data.full_edge_attr = self.rbf(full_edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type(data.pos.type())
            data.full_edge_sh = o3.spherical_harmonics(
                    self.sh_irrep, full_edge_vec[:, [1, 2, 0]],
                    normalize=True, normalization='component').type(data.pos.type())

            # full_edge_info = self.build_graph(data, 10_000_000)
            # (_, data.full_edge_index, data.full_edge_attr, 
            # data.full_edge_sh, data.transpose_edge_index) = full_edge_info


            node_features = data['node_vec2']
            fii = None
            fij = None

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
        
        node_attr = data["node_attr"]
        fii = data["fii"]
        fij = data["fij"]
        data['ptr'] = torch.cat([torch.Tensor([0]).to(data["molecule_size"].device).int(),
                              torch.cumsum(data["molecule_size"],dim = 0)],dim = 0)
        full_dst, full_src = data.full_edge_index
        # transpose_edge_index = get_transpose_index(data, data.full_edge_index)
        transpose_edge_index = data.transpose_edge_index

        hamiltonian_diagonal_matrix = self.expand_ii['hamiltonian'](
            fii, self.fc_ii['hamiltonian'](node_attr), self.fc_ii_bias['hamiltonian'](node_attr))
        node_pair_embedding = torch.cat([node_attr[full_dst], node_attr[full_src]], dim=-1)
        hamiltonian_non_diagonal_matrix = self.expand_ij['hamiltonian'](
            fij, self.fc_ij['hamiltonian'](node_pair_embedding),
            self.fc_ij_bias['hamiltonian'](node_pair_embedding))


        ret_hamiltonian_diagonal_matrix = 0.5*(hamiltonian_diagonal_matrix +
                                        hamiltonian_diagonal_matrix.transpose(-1, -2))

        # the transpose should considers the i, j
        ret_hamiltonian_non_diagonal_matrix = 0.5*(hamiltonian_non_diagonal_matrix + 
                    hamiltonian_non_diagonal_matrix[transpose_edge_index].transpose(-1, -2))

        data['pred_hamiltonian_diagonal_blocks'] = ret_hamiltonian_diagonal_matrix
        data['pred_hamiltonian_non_diagonal_blocks'] = ret_hamiltonian_non_diagonal_matrix
        return data

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
        
        # for idx,ins in enumerate(instruction_node):
        #     if random.random() < 0.7:
        #         instruction_node[idx] = (ins[0], ins[1], ins[2], ins[3], ins[4], 0.00)

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
            # self.norm = EquivariantLayerNormArray(self.lmax,self.irrep_tp_out_node[0][0])
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

        if self.use_pair_sparse:
            pair_weight = self.pair_select_layer(s0)
            quantile = torch.quantile(pair_weight, q=self.pair_sparsity)
            pair_weight = torch.sigmoid(pair_weight-quantile)
            pair_weight = torch.where(pair_weight < 0.5, torch.zeros_like(pair_weight), pair_weight)
            path_mask = torch.where(pair_weight!=0)[0]
            # path_mask = torch.where(pair_weight>=quantile)[0]
            edge_src = edge_src[path_mask]
            edge_dst = edge_dst[path_mask]
            s0 = (s0*pair_weight)[path_mask]
            data.edge_sh = data.edge_sh[path_mask]
            data.edge_attr = data.edge_attr[path_mask]

        edge_features = self.tp_node(
            x[edge_src], data.edge_sh, self.fc_node(data.edge_attr) * self.layer_l0(s0))#[:,self.indice], torch.randint(0,1,(len(self.tp_node.instructions),)))

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
        
        # for idx,ins in enumerate(instr):
        #     if random.random() < 0.7:
        #         instr[idx] = (ins[0], ins[1], ins[2], ins[3], ins[4], 0.00)
        self.tp = o3.TensorProduct(self.irrep_in, self.irrep_in, irrep_out, instr, irrep_normalization="component")
        self.irrep_out = irrep_out.simplify()

    def forward(self, features_1, features_2):
        # for idx,ins in enumerate(self.tp.instructions):
        #     if random.random() < 0.7:
        #         self.tp.instructions[idx] = Instruction(ins[0], ins[1], ins[2], ins[3], ins[4], 0.00, ins[6])
        # out = self.tp(features_1, features_2, path_mask = torch.randint(0,1,(len(self.tp.instructions),)))
        # out = self.tp(features_1, features_2, path_mask = torch.bernoulli(torch.full((len(self.tp.instructions),), 0.7)).long())
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
            resnet: bool = True,
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


class PairNetLayer(torch.nn.Module):
    def __init__(self,
                 irrep_in_node,
                 irrep_bottle_hidden,
                 irrep_out,
                 sh_irrep,
                 edge_attr_dim,
                 node_attr_dim,
                 resnet: bool = True,
                 invariant_layers=1,
                 invariant_neurons=8,
                 tp_mode = "uuu",
                 nonlinear='ssp',
                 use_sparse_tp=False,
                 sparsity = 0.7,):
        super(PairNetLayer, self).__init__()
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
        self.irrep_tp_out_node_pair, instruction_node_pair = get_feasible_irrep(
            self.irrep_tp_in_node, self.irrep_tp_in_node, self.irrep_bottle_hidden, tp_mode=tp_mode)

        # self.irrep_tp_out_node_pair_msg, instruction_node_pair_msg = get_feasible_irrep(
        #     self.irrep_tp_in_node, self.sh_irrep, self.irrep_bottle_hidden, tp_mode='uvu')

        self.linear_node_pair = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_tp_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        self.linear_node_pair_n = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_tp_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )
        self.linear_node_pair_inner = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        # for idx,ins in enumerate(instruction_node_pair):
        #     if random.random() < 0.7:
        #         instruction_node_pair[idx] = (ins[0], ins[1], ins[2], ins[3], ins[4], 0.00)

        if use_sparse_tp:
            self.tp_node_pair = Sparse_TensorProduct(
                self.irrep_tp_in_node,
                self.irrep_tp_in_node,
                self.irrep_tp_out_node_pair,
                instruction_node_pair,
                shared_weights=False,
                internal_weights=False,
                sparsity=sparsity,
            )
            # len_ins = len(self.irrep_tp_out_node_pair)
            # path_mask = random.sample(range(len_ins),int(len_ins*(1-sparsity))+1)
            # instruction_weight = torch.tensor([ins.path_weight for ins in self.tp_node_pair.instructions])#.to(x.device)
            # instructions = []
            # self.indice = []
            # count = 0
            # path_weight_sum = instruction_weight[path_mask].sum()/instruction_weight.sum()
            # for i in range(len(self.tp_node_pair.instructions)):
            #     current_path = prod(self.tp_node_pair.instructions[i].path_shape)
            #     if i in path_mask:
            #         instructions.append(self.tp_node_pair.instructions[i]._replace(path_weight=self.tp_node_pair.instructions[i].path_weight/path_weight_sum))
            #         self.indice+=list(range(count,count+current_path))
            #     count += current_path
            # self.reset_tp(instructions)
        else:
            self.tp_node_pair = TensorProduct(
                self.irrep_tp_in_node,
                self.irrep_tp_in_node,
                self.irrep_tp_out_node_pair,
                instruction_node_pair,
                shared_weights=False,
                internal_weights=False,
            )

            # self.filter_ir_out = e3nn.s2_irreps(
            #             self.irrep_in_node.irreps.lmax, p_val=1 * 1, p_arg=-1
            #         )
            # gaunt_tensor_product_fourier_2D_fn = lambda x1, x2, filter_ir_out: functional.gaunt_tensor_product_fourier_2D(
            #     x1, x2,
            #     res_theta=6, res_phi=6, convolution_type="direct", filter_ir_out=filter_ir_out
            # )
            # self.tp_node_pair = jax.jit(gaunt_tensor_product_fourier_2D_fn)


        self.irrep_tp_out_node_pair_2, instruction_node_pair_2 = get_feasible_irrep(
            self.irrep_tp_out_node_pair, self.irrep_tp_out_node_pair, self.irrep_bottle_hidden, tp_mode='uuu')

        # for idx,ins in enumerate(instruction_node_pair_2):
        #     if random.random() < 0.7:
        #         instruction_node_pair_2[idx] = (ins[0], ins[1], ins[2], ins[3], ins[4], 0.00)
        self.tp_node_pair_2 = Sparse_TensorProduct(
            self.irrep_tp_out_node_pair,
            self.irrep_tp_out_node_pair,
            self.irrep_tp_out_node_pair_2,
            instruction_node_pair_2,
            shared_weights=True,
            internal_weights=True
        )


        self.fc_node_pair = FullyConnectedNet(
            [self.edge_attr_dim] + invariant_layers * [invariant_neurons] + [self.tp_node_pair.weight_numel],
            self.nonlinear_layer
        )

        self.linear_node_pair_2 = Linear(
            irreps_in=self.irrep_tp_out_node_pair_2,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        if self.irrep_in_node == self.irrep_out and resnet:
            self.resnet = True
        else:
            self.resnet = False

        self.linear_node_pair = Linear(
            irreps_in=self.irrep_tp_out_node_pair,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )
        self.norm_gate = NormGate(self.irrep_tp_out_node_pair)
        self.inner_product = InnerProduct(self.irrep_in_node)
        self.norm = o3.Norm(self.irrep_in_node)
        num_mul = 0
        for mul, ir in self.irrep_in_node:
            num_mul = num_mul + mul

        self.norm_gate_pre = NormGate(self.irrep_in_node)
        self.fc = nn.Sequential(
            nn.Linear(self.irrep_in_node[0][0] + num_mul, self.irrep_in_node[0][0]),
            nn.SiLU(),
            nn.Linear(self.irrep_in_node[0][0], self.tp_node_pair.weight_numel))
        # self.path_layer = nn.Linear(self.irrep_tp_in_node[0][0]*2, len(self.tp_node_pair.instructions))
        # self.soft = nn.Softmax(len(self.tp_node_pair.instructions))

        self.count = 0

    @property
    def device(self):
        return next(self.parameters()).device

    def reset_tp(self, instructions):
        instruction_nodes = []
        for ins in instructions:
            instruction_nodes.append((ins.i_in1, ins.i_in2, ins.i_out, ins.connection_mode, ins.has_weight, ins.path_weight))
        self.tp_node_pair = TensorProduct(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_tp_out_node_pair,
            instruction_nodes,
            shared_weights=False,
            internal_weights=False,
        ).to(self.device)

    def forward(self, data, node_attr, node_pair_attr=None):
        dst, src = data['full_edge_index']
        node_attr_0 = self.linear_node_pair_inner(node_attr)
        s0 = self.inner_product(node_attr_0[dst], node_attr_0[src])[:, self.irrep_in_node.slices()[0].stop:]
        s0 = torch.cat([node_attr_0[dst][:, self.irrep_in_node.slices()[0]],
                        node_attr_0[src][:, self.irrep_in_node.slices()[0]], s0], dim=-1)

        node_attr = self.norm_gate_pre(node_attr)
        node_attr = self.linear_node_pair_n(node_attr)

        node_pair = self.tp_node_pair(node_attr[src], node_attr[dst],
            self.fc_node_pair(data['full_edge_attr']) * self.fc(s0))
        # node_pair = self.tp_node_pair(node_attr[src], node_attr[dst], self.filter_ir_out)

        # self.count += 1
        # if self.count % 100 == 0:
        #     print("pairnet mask: ", path_mask)

        # node_pair = self.tp_node_pair(node_attr[src], node_attr[dst],    # erpai's version
        #     self.fc_node_pair(data.full_edge_attr) * self.fc(s0), path_mask = torch.bernoulli(torch.full((len(self.tp_node_pair.instructions),), 0.85)).long())
        # node_pair = self.tp_node_pair(node_attr[src], node_attr[dst],    # xinran's version
        #     self.fc_node_pair(data.full_edge_attr) * self.fc(s0), path_mask = torch.randint(0,1,(len(self.tp_node_pair.instructions),)))

        # node_pair = self.tp_node_pair(node_attr[src], node_attr[dst],    # original
        #     self.fc_node_pair(data.full_edge_attr) * self.fc(s0))


        node_pair = self.norm_gate(node_pair)
        node_pair = self.linear_node_pair(node_pair)

        if self.resnet and node_pair_attr is not None:
            node_pair = node_pair + node_pair_attr
        return node_pair


class SelfNetLayer(torch.nn.Module):
    def __init__(self,
                 irrep_in_node,
                 irrep_bottle_hidden,
                 irrep_out,
                 sh_irrep,
                 edge_attr_dim,
                 node_attr_dim,
                 resnet: bool = True,
                 tp_mode = "uuu",
                 nonlinear='ssp',
                 use_sparse_tp=False,
                 sparsity=0.7,):
        super(SelfNetLayer, self).__init__()
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
        
        # for idx,ins in enumerate(instruction_node):
        #     if random.random() < 0.7:
        #         instruction_node[idx] = (ins[0], ins[1], ins[2], ins[3], ins[4], 0.00)

        if use_sparse_tp:
            self.tp = Sparse_TensorProduct(
                self.irrep_tp_in_node,
                self.irrep_tp_in_node,
                self.irrep_tp_out_node,
                instruction_node,
                shared_weights=True,
                internal_weights=True,
                sparsity=sparsity,
            )
            # len_ins = len(instruction_node)
            # path_mask = random.sample(range(len_ins),int(len_ins*(1-sparsity))+1)
            # instruction_weight = torch.tensor([ins.path_weight for ins in self.tp.instructions])#.to(x.device)
            # instructions = []
            # self.indice = []
            # count = 0
            # path_weight_sum = instruction_weight[path_mask].sum()/instruction_weight.sum()
            # for i in range(len(self.tp.instructions)):
            #     current_path = prod(self.tp.instructions[i].path_shape)
            #     if i in path_mask:
            #         instructions.append(self.tp.instructions[i]._replace(path_weight=self.tp.instructions[i].path_weight/path_weight_sum))
            #         self.indice+=list(range(count,count+current_path))
            #     count += current_path
            # self.reset_tp(instructions)
        else:
            self.tp = TensorProduct(
                self.irrep_tp_in_node,
                self.irrep_tp_in_node,
                self.irrep_tp_out_node,
                instruction_node,
                shared_weights=True,
                internal_weights=True
            )
            # self.tp = functional.gaunt_tensor_product_fourier_2D
            # self.lmax1=self.irrep_in_node.lmax
            # self.lmax2=self.irrep_in_node.lmax
            # self.y1_grid = torch.load(f'/home/weixinran/erpailuo/se_3_equivariant_prediction/PriceofFreedom-main/src/tensor_products/grid/grid_{int(self.lmax1)}.pt')
            # self.y2_grid = torch.load(f'/home/weixinran/erpailuo/se_3_equivariant_prediction/PriceofFreedom-main/src/tensor_products/grid/grid_{int(self.lmax1)}.pt')
            # self.z_grid = torch.load(f'/home/weixinran/erpailuo/se_3_equivariant_prediction/PriceofFreedom-main/src/tensor_products/grid/grid_{int(self.lmax1*2)}.pt')
            # self.grid_device = True

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
        # self.path_layer = nn.Linear(self.irrep_tp_in_node[0][0]*2, len(self.tp.instructions))
        # self.soft = nn.Softmax(len(self.tp.instructions))

        self.step = 0
    
    def reset_tp(self, instructions, weight):
        instruction_nodes = []
        for ins in instructions:
            instruction_nodes.append((ins.i_in1, ins.i_in2, ins.i_out, ins.connection_mode, ins.has_weight, ins.path_weight**2))
        self.tp = TensorProduct(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_tp_out_node,
            instruction_nodes,
            shared_weights=True,
            internal_weights=True,
            path_normalization="none",
            irrep_normalization="none",
        ).to(self.device)
        self.tp.weight.data = weight

    def forward(self, data, x, old_fii):
        old_x = x
        xl = self.norm_gate_1(x)
        xl = self.linear_node_1(xl)
        xr = self.norm_gate_2(x)
        xr = self.linear_node_2(xr)

        # for idx,ins in enumerate(self.tp.instructions):
        #     if random.random() < 0.7:
        #         self.tp.instructions[idx] = Instruction(ins[0], ins[1], ins[2], ins[3], ins[4], 0.00, ins[6])

        # path_input = torch.concat((xl[:, self.irrep_tp_in_node.slices()[0]],xr[:, self.irrep_tp_in_node.slices()[0]]),dim=-1)
        # path_weight = self.path_layer(path_input).sum(axis=0)   # learnable version
        # path_mask = (path_weight>torch.quantile(path_weight, 0.7)).long().to(path_input.device)
        # x = self.tp(xl, xr)  

        if self.step%600 == 1 and self.step<-5100:
            x, path_mask_tp = self.tp(xl, xr, return_path=True)
            if old_fii is None:
                print('layer1 selfnet: ',path_mask_tp)
            else:
                print('layer2 selfnet: ',path_mask_tp)
        else:
            x = self.tp(xl, xr)
        self.step += 1

        # if self.grid_device:
        #     self.y1_grid = self.y1_grid.to(xl.device)
        #     self.y2_grid = self.y2_grid.to(xl.device)
        #     self.z_grid = self.z_grid.to(xl.device)
        #     self.grid_device = False
        # x = self.tp(xl, xr, y1_grid=self.y1_grid, y2_grid=self.y2_grid, z_grid=self.z_grid,
        #             res_theta=6, res_phi=6, convolution_type="direct",
        #             lmax1=self.lmax1,lmax2=self.lmax2)

        if isinstance(x, tuple):
            print('dynamic sparse tensor product finished, returning to fixed graph tensor product')
            self.reset_tp(x[0],x[3])
            x = x[1]

        # self.count += 1
        # if self.count % 100 == 0:
        #     print("selfnet mask: ", path_mask)
        
        # x = self.tp(xl, xr, path_mask = torch.bernoulli(torch.full((len(self.tp.instructions),), 0.85)).long())  # erpai's version
        # x = self.tp(xl, xr, path_mask = torch.randint(0,1,(len(self.tp.instructions),)))  # xinran's version

        # x = self.tp(xl, xr)   # original

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


# class Expansion(nn.Module):
#     def __init__(self, irrep_in, irrep_out_1, irrep_out_2):
#         super(Expansion, self).__init__()
#         self.irrep_in = irrep_in
#         self.irrep_out_1 = irrep_out_1
#         self.irrep_out_2 = irrep_out_2
#         self.instructions = self.get_expansion_path(irrep_in, irrep_out_1, irrep_out_2)

#         # filter_ratio = 0.3
#         # self.instructions = random.sample(self.instructions,math.ceil(len(self.instructions)*filter_ratio))

#         self.num_path_weight = sum(prod(ins[-1]) for ins in self.instructions if ins[3])
#         self.num_bias = sum([prod(ins[-1][1:]) for ins in self.instructions if ins[0] == 0])
#         if self.num_path_weight > 0:
#             self.weights = nn.Parameter(torch.rand(self.num_path_weight + self.num_bias))
#         self.num_weights = self.num_path_weight + self.num_bias
#         # self.path_layer = nn.Linear(self.irrep_in[0][0], len(self.instructions))
#         # self.tp_times=1
#         # self.sparsity=0.007

#     def pick_path(self, path_weight, sparsity): 
#         # 计算总共需要选择的元素数量（30%的tensor大小）  
#         num_elements = path_weight.numel()  
#         num_elements_to_select = int(sparsity * num_elements)+1  
#         _, selected_indices = torch.topk(path_weight, num_elements_to_select)  

#         return selected_indices        

#     def forward(self, x_in, weights=None, bias_weights=None, use_sparse_tp=False):
#         batch_num = x_in.shape[0]
#         if len(self.irrep_in) == 1:
#             x_in_s = [x_in.reshape(batch_num, self.irrep_in[0].mul, self.irrep_in[0].ir.dim)]
#         else:
#             x_in_s = [
#                 x_in[:, i].reshape(batch_num, mul_ir.mul, mul_ir.ir.dim)
#             for i, mul_ir in zip(self.irrep_in.slices(), self.irrep_in)]

#         # use_sparse_tp = False
#         # if use_sparse_tp:
#         #     # calculate path weight
#         #     path_input = x_in_s[0].squeeze(-1)
#         #     path_weight = self.path_layer(path_input).sum(axis=0)   # learnable version

#         #     if self.tp_times < 20000:
#         #         # min_val = torch.min(path_weight)  
#         #         # range_val = torch.max(path_weight) - min_val  
#         #         # scaled_tensor = (path_weight - min_val) / range_val  
#         #         # path_weight = nn.Softmax(dim=0)(scaled_tensor)
#         #         path_mask = self.pick_path(path_weight,sparsity=1-self.sparsity) # pick 30%
#         #         if self.tp_times % 200 == 0:
#         #             self.sparsity += 0.007
#         #     else:
#         #         _, path_mask = torch.topk(path_weight, int(path_weight.shape[0]*(1-0.7))+1) # pick top 30%
            
#         #     self.tp_times += 1

#         #     # pick instruction path
#         #     instructions = []
#         #     indice = []
#         #     bias_indice = []
#         #     count = 0
#         #     bias_count = 0
#         #     for i in range(len(self.instructions)):
#         #         current_path = prod(self.instructions[i][5])
#         #         if self.instructions[i][0] == 0:
#         #             current_bias_path = prod(self.instructions[i][-1][1:])
#         #             if i in path_mask:
#         #                 bias_indice+=list(range(bias_count,bias_count+current_bias_path))
#         #             bias_count += current_bias_path
#         #         if i in path_mask:
#         #             instructions.append(self.instructions[i])
#         #             indice+=list(range(count,count+current_path))
#         #         count += current_path

#         #     weights = weights[indice] if len(weights.shape)==1 else weights[:,indice]
#         #     bias_weights = bias_weights[bias_indice] if len(bias_weights.shape)==1 else bias_weights[:,bias_indice]
#         # else:
#         instructions = self.instructions

#         outputs = {}
#         flat_weight_index = 0
#         bias_weight_index = 0
#         for ins in instructions:
#             mul_ir_in = self.irrep_in[ins[0]]
#             mul_ir_out1 = self.irrep_out_1[ins[1]]
#             mul_ir_out2 = self.irrep_out_2[ins[2]]
#             x1 = x_in_s[ins[0]]
#             x1 = x1.reshape(batch_num, mul_ir_in.mul, mul_ir_in.ir.dim)
#             w3j_matrix = o3.wigner_3j(ins[1], ins[2], ins[0]).to(self.device).type(x1.type())
#             if ins[3] is True or weights is not None:
#                 if weights is None:
#                     weight = self.weights[flat_weight_index:flat_weight_index + prod(ins[-1])].reshape(ins[-1])
#                     result = torch.einsum(
#                         f"wuv, ijk, bwk-> buivj", weight, w3j_matrix, x1) / mul_ir_in.mul
#                 else:
#                     ### weights:  w u v -> hidden duplicate_order_orbital duplicate_order_orbital
#                     ## x1 : b w k -> edge , hidden, l3
#                     ## w3jmatrix:  i,j,k -> l1,l2,3
#                     weight = weights[:, flat_weight_index:flat_weight_index + prod(ins[-1])].reshape([-1] + ins[-1])
#                     result = torch.einsum(f"bwuv, bwk-> buvk", weight, x1)
#                     if ins[0] == 0 and bias_weights is not None:
#                         bias_weight = bias_weights[:,bias_weight_index:bias_weight_index + prod(ins[-1][1:])].\
#                             reshape([-1] + ins[-1][1:])
#                         bias_weight_index += prod(ins[-1][1:])
#                         result = result + bias_weight.unsqueeze(-1)
#                     result = torch.einsum(f"ijk, buvk->buivj", w3j_matrix, result) / mul_ir_in.mul
#                 flat_weight_index += prod(ins[-1])
#             else:
#                 result = torch.einsum(
#                     f"uvw, ijk, bwk-> buivj", torch.ones(ins[-1]).type(x1.type()).to(self.device), w3j_matrix,
#                     x1.reshape(batch_num, mul_ir_in.mul, mul_ir_in.ir.dim)
#                 )

#             result = result.reshape(batch_num, mul_ir_out1.dim, mul_ir_out2.dim)
#             key = (ins[1], ins[2])
#             if key in outputs.keys():
#                 outputs[key] = outputs[key] + result
#             else:
#                 outputs[key] = result

#         rows = []
#         for i in range(len(self.irrep_out_1)):
#             blocks = []
#             for j in range(len(self.irrep_out_2)):
#                 if (i, j) not in outputs.keys():
#                     blocks += [torch.zeros((x_in.shape[0], self.irrep_out_1[i].dim, self.irrep_out_2[j].dim),
#                                            device=x_in.device).type(x_in.type())]
#                 else:
#                     blocks += [outputs[(i, j)]]
#             rows.append(torch.cat(blocks, dim=-1))
#         output = torch.cat(rows, dim=-2)
#         return output

#     def get_expansion_path(self, irrep_in, irrep_out_1, irrep_out_2):
#         instructions = []
#         for  i, (num_in, ir_in) in enumerate(irrep_in):
#             for  j, (num_out1, ir_out1) in enumerate(irrep_out_1):
#                 for k, (num_out2, ir_out2) in enumerate(irrep_out_2):
#                     if ir_in in ir_out1 * ir_out2:
#                         instructions.append([i, j, k, True, 1.0, [num_in, num_out1, num_out2]])
#         return instructions

#     @property
#     def device(self):
#         return next(self.parameters()).device

#     def __repr__(self):
#         return f'{self.irrep_in} -> {self.irrep_out_1}x{self.irrep_out_1} and bias {self.num_bias}' \
#                f'with parameters {self.num_path_weight}'


class QHNet(nn.Module):
    def __init__(self,
                 in_node_features=1,
                 order=4,
                 embedding_dimension=128,
                 bottle_hidden_size=32,
                 num_gnn_layers=5,
                 max_radius_cutoff=12,
                 num_nodes=20,
                 radius_embed_dim=32,
                 use_equi_norm=False,
                 **kwargs):  # maximum nuclear charge (+1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super(QHNet, self).__init__()
        # store hyperparameter values
        self.atom_orbs = [
            [[8, 0, '1s'], [8, 0, '2s'], [8, 0, '3s'], [8, 1, '2p'], [8, 1, '3p'], [8, 2, '3d']],
            [[1, 0, '1s'], [1, 0, '2s'], [1, 1, '2p']],
            [[1, 0, '1s'], [1, 0, '2s'], [1, 1, '2p']]
        ]
        self.order = order

        self.sh_irrep = o3.Irreps.spherical_harmonics(lmax=self.order)
        self.hs = embedding_dimension
        self.hbs = bottle_hidden_size
        self.radius_embed_dim = radius_embed_dim
        self.max_radius_cutoff = max_radius_cutoff
        self.num_gnn_layers = num_gnn_layers
        self.node_embedding = nn.Embedding(num_nodes, self.hs)
        if self.order == 4:
            self.hidden_irrep = o3.Irreps(f'{self.hs}x0e + {self.hs}x1o + {self.hs}x2e + {self.hs}x3o + {self.hs}x4e')
            self.hidden_irrep_base = o3.Irreps(f'{self.hs}x0e + {self.hs}x1e + {self.hs}x2e + {self.hs}x3e + {self.hs}x4e')
        elif self.order == 2:
            self.hidden_irrep = o3.Irreps(f'{self.hs}x0e + {self.hs}x1o + {self.hs}x2e')
            self.hidden_irrep_base = o3.Irreps(f'{self.hs}x0e + {self.hs}x1e + {self.hs}x2e')
        elif self.order == 6:
            self.hidden_irrep = o3.Irreps(f'{self.hs}x0e + {self.hs}x1o + {self.hs}x2e + {self.hs}x3o + {self.hs}x4e+{self.hs}x5o + {self.hs}x6e')
            self.hidden_irrep_base = o3.Irreps(f'{self.hs}x0e + {self.hs}x1e + {self.hs}x2e + {self.hs}x3e + {self.hs}x4e+ {self.hs}x5e + {self.hs}x6e')
 
        else:
            raise ValueError('invalid order')
        self.hidden_bottle_irrep = o3.Irreps(f'{self.hbs}x0e + {self.hbs}x1o + {self.hbs}x2e + {self.hbs}x3o + {self.hbs}x4e')
        self.hidden_bottle_irrep_base = o3.Irreps(
            f'{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e')
        self.final_out_irrep = o3.Irreps(f'{self.hs * 3}x0e + {self.hs * 2}x1o + {self.hs}x2e').simplify()
        self.input_irrep = o3.Irreps(f'{self.hs}x0e')
        self.radial_basis_functions = ExponentialBernsteinRadialBasisFunctions(self.radius_embed_dim, self.max_radius_cutoff)
        self.nonlinear_scalars = {1: "ssp", -1: "tanh"}
        self.nonlinear_gates = {1: "ssp", -1: "abs"}
        self.num_fc_layer = 1
        self.irreps_node_embedding = self.hidden_irrep

        self.e3_gnn_layer = nn.ModuleList()
        self.e3_gnn_node_pair_layer = nn.ModuleList()
        self.e3_gnn_node_layer = nn.ModuleList()
        self.udpate_layer = nn.ModuleList()
        self.start_layer = 2
        for i in range(self.num_gnn_layers):
            input_irrep = self.input_irrep if i == 0 else self.hidden_irrep
            self.e3_gnn_layer.append(ConvNetLayer(
                irrep_in_node=input_irrep,
                irrep_hidden=self.hidden_irrep,
                irrep_out=self.hidden_irrep,
                edge_attr_dim=self.radius_embed_dim,
                node_attr_dim=self.hs,
                sh_irrep=self.sh_irrep,
                resnet=True,
                use_norm_gate=True if i != 0 else False,
                use_equi_norm=use_equi_norm
            ))

            if i > self.start_layer:
                self.e3_gnn_node_layer.append(SelfNetLayer(
                        irrep_in_node=self.hidden_irrep_base,
                        irrep_bottle_hidden=self.hidden_irrep_base,
                        irrep_out=self.hidden_irrep_base,
                        sh_irrep=self.sh_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hs,
                        resnet=True,
                ))

                self.e3_gnn_node_pair_layer.append(PairNetLayer(
                        irrep_in_node=self.hidden_irrep_base,
                        irrep_bottle_hidden=self.hidden_irrep_base,
                        irrep_out=self.hidden_irrep_base,
                        sh_irrep=self.sh_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hs,
                        invariant_layers=self.num_fc_layer,
                        invariant_neurons=self.hs,
                        resnet=True,
                ))

        self.nonlinear_layer = get_nonlinear('ssp')
        self.expand_ii, self.expand_ij, self.fc_ii, self.fc_ij, self.fc_ii_bias, self.fc_ij_bias = [nn.ModuleDict() for _ in range(6)]
        for name in {"hamiltonian"}:
            input_expand_ii = o3.Irreps(f"{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e")

            self.expand_ii[name] = Expansion(
                input_expand_ii,
                o3.Irreps("3x0e + 2x1e + 1x2e"),
                o3.Irreps("3x0e + 2x1e + 1x2e")
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
                o3.Irreps(f'{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e'),
                o3.Irreps("3x0e + 2x1e + 1x2e"),
                o3.Irreps("3x0e + 2x1e + 1x2e")
            )

            self.fc_ij[name] = torch.nn.Sequential(
                nn.Linear(self.hs * 2, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_path_weight)
            )

            self.fc_ij_bias[name] = torch.nn.Sequential(
                nn.Linear(self.hs * 2, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_bias)
            )

        self.output_ii = Linear(self.hidden_irrep, self.hidden_bottle_irrep)
        self.output_ij = Linear(self.hidden_irrep, self.hidden_bottle_irrep)

    def get_number_of_parameters(self):
        num = 0
        for param in self.parameters():
            if param.requires_grad:
                num += param.numel()
        return num

    def set(self, device):
        self = self.to(device)
        self.orbital_mask = self.get_orbital_mask()
        for key in self.orbital_mask.keys():
            self.orbital_mask[key] = self.orbital_mask[key].to(self.device)

    @property
    def device(self):
        return next(self.parameters()).device

    def save(self, path):
        torch.save({"state_dict": self.state_dict(),}, path)

    def forward(self, data, return_full_hami=False):
        # node_attr, edge_index = data['atomic_numbers'].squeeze(), data['edge_index']
        data.ptr = torch.tensor([data["molecule_size"][:i].sum() for i in range(data["molecule_size"].shape[0]+1)]).to(self.device)

        node_attr, edge_index, rbf_new, edge_sh, _ = self.build_graph(data, self.max_radius_cutoff)
        node_attr = self.node_embedding(node_attr)
        data.node_attr, data.edge_index, data.edge_attr, data.edge_sh = \
            node_attr, edge_index, rbf_new, edge_sh

        _, full_edge_index, full_edge_attr, full_edge_sh, transpose_edge_index = self.build_graph(data, 10000)
        data.full_edge_index, data.full_edge_attr, data.full_edge_sh = \
            full_edge_index, full_edge_attr, full_edge_sh

        full_dst, full_src = data.full_edge_index

        tic = time.time()
        fii = None
        fij = None
        for layer_idx, layer in enumerate(self.e3_gnn_layer):
            node_attr = layer(data, node_attr)
            if layer_idx > self.start_layer:
                fii = self.e3_gnn_node_layer[layer_idx-self.start_layer-1](data, node_attr, fii)
                fij = self.e3_gnn_node_pair_layer[layer_idx-self.start_layer-1](data, node_attr, fij)

        fii = self.output_ii(fii)
        fij = self.output_ij(fij)
        hamiltonian_diagonal_matrix = self.expand_ii['hamiltonian'](
            fii, self.fc_ii['hamiltonian'](data.node_attr), self.fc_ii_bias['hamiltonian'](data.node_attr))
        node_pair_embedding = torch.cat([data.node_attr[full_dst], data.node_attr[full_src]], dim=-1)
        hamiltonian_non_diagonal_matrix = self.expand_ij['hamiltonian'](
            fij, self.fc_ij['hamiltonian'](node_pair_embedding),
            self.fc_ij_bias['hamiltonian'](node_pair_embedding))
        if return_full_hami:
            hamiltonian_matrix = self.build_final_matrix(
                data, hamiltonian_diagonal_matrix, hamiltonian_non_diagonal_matrix)
            hamiltonian_matrix = hamiltonian_matrix + hamiltonian_matrix.transpose(-1, -2)
            data['hamiltonian'] = hamiltonian_matrix
            data['duration'] = torch.tensor([time.time() - tic])
            
        else:
            ret_hamiltonian_diagonal_matrix = hamiltonian_diagonal_matrix +\
                                          hamiltonian_diagonal_matrix.transpose(-1, -2)

            # the transpose should considers the i, j
            ret_hamiltonian_non_diagonal_matrix = hamiltonian_non_diagonal_matrix + \
                      hamiltonian_non_diagonal_matrix[transpose_edge_index].transpose(-1, -2)

            data['pred_hamiltonian_diagonal_blocks'] = ret_hamiltonian_diagonal_matrix
            data['pred_hamiltonian_non_diagonal_blocks'] = ret_hamiltonian_non_diagonal_matrix
        return data

    def build_graph(self, data, max_radius):
        node_attr = data.atomic_numbers.squeeze()
        radius_edges = radius_graph(data.pos, max_radius, data.batch,max_num_neighbors=1000)

        dst, src = radius_edges
        edge_vec = data.pos[dst.long()] - data.pos[src.long()]
        rbf = self.radial_basis_functions(edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type(data.pos.type())

        edge_sh = o3.spherical_harmonics(
            self.sh_irrep, edge_vec[:, [1, 2, 0]],
            normalize=True, normalization='component').type(data.pos.type())

        start_edge_index = 0
        all_transpose_index = []
        for graph_idx in range(data.ptr.shape[0] - 1):
            num_nodes = data.ptr[graph_idx +1] - data.ptr[graph_idx]
            graph_edge_index = radius_edges[:, start_edge_index:start_edge_index+num_nodes*(num_nodes-1)]
            sub_graph_edge_index = graph_edge_index - data.ptr[graph_idx]
            bias = (sub_graph_edge_index[0] < sub_graph_edge_index[1]).type(torch.int)
            transpose_index = sub_graph_edge_index[0] * (num_nodes - 1) + sub_graph_edge_index[1] - bias
            transpose_index = transpose_index + start_edge_index
            all_transpose_index.append(transpose_index)
            start_edge_index = start_edge_index + num_nodes*(num_nodes-1)

        return node_attr, radius_edges, rbf, edge_sh, torch.cat(all_transpose_index, dim=-1)

    def build_final_matrix(self, data, diagonal_matrix, non_diagonal_matrix):
        # concate the blocks together and then select once.
        final_matrix = []
        dst, src = data.full_edge_index
        for graph_idx in range(data.ptr.shape[0] - 1):
            matrix_block_col = []
            for src_idx in range(data.ptr[graph_idx], data.ptr[graph_idx+1]):
                matrix_col = []
                for dst_idx in range(data.ptr[graph_idx], data.ptr[graph_idx+1]):
                    if src_idx == dst_idx:
                        matrix_col.append(diagonal_matrix[src_idx].index_select(
                            -2, self.orbital_mask[data.atomic_numbers[dst_idx].item()]).index_select(
                            -1, self.orbital_mask[data.atomic_numbers[src_idx].item()])
                        )
                    else:
                        mask1 = (src == src_idx)
                        mask2 = (dst == dst_idx)
                        index = torch.where(mask1 & mask2)[0].item()

                        matrix_col.append(
                            non_diagonal_matrix[index].index_select(
                                -2, self.orbital_mask[data.atomic_numbers[dst_idx].item()]).index_select(
                                -1, self.orbital_mask[data.atomic_numbers[src_idx].item()]))
                matrix_block_col.append(torch.cat(matrix_col, dim=-2))
            final_matrix.append(torch.cat(matrix_block_col, dim=-1))
        final_matrix = torch.stack(final_matrix, dim=0)
        return final_matrix

    def get_orbital_mask(self):
        idx_1s_2s = torch.tensor([0, 1])
        idx_2p = torch.tensor([3, 4, 5])
        orbital_mask_line1 = torch.cat([idx_1s_2s, idx_2p])
        orbital_mask_line2 = torch.arange(14)
        orbital_mask = {}
        for i in range(1, 11):
            orbital_mask[i] = orbital_mask_line1 if i <=2 else orbital_mask_line2
        return orbital_mask

    def split_matrix(self, data):
        diagonal_matrix, non_diagonal_matrix = \
            torch.zeros(data.atoms.shape[0], 14, 14).type(data.pos.type()).to(self.device), \
            torch.zeros(data.edge_index.shape[1], 14, 14).type(data.pos.type()).to(self.device)

        data.matrix =  data.matrix.reshape(
            len(data.ptr) - 1, data.matrix.shape[-1], data.matrix.shape[-1])

        num_atoms = 0
        num_edges = 0
        for graph_idx in range(data.ptr.shape[0] - 1):
            slices = [0]
            for atom_idx in data.atoms[range(data.ptr[graph_idx], data.ptr[graph_idx + 1])]:
                slices.append(slices[-1] + len(self.orbital_mask[atom_idx.item()]))

            for node_idx in range(data.ptr[graph_idx], data.ptr[graph_idx+1]):
                node_idx = node_idx - num_atoms
                orb_mask = self.orbital_mask[data.atoms[node_idx].item()]
                diagonal_matrix[node_idx][orb_mask][:, orb_mask] = \
                    data.matrix[graph_idx][slices[node_idx]: slices[node_idx+1], slices[node_idx]: slices[node_idx+1]]

            for edge_index_idx in range(num_edges, data.edge_index.shape[1]):
                dst, src = data.edge_index[:, edge_index_idx]
                if dst > data.ptr[graph_idx + 1] or src > data.ptr[graph_idx + 1]:
                    break
                num_edges = num_edges + 1
                orb_mask_dst = self.orbital_mask[data.atoms[dst].item()]
                orb_mask_src = self.orbital_mask[data.atoms[src].item()]
                graph_dst, graph_src = dst - num_atoms, src - num_atoms
                non_diagonal_matrix[edge_index_idx][orb_mask_dst][:, orb_mask_src] = \
                    data.matrix[graph_idx][slices[graph_dst]: slices[graph_dst+1], slices[graph_src]: slices[graph_src+1]]

            num_atoms = num_atoms + data.ptr[graph_idx + 1] - data.ptr[graph_idx]
        return diagonal_matrix, non_diagonal_matrix


class QHNet_backbone(nn.Module):
    def __init__(self,
                 order = 4,
                 embedding_dimension=128,
                 bottle_hidden_size=32,
                 num_gnn_layers=5,
                 max_radius=15,
                 num_nodes=20,
                 radius_embed_dim=32,
                 use_equi_norm = False,
                 start_layer=2,
                 use_sparse_tp=False,
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
        
        super(QHNet_backbone, self).__init__()
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

        # self.e3_gnn_layer = nn.ModuleList()

        # for i in range(self.num_gnn_layers):
        #     input_irrep = self.input_irrep if i == 0 else self.hidden_irrep
        #     self.e3_gnn_layer.append(ConvNetLayer(
        #         irrep_in_node=input_irrep,
        #         irrep_hidden=self.hidden_irrep,
        #         irrep_out=self.hidden_irrep,
        #         edge_attr_dim=self.radius_embed_dim,
        #         node_attr_dim=self.hs,
        #         sh_irrep=self.sh_irrep,
        #         resnet=True,
        #         use_norm_gate=True if i != 0 else False,
        #         use_equi_norm=use_equi_norm,
        #         # use_sparse_tp=use_sparse_tp,
        #     ))

        self.e3_gnn_layer = nn.ModuleList()
        self.e3_gnn_node_pair_layer = nn.ModuleList()
        self.e3_gnn_node_layer = nn.ModuleList()
        self.udpate_layer = nn.ModuleList()
        self.start_layer = 2
        for i in range(self.num_gnn_layers):
            input_irrep = self.input_irrep if i == 0 else self.hidden_irrep
            self.e3_gnn_layer.append(ConvNetLayer(
                irrep_in_node=input_irrep,
                irrep_hidden=self.hidden_irrep,
                irrep_out=self.hidden_irrep,
                edge_attr_dim=self.radius_embed_dim,
                node_attr_dim=self.hs,
                sh_irrep=self.sh_irrep,
                resnet=True,
                use_norm_gate=True if i != 0 else False,
                use_equi_norm=use_equi_norm,
                # use_sparse_tp=use_sparse_tp,
            ))

            if i > self.start_layer:
                self.e3_gnn_node_layer.append(SelfNetLayer(
                        irrep_in_node=self.hidden_irrep_base,
                        irrep_bottle_hidden=self.hidden_irrep_base,
                        irrep_out=self.hidden_irrep_base,
                        sh_irrep=self.sh_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hs,
                        resnet=True,
                        use_sparse_tp=use_sparse_tp,
                ))

                self.e3_gnn_node_pair_layer.append(PairNetLayer(
                        irrep_in_node=self.hidden_irrep_base,
                        irrep_bottle_hidden=self.hidden_irrep_base,
                        irrep_out=self.hidden_irrep_base,
                        sh_irrep=self.sh_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hs,
                        invariant_layers=self.num_fc_layer,
                        invariant_neurons=self.hs,
                        resnet=True,
                        use_sparse_tp=use_sparse_tp,
                ))

        self.nonlinear_layer = get_nonlinear('ssp')
        self.expand_ii, self.expand_ij, self.fc_ii, self.fc_ij, self.fc_ii_bias, self.fc_ij_bias = [nn.ModuleDict() for _ in range(6)]
        for name in {"hamiltonian"}:
            input_expand_ii = o3.Irreps(f"{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e")

            self.expand_ii[name] = Expansion(
                input_expand_ii,
                o3.Irreps("3x0e + 2x1e + 1x2e"),
                o3.Irreps("3x0e + 2x1e + 1x2e")
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
                o3.Irreps(f'{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e'),
                o3.Irreps("3x0e + 2x1e + 1x2e"),
                o3.Irreps("3x0e + 2x1e + 1x2e")
            )

            self.fc_ij[name] = torch.nn.Sequential(
                nn.Linear(self.hs * 2, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_path_weight)
            )

            self.fc_ij_bias[name] = torch.nn.Sequential(
                nn.Linear(self.hs * 2, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_bias)
            )

        self.output_ii = Linear(self.hidden_irrep, self.hidden_bottle_irrep)
        self.output_ij = Linear(self.hidden_irrep, self.hidden_bottle_irrep)
        
    def reset_parameters(self):
        warnings.warn("reset parameter is not init in qhnet backbone model")

    def build_graph(self, data, max_radius):
        node_attr = data.atomic_numbers.squeeze()
        radius_edges = radius_graph(data.pos, max_radius, data.batch, max_num_neighbors=1000)

        dst, src = radius_edges
        edge_vec = data.pos[dst.long()] - data.pos[src.long()]
        rbf = self.radial_basis_functions(edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type(data.pos.type())

        edge_sh = o3.spherical_harmonics(
            self.sh_irrep, edge_vec[:, [1, 2, 0]],
            normalize=True, normalization='component').type(data.pos.type())

        start_edge_index = 0
        all_transpose_index = []
        for graph_idx in range(data.ptr.shape[0] - 1):
            num_nodes = data.ptr[graph_idx +1] - data.ptr[graph_idx]
            graph_edge_index = radius_edges[:, start_edge_index:start_edge_index+num_nodes*(num_nodes-1)]
            sub_graph_edge_index = graph_edge_index - data.ptr[graph_idx]
            bias = (sub_graph_edge_index[0] < sub_graph_edge_index[1]).type(torch.int)
            transpose_index = sub_graph_edge_index[0] * (num_nodes - 1) + sub_graph_edge_index[1] - bias
            transpose_index = transpose_index + start_edge_index
            all_transpose_index.append(transpose_index)
            start_edge_index = start_edge_index + num_nodes*(num_nodes-1)

        return node_attr, radius_edges, rbf, edge_sh, torch.cat(all_transpose_index, dim=-1)

    def forward(self, batch_data):
        batch_data['ptr'] = torch.cat([torch.Tensor([0]).to(batch_data["molecule_size"].device).int(),
                              torch.cumsum(batch_data["molecule_size"],dim = 0)],dim = 0)
        
        _, full_edge_index, full_edge_attr, full_edge_sh, transpose_edge_index = self.build_graph(batch_data, 10000)
        batch_data.full_edge_index, batch_data.full_edge_attr, batch_data.full_edge_sh = \
        full_edge_index, full_edge_attr, full_edge_sh

        # batch_data.full_edge_index = full_edge_index
        # batch_data.full_edge_attr = full_edge_attr
        batch_data.transpose_edge_index = transpose_edge_index
        
        
        edge_index = radius_graph(batch_data.pos, self.max_radius, batch_data.batch)
        # dst, src = edge_index
        edge_vec = batch_data.pos[edge_index[0].long()] - batch_data.pos[edge_index[1].long()]
        rbf_new = self.radial_basis_functions(edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type(batch_data.pos.type())

        edge_sh = o3.spherical_harmonics(
            self.sh_irrep, edge_vec[:, [1, 2, 0]],
            normalize=True, normalization='component').type(batch_data.pos.type())

        node_attr = self.node_embedding(batch_data.atomic_numbers.squeeze())
        
        batch_data.node_attr, batch_data.edge_index, batch_data.edge_attr, batch_data.edge_sh = \
            node_attr, edge_index, rbf_new, edge_sh


        fii = None
        fij = None
        for layer_idx, layer in enumerate(self.e3_gnn_layer):
            if layer_idx == 3:
                node_vec2 = node_attr
            node_attr = layer(batch_data, node_attr)
        #     if layer_idx > self.start_layer:
        #         fii = self.e3_gnn_node_layer[layer_idx-self.start_layer-1](batch_data, node_attr, fii)
        #         fij = self.e3_gnn_node_pair_layer[layer_idx-self.start_layer-1](batch_data, node_attr, fij)
        # fii = self.output_ii(fii)
        # fij = self.output_ij(fij)
        
        batch_data["node_embedding"] = batch_data.node_attr
        batch_data["node_vec"] = node_attr
        batch_data["node_vec2"] = node_vec2
        # batch_data["fii"] = fii
        # batch_data["fij"] = fij
           

        return batch_data


