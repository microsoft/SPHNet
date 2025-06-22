
import torch
import warnings
from torch import nn
from torch_cluster import radius_graph
from e3nn import o3

from .QHNet import *
from .QHNet import Expansion, ExponentialBernsteinRadialBasisFunctions, SelfNetLayer, PairNetLayer, NormGate
from .QHNet_modify import PairNetLayer_symmetry
from .utils import construct_o3irrps_base, construct_o3irrps, get_full_graph, get_transpose_index
from madftnn.dataset.buildblock import get_conv_variable_lin,block2matrix





class HamiHead_uuw(nn.Module):
    def __init__(self,
                 irrep_in_node =  "128x0e + 128x1e + 128x2e + 128x3e + 128x4e",
                 irreps_edge_embedding = "32x0e + 32x1e + 32x2e + 32x3e + 32x4e",
                 order = 4,
                 pyscf_basis_name = "def2-svp",
                 radius_embed_dim=64,
                 max_radius_cutoff=20,
                 num_layer=2,
                 num_nodes = 20,
                 ):
        super().__init__()

        # #convert the model to the correct dtype
        # self.model.to(torch.float32)
        irrep_in_node = o3.Irreps(str(irrep_in_node).replace('o', 'e'))

        if str(irrep_in_node[0][1])!="0e":
            raise ValueError("The input node irrep should start with 0e")
        self.hs = irrep_in_node[0][0]
        # self.hidden_irrep = o3.Irreps(construct_o3irrps(self.hs,order = order))
        # self.init_sph_irrep = o3.Irreps(construct_o3irrps(1,order = order))
        # self.hidden_irrep_base = o3.Irreps(construct_o3irrps(self.hs,order = order))

        if pyscf_basis_name == 'def2-svp':
            exp_irrp = o3.Irreps("3x0e + 2x1e + 1x2e")
        elif pyscf_basis_name == 'def2-tzvp':
            exp_irrp = o3.Irreps("5x0e + 5x1e + 2x2e + 1x3e")
        else:
            raise ValueError('invalid base')
        self.conv,_,self.mask_lin,_ = get_conv_variable_lin(pyscf_basis_name)
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
        self.hidden_irrep = o3.Irreps(construct_o3irrps(self.hs, order=self.order))
        self.hidden_irrep_base = o3.Irreps(construct_o3irrps_base(self.hs, order=self.order))
        self.edge_irrep = o3.Irreps(irreps_edge_embedding)

        self.e3_gnn_node_pair_layer = nn.ModuleList()
        self.e3_gnn_node_layer = nn.ModuleList()
        for l in range(self.num_layer):
            self.e3_gnn_node_layer.append(SelfNetLayer(
                irrep_in_node=irrep_in_node,
                irrep_bottle_hidden=self.edge_irrep,
                irrep_out=self.edge_irrep,
                sh_irrep=self.sh_irrep,
                edge_attr_dim=self.radius_embed_dim,
                node_attr_dim=self.hs,
                tp_mode = "uuw",
                resnet=False,
            ))
            self.e3_gnn_node_pair_layer.append(PairNetLayer(
                irrep_in_node=irrep_in_node,
                irrep_bottle_hidden=self.edge_irrep,
                irrep_out=self.edge_irrep,
                sh_irrep=self.sh_irrep,
                edge_attr_dim=self.radius_embed_dim,
                node_attr_dim=self.hs,
                invariant_layers=1,
                invariant_neurons=self.hs,
                tp_mode = "uuw",
                resnet=False,
            ))
            # if increase_order:
            #     self.e3_gnn_node_pair_layer[l].norm_gate_pre = NormGate(irrep_in_node)

        self.output_ii = o3.Linear(self.edge_irrep, self.edge_irrep)
        self.output_ij = o3.Linear(self.edge_irrep, self.edge_irrep)

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
            non_diag = batch_data["pred_hamiltonian_non_diagonal_blocks"][atom_pair_start:atom_pair_start+n_atom*(n_atom-1)]
            rebuildfock = block2matrix(Z,diag,non_diag,self.mask_lin,self.conv.max_block_size,sym = False)
            
            _diag = batch_data["diag_hamiltonian"][atom_start:atom_start+n_atom]
            _non_diag = batch_data["non_diag_hamiltonian"][atom_pair_start:atom_pair_start+n_atom*(n_atom-1)]
            fock = block2matrix(Z,_diag,_non_diag,self.mask_lin,self.conv.max_block_size,sym = False)
            
            rebuildfocks.append(rebuildfock)
            gt_focks.append(fock)
            atom_start += n_atom
            atom_pair_start += n_atom*(n_atom-1)
        
        batch_data["pred_hamiltonian"] = rebuildfocks
        batch_data["hamiltonian"] = gt_focks
        return rebuildfocks

    def forward(self, data):
        if 'fii' not in data.keys():
            full_edge_index = get_full_graph(data)
            data["full_edge_index"] = full_edge_index
            
            full_edge_vec = data.pos[full_edge_index[0].long()] - data.pos[full_edge_index[1].long()]
            data.full_edge_attr = self.rbf(full_edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type(data.pos.type())
            data.full_edge_sh = o3.spherical_harmonics(
                    self.sh_irrep, full_edge_vec[:, [1, 2, 0]],
                    normalize=True, normalization='component').type(data.pos.type())

            node_features = data['node_vec']
            fii = None
            fij = None

            for layer_idx in range(self.num_layer):
                if layer_idx == 0:
                    fii = self.e3_gnn_node_layer[layer_idx](data, node_features, None)
                    fij = self.e3_gnn_node_pair_layer[layer_idx](data, node_features, None)
                else:
                    fii = fii + self.e3_gnn_node_layer[layer_idx](data, node_features, None)
                    fij = fij + self.e3_gnn_node_pair_layer[layer_idx](data, node_features, None)

            fii = self.output_ii(fii)
            fij = self.output_ij(fij)

            data['fii'], data['fij'] = fii, fij
        
        node_attr = data["node_attr"]
        fii = data["fii"]
        fij = data["fij"]
        data['ptr'] = torch.cat([torch.Tensor([0]).to(data["molecule_size"].device).int(),
                              torch.cumsum(data["molecule_size"],dim = 0)],dim = 0)
        full_dst, full_src = data.full_edge_index
        transpose_edge_index = get_transpose_index(data, data.full_edge_index)

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

 


class HamiHeadSymmetry_uuw(nn.Module):
    def __init__(self,
                 irrep_in_node =  "128x0e + 128x1e + 128x2e + 128x3e + 128x4e",
                 irreps_edge_embedding = "32x0e + 32x1e + 32x2e + 32x3e + 32x4e",
                 order = 4,
                 pyscf_basis_name = "def2-svp",
                 radius_embed_dim=64,
                 max_radius_cutoff=20,
                 bottle_hidden_size=64,
                 num_layer=2,
                 num_nodes=20,
                 ):
        """
        
        """
        super().__init__()
        
        irrep_in_node = o3.Irreps(str(irrep_in_node).replace('o', 'e'))
        if str(irrep_in_node[0][1])!="0e":
            raise ValueError("The input node irrep should start with 0e")
        self.hs = irrep_in_node[0][0]

        self.pyscf_basis_name = pyscf_basis_name
        if pyscf_basis_name == 'def2-svp':
            exp_irrp = o3.Irreps("3x0e + 2x1e + 1x2e")
        elif pyscf_basis_name == 'def2-tzvp':
            exp_irrp = o3.Irreps("5x0e + 5x1e + 2x2e + 1x3e")
        else:
            raise ValueError('invalid base')
        
        
        self.node_embedding = nn.Embedding(num_nodes, self.hs)
        self.conv,_,self.mask_lin,_ = get_conv_variable_lin(pyscf_basis_name)
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
        self.hidden_irrep = o3.Irreps(construct_o3irrps(self.hs, order=self.order))
        self.hidden_irrep_base = o3.Irreps(construct_o3irrps_base(self.hs, order=self.order))
        self.edge_irrep = o3.Irreps(irreps_edge_embedding)

        self.e3_gnn_node_pair_layer = nn.ModuleList()
        self.e3_gnn_node_layer = nn.ModuleList()
        for l in range(self.num_layer):
            self.e3_gnn_node_layer.append(SelfNetLayer(
                irrep_in_node=irrep_in_node,
                irrep_bottle_hidden=self.edge_irrep,
                irrep_out=self.edge_irrep,
                sh_irrep=self.sh_irrep,
                edge_attr_dim=self.radius_embed_dim,
                node_attr_dim=self.hs,
                tp_mode = "uuw",
                resnet=False,
            ))
            self.e3_gnn_node_pair_layer.append(PairNetLayer_symmetry(
                irrep_in_node=irrep_in_node,
                irrep_bottle_hidden=self.edge_irrep,
                irrep_out=self.edge_irrep,
                sh_irrep=self.sh_irrep,
                edge_attr_dim=self.radius_embed_dim,
                node_attr_dim=self.hs,
                invariant_layers=1,
                invariant_neurons=self.hs,
                tp_mode = "uuw",
                resnet=False,
            ))
            # if increase_order:
            #     self.e3_gnn_node_pair_layer[l].norm_gate_pre = NormGate(irrep_in_node)

        self.output_ii = o3.Linear(self.edge_irrep, self.edge_irrep)
        self.output_ij = o3.Linear(self.edge_irrep, self.edge_irrep)

    def reset_parameters(self):
        warnings.warn("reset parameter is not init in hami head model")
        

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
            
            _diag = batch_data["diag_hamiltonian"][atom_start:atom_start+n_atom]
            _non_diag = batch_data["non_diag_hamiltonian"][atom_pair_start:atom_pair_start+n_atom*(n_atom-1)//2]
            fock = block2matrix(Z,_diag,_non_diag,self.mask_lin,self.conv.max_block_size,sym = True)
            
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
            data["non_diag_hamiltonian"] = data["non_diag_hamiltonian"][full_edge_index[0]>full_edge_index[1]]
            data['non_diag_mask'] = data["non_diag_mask"][full_edge_index[0]>full_edge_index[1]]
            full_edge_index = full_edge_index[:,full_edge_index[0]>full_edge_index[1]]
            data["full_edge_index"] = full_edge_index
            
            full_edge_vec = data.pos[full_edge_index[0].long()] - data.pos[full_edge_index[1].long()]
            data.full_edge_attr = self.rbf(full_edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type(data.pos.type())
            data.full_edge_sh = o3.spherical_harmonics(
                    self.sh_irrep, full_edge_vec[:, [1, 2, 0]],
                    normalize=True, normalization='component').type(data.pos.type())

            node_features = data['node_vec']
            fii = None
            fij = None

            for layer_idx in range(self.num_layer):
                if layer_idx == 0:
                    fii = self.e3_gnn_node_layer[layer_idx](data, node_features, None)
                    fij = self.e3_gnn_node_pair_layer[layer_idx](data, node_features, None)
                else:
                    fii = fii + self.e3_gnn_node_layer[layer_idx](data, node_features, None)
                    fij = fij + self.e3_gnn_node_pair_layer[layer_idx](data, node_features, None)
            fii = self.output_ii(fii)
            fij = self.output_ij(fij)

            data['fii'], data['fij'] = fii, fij
        
        fii = data["fii"]
        fij = data["fij"]

        node_attr  = data["node_attr"] #self.node_embedding(data.atomic_numbers.squeeze())
        
        data['ptr'] = torch.cat([torch.Tensor([0]).to(data["molecule_size"].device).int(),
                              torch.cumsum(data["molecule_size"],dim = 0)],dim = 0)
        full_dst, full_src = data.full_edge_index

        hamiltonian_diagonal_matrix = self.expand_ii['hamiltonian'](
            fii, self.fc_ii['hamiltonian'](node_attr), self.fc_ii_bias['hamiltonian'](node_attr))
        node_pair_embedding = node_attr[full_dst] + node_attr[full_src]
        hamiltonian_non_diagonal_matrix = self.expand_ij['hamiltonian'](
            fij, self.fc_ij['hamiltonian'](node_pair_embedding),
            self.fc_ij_bias['hamiltonian'](node_pair_embedding))
        data['pred_hamiltonian_diagonal_blocks'] = hamiltonian_diagonal_matrix
        data['pred_hamiltonian_non_diagonal_blocks'] = hamiltonian_non_diagonal_matrix
        

        return data
    
    

class HamiHeadSymmetry_2output_uuw(nn.Module):
    def __init__(self,
                 irrep_in_node =  "128x0e + 128x1e + 128x2e + 128x3e + 128x4e",
                 irreps_edge_embedding = "32x0e + 32x1e + 32x2e + 32x3e + 32x4e",
                 order = 4,
                 pyscf_basis_name = "def2-svp",
                 radius_embed_dim=64,
                 max_radius_cutoff=20,
                 bottle_hidden_size=64,
                 num_layer=2,
                 num_nodes=20,
                 ):
        """
        
        """
        super().__init__()
        
        irrep_in_node = o3.Irreps(str(irrep_in_node).replace('o', 'e'))
        if str(irrep_in_node[0][1])!="0e":
            raise ValueError("The input node irrep should start with 0e")
        self.hs = irrep_in_node[0][0]

        self.pyscf_basis_name = pyscf_basis_name
        if pyscf_basis_name == 'def2-svp':
            exp_irrp = o3.Irreps("3x0e + 2x1e + 1x2e")
        elif pyscf_basis_name == 'def2-tzvp':
            exp_irrp = o3.Irreps("5x0e + 5x1e + 2x2e + 1x3e")
        else:
            raise ValueError('invalid base')
        
        
        self.node_embedding = nn.Embedding(num_nodes, self.hs)
        self.conv,_,self.mask_lin,_ = get_conv_variable_lin(pyscf_basis_name)
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
        self.hidden_irrep = o3.Irreps(construct_o3irrps(self.hs, order=self.order))
        self.hidden_irrep_base = o3.Irreps(construct_o3irrps_base(self.hs, order=self.order))
        self.edge_irrep = o3.Irreps(irreps_edge_embedding)

        self.e3_gnn_node_pair_layer = nn.ModuleList()
        self.e3_gnn_node_layer = nn.ModuleList()
        for l in range(self.num_layer):
            self.e3_gnn_node_layer.append(SelfNetLayer(
                irrep_in_node=irrep_in_node,
                irrep_bottle_hidden=self.edge_irrep,
                irrep_out=self.edge_irrep,
                sh_irrep=self.sh_irrep,
                edge_attr_dim=self.radius_embed_dim,
                node_attr_dim=self.hs,
                tp_mode = "uuw",
                resnet=False,
            ))
            self.e3_gnn_node_pair_layer.append(PairNetLayer_symmetry(
                irrep_in_node=irrep_in_node,
                irrep_bottle_hidden=self.edge_irrep,
                irrep_out=self.edge_irrep,
                sh_irrep=self.sh_irrep,
                edge_attr_dim=self.radius_embed_dim,
                node_attr_dim=self.hs,
                invariant_layers=1,
                invariant_neurons=self.hs,
                tp_mode = "uuw",
                resnet=False,
            ))
            # if increase_order:
            #     self.e3_gnn_node_pair_layer[l].norm_gate_pre = NormGate(irrep_in_node)

        self.output_ii = o3.Linear(self.edge_irrep, self.edge_irrep)
        self.output_ij = o3.Linear(self.edge_irrep, self.edge_irrep)

    def reset_parameters(self):
        warnings.warn("reset parameter is not init in hami head model")
        

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
            
            _diag = batch_data["diag_hamiltonian"][atom_start:atom_start+n_atom]
            _non_diag = batch_data["non_diag_hamiltonian"][atom_pair_start:atom_pair_start+n_atom*(n_atom-1)//2]
            fock = block2matrix(Z,_diag,_non_diag,self.mask_lin,self.conv.max_block_size,sym = True)
            
            
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
            data["non_diag_hamiltonian"] = data["non_diag_hamiltonian"][full_edge_index[0]>full_edge_index[1]]
            data['non_diag_mask'] = data["non_diag_mask"][full_edge_index[0]>full_edge_index[1]]
            full_edge_index = full_edge_index[:,full_edge_index[0]>full_edge_index[1]]
            data["full_edge_index"] = full_edge_index
            
            full_edge_vec = data.pos[full_edge_index[0].long()] - data.pos[full_edge_index[1].long()]
            data.full_edge_attr = self.rbf(full_edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type(data.pos.type())
            data.full_edge_sh = o3.spherical_harmonics(
                    self.sh_irrep, full_edge_vec[:, [1, 2, 0]],
                    normalize=True, normalization='component').type(data.pos.type())

            node_features = data['node_vec']
            bs,length2 = node_features.shape
            node_features1,node_features2 = node_features[:,:length2//2],node_features[:,length2//2:]
            fii = None
            fij = None

            for layer_idx in range(self.num_layer):
                if layer_idx == 0:
                    fii = self.e3_gnn_node_layer[layer_idx](data, node_features1, None)
                    fij = self.e3_gnn_node_pair_layer[layer_idx](data, node_features1, None)
                else:
                    fii = fii + self.e3_gnn_node_layer[layer_idx](data, node_features2, None)
                    fij = fij + self.e3_gnn_node_pair_layer[layer_idx](data, node_features2, None)
            fii = self.output_ii(fii)
            fij = self.output_ij(fij)

            data['fii'], data['fij'] = fii, fij
        
        fii = data["fii"]
        fij = data["fij"]

        node_attr  = data["node_attr"] #self.node_embedding(data.atomic_numbers.squeeze())
        
        data['ptr'] = torch.cat([torch.Tensor([0]).to(data["molecule_size"].device).int(),
                              torch.cumsum(data["molecule_size"],dim = 0)],dim = 0)
        full_dst, full_src = data.full_edge_index

        hamiltonian_diagonal_matrix = self.expand_ii['hamiltonian'](
            fii, self.fc_ii['hamiltonian'](node_attr), self.fc_ii_bias['hamiltonian'](node_attr))
        node_pair_embedding = node_attr[full_dst] + node_attr[full_src]
        hamiltonian_non_diagonal_matrix = self.expand_ij['hamiltonian'](
            fij, self.fc_ij['hamiltonian'](node_pair_embedding),
            self.fc_ij_bias['hamiltonian'](node_pair_embedding))
        data['pred_hamiltonian_diagonal_blocks'] = hamiltonian_diagonal_matrix
        data['pred_hamiltonian_non_diagonal_blocks'] = hamiltonian_non_diagonal_matrix
        

        return data