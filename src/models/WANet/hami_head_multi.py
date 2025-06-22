
import torch
import warnings
from torch import nn
from torch_cluster import radius_graph
from e3nn import o3


from .QHNet_utils import PairNetLayer_symmetry, Expansion, ExponentialBernsteinRadialBasisFunctions, SelfNetLayer, PairNetLayer
from ...dataset.buildblock import get_conv_variable_lin,block2matrix
from ..utils import construct_o3irrps_base, construct_o3irrps, get_full_graph, get_transpose_index

class HamiHead_Multi(nn.Module):
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
        self.hbs = bottle_hidden_size # 
        self.hidden_irrep = o3.Irreps(construct_o3irrps(self.hs, order=self.order))
        self.hidden_irrep_base = o3.Irreps(construct_o3irrps_base(self.hs, order=self.order))
        self.hidden_bottle_irrep = o3.Irreps(construct_o3irrps(self.hbs, order=self.order))

        self.e3_gnn_node_pair_layer = nn.ModuleList()
        self.e3_gnn_node_layer = nn.ModuleList()
        irrep_in_node = o3.Irreps(str(irrep_in_node).replace('o', 'e'))
        for l in range(self.num_layer):
            self.e3_gnn_node_layer.append(SelfNetLayer(
                irrep_in_node=irrep_in_node,
                irrep_bottle_hidden=self.hidden_irrep_base,
                irrep_out=self.hidden_irrep_base,
                sh_irrep=self.sh_irrep,
                edge_attr_dim=self.radius_embed_dim,
                node_attr_dim=self.hs,
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
                resnet=False,
            ))

        self.output_ii = o3.Linear(self.hidden_irrep_base, self.hidden_irrep_base)
        self.output_ij = o3.Linear(self.hidden_irrep_base, self.hidden_irrep_base)
        
        self.e3_gnn_node_pair_layer_1 = nn.ModuleList()
        self.e3_gnn_node_pair_layer_2 = nn.ModuleList()
        self.e3_gnn_node_pair_layer_3 = nn.ModuleList()
        self.e3_gnn_node_pair_layer_list = [self.e3_gnn_node_pair_layer, self.e3_gnn_node_pair_layer_1, self.e3_gnn_node_pair_layer_2, self.e3_gnn_node_pair_layer_3]
        for n in range(3):
            for l in range(self.num_layer):
                self.e3_gnn_node_pair_layer_list[n+1].append(PairNetLayer(
                    irrep_in_node=irrep_in_node,
                    irrep_bottle_hidden=self.hidden_irrep_base,
                    irrep_out=self.hidden_irrep_base,
                    sh_irrep=self.sh_irrep,
                    edge_attr_dim=self.radius_embed_dim,
                    node_attr_dim=self.hs,
                    invariant_layers=1,
                    invariant_neurons=self.hs,
                    resnet=False,
                ))
        for name in {"hamiltonian"}:
            self.fc_ij_1 = torch.nn.Sequential(
                nn.Linear(self.hs * 2, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_path_weight)
            )
            self.fc_ij_2 = torch.nn.Sequential(
                nn.Linear(self.hs * 2, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_path_weight)
            )
            self.fc_ij_3 = torch.nn.Sequential(
                nn.Linear(self.hs * 2, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_path_weight)
            )

            self.fc_ij_bias_1 = torch.nn.Sequential(
                nn.Linear(self.hs * 2, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_bias)
            )
            self.fc_ij_bias_2 = torch.nn.Sequential(
                nn.Linear(self.hs * 2, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_bias)
            )
            self.fc_ij_bias_3 = torch.nn.Sequential(
                nn.Linear(self.hs * 2, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_bias)
            )
            self.expand_ij_1 = Expansion(
                o3.Irreps(irreps_edge_embedding),
                exp_irrp,
                exp_irrp
            )
            self.expand_ij_2 = Expansion(
                o3.Irreps(irreps_edge_embedding),
                exp_irrp,
                exp_irrp
            )
            self.expand_ij_3 = Expansion(
                o3.Irreps(irreps_edge_embedding),
                exp_irrp,
                exp_irrp
            )
            
            self.output_ij_1 = o3.Linear(self.hidden_irrep_base, self.hidden_irrep_base)
            self.output_ij_2 = o3.Linear(self.hidden_irrep_base, self.hidden_irrep_base)
            self.output_ij_3 = o3.Linear(self.hidden_irrep_base, self.hidden_irrep_base)
            self.fc_ij_list = [self.fc_ij[name], self.fc_ij_1, self.fc_ij_2, self.fc_ij_3]
            self.fc_ij_bias_list = [self.fc_ij_bias[name], self.fc_ij_bias_1, self.fc_ij_bias_2, self.fc_ij_bias_3]
            self.output_ij_list = [self.output_ij, self.output_ij_1, self.output_ij_2, self.output_ij_3]
            self.expand_ij_list = [self.expand_ij[name], self.expand_ij_1, self.expand_ij_2, self.expand_ij_3]
        

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
            
            diag = batch_data["diag_hamiltonian"][atom_start:atom_start+n_atom]
            non_diag = batch_data["non_diag_hamiltonian"][atom_pair_start:atom_pair_start+n_atom*(n_atom-1)]
            fock = block2matrix(Z,diag,non_diag,self.mask_lin,self.conv.max_block_size,sym = False)
            
            rebuildfocks.append(rebuildfock)
            gt_focks.append(fock)
            atom_start += n_atom
            atom_pair_start += n_atom*(n_atom-1)
        
        batch_data["pred_hamiltonian"] = rebuildfocks
        batch_data["hamiltonian"] = gt_focks
        return rebuildfocks

    def get_distance_index(self, data, cutoff, full_edge_index, symmetry= False):
        short_edge_index = radius_graph(data.pos, cutoff, data.batch, max_num_neighbors=1000)
        if symmetry:
            short_edge_index = short_edge_index[:,short_edge_index[0]>short_edge_index[1]]
        comparison = (full_edge_index.unsqueeze(2) == short_edge_index.unsqueeze(1)).all(dim=0) 
        short_indices = torch.nonzero(comparison).t()[1]
        return short_indices
        

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

        full_edge_index = data.full_edge_index
        # full_edge_attr = copy.deepcopy(data.full_edge_attr)
        # fine the pairs that have short distance
        cut_off_list = [4,8,12,1000] # must be ascending order here
        indices_list = [self.get_distance_index(data,cut_off_list[0],full_edge_index)]
        for i in range(1,len(cut_off_list)):
            cur_indices = self.get_distance_index(data,cut_off_list[i],full_edge_index)
            last_indices = self.get_distance_index(data,cut_off_list[i-1],full_edge_index)
            cur_indices = cur_indices[~torch.isin(cur_indices, last_indices)]
            indices_list.append(cur_indices)
        
        data['multi_head_indices'] = indices_list
        for layer_idx in range(self.num_layer):
                fii = self.e3_gnn_node_layer[layer_idx](data, node_features, fii)
        
        # diag part
        fii = self.output_ii(fii)
        data['fii'] = fii
        node_attr = data["node_attr"]
        fii = data["fii"]
        data['ptr'] = torch.cat([torch.Tensor([0]).to(data["molecule_size"].device).int(),
                            torch.cumsum(data["molecule_size"],dim = 0)],dim = 0)
        hamiltonian_diagonal_matrix = self.expand_ii['hamiltonian'](
            fii, self.fc_ii['hamiltonian'](node_attr), self.fc_ii_bias['hamiltonian'](node_attr))  
        ret_hamiltonian_diagonal_matrix = 0.5*(hamiltonian_diagonal_matrix +
                                        hamiltonian_diagonal_matrix.transpose(-1, -2))
        data['pred_hamiltonian_diagonal_blocks'] = ret_hamiltonian_diagonal_matrix

        # non-diag part
        hamiltonian_non_diagonal_matrix = torch.zeros((full_edge_index.shape[-1], ret_hamiltonian_diagonal_matrix.shape[-1], 
                                                    ret_hamiltonian_diagonal_matrix.shape[-1]),device=data["molecule_size"].device)
        
        for n in range(0,4):
            data_part = {}
            data_part['full_edge_index'] = full_edge_index[:,indices_list[n]]
            data_part['full_edge_attr'] = data.full_edge_attr[indices_list[n]]
            # data.full_edge_index = full_edge_index[:,indices_list[n]]
            # data.full_edge_attr = full_edge_attr[indices_list[n]]
            fij_part = None
            # self.e3_gnn_node_pair_layer_list[n].to(data["molecule_size"].device)
            # self.output_ij_list[n].to(data["molecule_size"].device)
            # self.fc_ij_list[n].to(data["molecule_size"].device)
            # self.fc_ij_bias_list[n].to(data["molecule_size"].device)

            for layer_idx in range(self.num_layer):
                fij_part = self.e3_gnn_node_pair_layer_list[n][layer_idx](data_part, node_features, fij_part)

            fij_part = self.output_ij_list[n](fij_part)
            if n==0: 
                data['fij'] = torch.zeros((full_edge_index.shape[-1],fij_part.shape[-1]),device=data["molecule_size"].device)
            data['fij'][indices_list[n]] = fij_part
            node_attr = data["node_attr"]
            full_dst, full_src = full_edge_index[0][indices_list[n]], full_edge_index[1][indices_list[n]]
            
            node_pair_embedding = torch.cat([node_attr[full_dst], node_attr[full_src]], dim=-1)
            
            hamiltonian_non_diagonal_matrix[indices_list[n]] = self.expand_ij_list[n](
                fij_part, self.fc_ij_list[n](node_pair_embedding), self.fc_ij_bias_list[n](node_pair_embedding))
                
        data['ptr'] = torch.cat([torch.Tensor([0]).to(data["molecule_size"].device).int(),
                                torch.cumsum(data["molecule_size"],dim = 0)],dim = 0)
        # the transpose should considers the i, j
        transpose_edge_index = get_transpose_index(data, data.full_edge_index)
        ret_hamiltonian_non_diagonal_matrix = 0.5*(hamiltonian_non_diagonal_matrix + 
                        hamiltonian_non_diagonal_matrix[transpose_edge_index].transpose(-1, -2))

        data['pred_hamiltonian_non_diagonal_blocks'] = ret_hamiltonian_non_diagonal_matrix

        return data

 
class HamiHeadSymmetry_Multi(nn.Module):
    def __init__(self,
                 irrep_in_node =  "128x0e + 128x1e + 128x2e + 128x3e + 128x4e",
                 irreps_edge_embedding = "32x0e + 32x1e + 32x2e + 32x3e + 32x4e",
                 order = 4,
                 pyscf_basis_name = "def2-svp",
                 radius_embed_dim=64,
                 max_radius_cutoff=15,
                 bottle_hidden_size=64,
                 num_layer=2,
                 num_nodes=20,
                 **kwargs):
        """
        
        """
        super().__init__()

        self.hs = o3.Irreps(irrep_in_node)[0][0]
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

        self.hbs = bottle_hidden_size
        self.hidden_irrep = o3.Irreps(construct_o3irrps(self.hs, order=self.order))
        self.hidden_irrep_base = o3.Irreps(construct_o3irrps_base(self.hs, order=self.order))
        self.hidden_bottle_irrep = o3.Irreps(construct_o3irrps(self.hbs, order=self.order))

        self.e3_gnn_node_pair_layer = nn.ModuleList()
        self.e3_gnn_node_layer = nn.ModuleList()
        irrep_in_node = o3.Irreps(str(irrep_in_node).replace('o', 'e'))
        for l in range(self.num_layer):
            self.e3_gnn_node_layer.append(SelfNetLayer(
                irrep_in_node=irrep_in_node,
                irrep_bottle_hidden=self.hidden_irrep_base,
                irrep_out=self.hidden_irrep_base,
                sh_irrep=self.sh_irrep,
                edge_attr_dim=self.radius_embed_dim,
                node_attr_dim=self.hs,
                resnet=False,
            ))
            self.e3_gnn_node_pair_layer.append(PairNetLayer_symmetry(
                irrep_in_node=irrep_in_node,
                irrep_bottle_hidden=self.hidden_irrep_base,
                irrep_out=self.hidden_irrep_base,
                sh_irrep=self.sh_irrep,
                edge_attr_dim=self.radius_embed_dim,
                node_attr_dim=self.hs,
                invariant_layers=1,
                invariant_neurons=self.hs,
                resnet=False,
            ))

        self.output_ii = o3.Linear(self.hidden_irrep, self.hidden_bottle_irrep)
        self.output_ij = o3.Linear(self.hidden_irrep, self.hidden_bottle_irrep)

        self.e3_gnn_node_pair_layer_1 = nn.ModuleList()
        self.e3_gnn_node_pair_layer_2 = nn.ModuleList()
        self.e3_gnn_node_pair_layer_3 = nn.ModuleList()
        self.e3_gnn_node_pair_layer_list = [self.e3_gnn_node_pair_layer, self.e3_gnn_node_pair_layer_1, self.e3_gnn_node_pair_layer_2, self.e3_gnn_node_pair_layer_3]
        for n in range(3):
            for l in range(self.num_layer):
                self.e3_gnn_node_pair_layer_list[n+1].append(PairNetLayer_symmetry(
                    irrep_in_node=irrep_in_node,
                    irrep_bottle_hidden=self.hidden_irrep_base,
                    irrep_out=self.hidden_irrep_base,
                    sh_irrep=self.sh_irrep,
                    edge_attr_dim=self.radius_embed_dim,
                    node_attr_dim=self.hs,
                    invariant_layers=1,
                    invariant_neurons=self.hs,
                    resnet=False,
                ))
        for name in {"hamiltonian"}:
            self.fc_ij_1 = torch.nn.Sequential(
                nn.Linear(self.hs, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_path_weight)
            )
            self.fc_ij_2 = torch.nn.Sequential(
                nn.Linear(self.hs, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_path_weight)
            )
            self.fc_ij_3 = torch.nn.Sequential(
                nn.Linear(self.hs, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_path_weight)
            )

            self.fc_ij_bias_1 = torch.nn.Sequential(
                nn.Linear(self.hs, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_bias)
            )
            self.fc_ij_bias_2 = torch.nn.Sequential(
                nn.Linear(self.hs, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_bias)
            )
            self.fc_ij_bias_3 = torch.nn.Sequential(
                nn.Linear(self.hs, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_bias)
            )
            self.expand_ij_1 = Expansion(
                o3.Irreps(irreps_edge_embedding),
                exp_irrp,
                exp_irrp
            )
            self.expand_ij_2 = Expansion(
                o3.Irreps(irreps_edge_embedding),
                exp_irrp,
                exp_irrp
            )
            self.expand_ij_3 = Expansion(
                o3.Irreps(irreps_edge_embedding),
                exp_irrp,
                exp_irrp
            )
            
            self.output_ij_1 = o3.Linear(self.hidden_irrep, self.hidden_bottle_irrep)
            self.output_ij_2 = o3.Linear(self.hidden_irrep, self.hidden_bottle_irrep)
            self.output_ij_3 = o3.Linear(self.hidden_irrep, self.hidden_bottle_irrep)
            self.fc_ij_list = [self.fc_ij[name], self.fc_ij_1, self.fc_ij_2, self.fc_ij_3]
            self.fc_ij_bias_list = [self.fc_ij_bias[name], self.fc_ij_bias_1, self.fc_ij_bias_2, self.fc_ij_bias_3]
            self.output_ij_list = [self.output_ij, self.output_ij_1, self.output_ij_2, self.output_ij_3]
            self.expand_ij_list = [self.expand_ij[name], self.expand_ij_1, self.expand_ij_2, self.expand_ij_3]

    def get_distance_index(self, data, cutoff, full_edge_index, symmetry= True):
        short_edge_index = radius_graph(data.pos, cutoff, data.batch, max_num_neighbors=1000)
        if symmetry:
            short_edge_index = short_edge_index[:,short_edge_index[0]>short_edge_index[1]]
        comparison = (full_edge_index.unsqueeze(2) == short_edge_index.unsqueeze(1)).all(dim=0) 
        short_indices = torch.nonzero(comparison).t()[1]
        return short_indices
    
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

        full_edge_index = data.full_edge_index
        # full_edge_attr = copy.deepcopy(data.full_edge_attr)
        # fine the pairs that have short distance
        cut_off_list = [4,8,12,1000] # must be ascending order here
        indices_list = [self.get_distance_index(data,cut_off_list[0],full_edge_index)]
        for i in range(1,len(cut_off_list)):
            cur_indices = self.get_distance_index(data,cut_off_list[i],full_edge_index)
            last_indices = self.get_distance_index(data,cut_off_list[i-1],full_edge_index)
            cur_indices = cur_indices[~torch.isin(cur_indices, last_indices)]
            indices_list.append(cur_indices)
        
        data['multi_head_indices'] = indices_list

        # diag part
        for layer_idx in range(self.num_layer):
            if layer_idx == 0:
                fii = self.e3_gnn_node_layer[layer_idx](data, node_features, None)
            else:
                fii = fii + self.e3_gnn_node_layer[layer_idx](data, node_features, None)
        fii = self.output_ii(fii)
        data['fii'] = fii
        node_attr = data["node_attr"]
        fii = data["fii"]
        hamiltonian_diagonal_matrix = self.expand_ii['hamiltonian'](
            fii, self.fc_ii['hamiltonian'](node_attr), self.fc_ii_bias['hamiltonian'](node_attr))
        data['pred_hamiltonian_diagonal_blocks'] = hamiltonian_diagonal_matrix


        # non-diag part
        hamiltonian_non_diagonal_matrix = torch.zeros((full_edge_index.shape[-1], hamiltonian_diagonal_matrix.shape[-1], 
                                                    hamiltonian_diagonal_matrix.shape[-1]),device=data["molecule_size"].device)
        for n in range(0,4):
            data_part = {}
            data_part['full_edge_index'] = full_edge_index[:,indices_list[n]]
            data_part['full_edge_attr'] = data.full_edge_attr[indices_list[n]]
            fij_part = None
    
            for layer_idx in range(self.num_layer):
                fij_part = self.e3_gnn_node_pair_layer_list[n][layer_idx](data_part, node_features, fij_part)

            fij_part = self.output_ij_list[n](fij_part)
            if n==0: 
                data['fij'] = torch.zeros((full_edge_index.shape[-1],fij_part.shape[-1]),device=data["molecule_size"].device)
            data['fij'][indices_list[n]] = fij_part
            node_attr = data["node_attr"]
            full_dst, full_src = full_edge_index[0][indices_list[n]], full_edge_index[1][indices_list[n]]
            
            node_pair_embedding = node_attr[full_dst] + node_attr[full_src]
            
            hamiltonian_non_diagonal_matrix[indices_list[n]] = self.expand_ij_list[n](
                fij_part, self.fc_ij_list[n](node_pair_embedding), self.fc_ij_bias_list[n](node_pair_embedding))
        
        data['ptr'] = torch.cat([torch.Tensor([0]).to(data["molecule_size"].device).int(),
                            torch.cumsum(data["molecule_size"],dim = 0)],dim = 0)
        
    
    
        node_pair_embedding = node_attr[full_dst] + node_attr[full_src]
        data['pred_hamiltonian_non_diagonal_blocks'] = hamiltonian_non_diagonal_matrix
        
        return data


 
