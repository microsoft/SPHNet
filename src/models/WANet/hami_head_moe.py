
import torch
import warnings
from torch import nn
from torch_cluster import radius_graph
from e3nn import o3
from .moe_utils import MoEPairLayer, MoEPairLayerSymmetry
from .hami_head_uuw import HamiHead_uuw, HamiHeadSymmetry_uuw



class HamiHead_uuwMoE(HamiHead_uuw):
    def __init__(self,
                 irrep_in_node =  "128x0e + 128x1e + 128x2e + 128x3e + 128x4e",
                 irreps_edge_embedding = "32x0e + 32x1e + 32x2e + 32x3e + 32x4e",
                 order = 4,
                 pyscf_basis_name = "def2-svp",
                 radius_embed_dim=64,
                 max_radius_cutoff=15,
                 num_layer=2,
                 num_nodes = 20,
                 ):
        super().__init__(irrep_in_node, irreps_edge_embedding, order, pyscf_basis_name, radius_embed_dim, max_radius_cutoff, num_layer, num_nodes)
        
        self.e3_gnn_node_pair_layer = nn.ModuleList()
        for l in range(self.num_layer):
            self.e3_gnn_node_pair_layer.append(MoEPairLayer(num_experts=4, gate_hidden_size=self.radius_embed_dim, noisy_gating=True, k=2,
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



class HamiHeadSymmetry_uuwMoE(HamiHeadSymmetry_uuw):
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
                 ):
        """
        
        """
        super().__init__(irrep_in_node, irreps_edge_embedding, order, pyscf_basis_name, radius_embed_dim, max_radius_cutoff, bottle_hidden_size, num_layer, num_nodes)

        self.e3_gnn_node_pair_layer = nn.ModuleList()
        for l in range(self.num_layer):
            self.e3_gnn_node_pair_layer.append(MoEPairLayerSymmetry(num_experts=4, gate_hidden_size=self.radius_embed_dim, noisy_gating=True, k=2,
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
