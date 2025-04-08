import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_scatter import scatter

from itertools import combinations

class NVTransform(BaseTransform):
    
    def __init__(self, aggr: str, cutoff, An2=False):
        
        self.aggr = aggr
        self.cutoff = cutoff
        self.An2 = An2
        assert self.aggr in ["sum", "mean", "weight"]
        # * types of aggregation for normal vectors 
    
    @staticmethod
    def Rotation2Z_matrix(xyz: torch.Tensor) -> torch.Tensor:

        r"""
            Generate Rotation Matrix to rotate nodes to axis-Z
        """

        batch_size = xyz.shape[0]
        R1 = torch.zeros((batch_size, 3, 3), dtype=torch.float)
        R2 = torch.zeros((batch_size, 3, 3), dtype=torch.float)

        r_norm = torch.linalg.vector_norm(xyz, dim=1)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        cos_theta = z / torch.sqrt(z**2 + y**2)
        sin_theta = y / torch.sqrt(z**2 + y**2)


        R1[:, 0, 0] = 1.0
        R1[:, 1, 1] = cos_theta
        R1[:, 1, 2] = sin_theta
        R1[:, 2, 1] = - sin_theta
        R1[:, 2, 2] = cos_theta

        cos_fai = torch.sqrt(y**2 + z**2) / r_norm
        sin_fai = x / r_norm

        R2[:, 0, 0] = cos_fai
        R2[:, 0, 2] = sin_fai
        R2[:, 1, 1] = 1.0
        R2[:, 2, 0] = - sin_fai
        R2[:, 2, 2] = cos_fai

        return torch.bmm(R1, R2)
    
    @staticmethod
    def fast_perm(xyz:torch.Tensor, nEdges: int, sorted_nbrs: torch.Tensor, rule=None):
                
        edge_index = sorted_nbrs[:, [0, 1]].long()
        node_ids, num_neighbors = edge_index[:, 1].unique(return_counts=True)
        
        if rule is None:
            
            perm_add_idx = torch.cat(
                (torch.tensor([0], dtype=torch.long, device=num_neighbors.device), num_neighbors), 
                dim=-1).cumsum(dim=-1)[:-1]
            perm_add = torch.zeros((nEdges), dtype=torch.long, device=perm_add_idx.device)
            perm_add[perm_add_idx] += num_neighbors
            perm_edge_id = (torch.arange(nEdges, device=perm_add.device) - 1) + perm_add
            
            src_vector = xyz[edge_index[:, 0]] - xyz[edge_index[:, 1]]
            trg_vector = xyz[edge_index[:, 0][perm_edge_id]] - xyz[edge_index[:, 1]]
        
            return src_vector, trg_vector, edge_index[:, 1]
        
        else:
            
            src_vector, trg_vector, scatter_idx = torch.tensor([]), torch.tensor([]), torch.tensor([])
            s, e = 0, 0
            for idx, n_id in enumerate(node_ids):
                
                cur_num_nbrs = num_neighbors[idx]
                e = s + cur_num_nbrs
                cur_edge_index = edge_index[s:e]
                assert cur_num_nbrs == cur_edge_index.shape[0]
                
                anchor = rule(cur_edge_index, xyz)
                perm_id = torch.arange(cur_num_nbrs)
                perm_id = torch.cat((perm_id[anchor:], perm_id[:anchor]), dim=-1)
                An2 = torch.tensor(list(combinations(perm_id, r=2)), dtype=torch.long)
                rpt_trg = torch.repeat_interleave(n_id, An2.shape[0])
                scatter_idx = torch.cat((scatter_idx, rpt_trg), dim=-1)
                
                src_vector = torch.cat((src_vector, xyz[cur_edge_index[An2[:, 0], 0]] - xyz[rpt_trg]), dim=0)
                trg_vector = torch.cat((trg_vector, xyz[cur_edge_index[An2[:, 1], 0]] - xyz[rpt_trg]), dim=0)
                s = e
            
            return src_vector, trg_vector, scatter_idx.long()
                
    @staticmethod
    def shortest_dist(edge_index, xyz):
        
        rel_dist = torch.norm(xyz[edge_index[:, 0]] - xyz[edge_index[:, 1]], dim=-1)
        
        return torch.argmin(rel_dist, dim=-1)
        
    
    @staticmethod
    def get_nv_j(xyz: torch.Tensor, edge_index: torch.Tensor, An2: bool, aggr: str) -> torch.Tensor:
        
        # TODO precision of such method is not high enough (see equivariant_test.ipynb)
        
        nbrs = edge_index.T
        nEdges = edge_index.size(1)
        
        # * get geometric mass
        mass = torch.mean(xyz, dim=0)

        # * mass -> i (deal with src nodes)
        # * so they can correctly pass vectors to trg nodes
        r_im = xyz[nbrs[:, 1]] - mass

        # * get rotation matrix S for all im
        S = NVTransform.Rotation2Z_matrix(r_im)
        S = S.to(xyz.device)

        # get rotation matrix S_node for all i
        # S_node = Rotation2Z_matrix(xyz - mass)

        # get inverse rotation matrix
        # S_inv = torch.inverse(S_node)

        # S_m = torch.bmm(r_im.unsqueeze(1), S)
        # print(S_m)

        # * mass -> j
        S_nbr = xyz[nbrs[:, 0]] - mass
        
        # * rotate neighbors
        S_nbr = torch.bmm(S_nbr.unsqueeze(1), S).squeeze(1)
        # * S_nbr.size (num_edge, 3)
        theta = torch.atan2(S_nbr[:, 1], S_nbr[:, 0]) # + torch.where(S_nbr[:, 0] < 0, 1, 0) * torch.pi
        
        edge_id = torch.arange(nEdges).to(nbrs.device)
        cat_theta = torch.cat(
            (
                nbrs[:, 0].unsqueeze(-1), 
                nbrs[:, 1].unsqueeze(-1),
                edge_id.unsqueeze(-1), 
                theta.unsqueeze(-1)
            ), 
            dim=1
        )
        sorted_nbrs = torch.stack(sorted(cat_theta, key=lambda x: (x[1], x[3])), dim=0)
        
        rule = NVTransform.shortest_dist if An2 else None
        src_vector, trg_vector, scatter_idx = NVTransform.fast_perm(xyz=xyz, nEdges=nEdges, sorted_nbrs=sorted_nbrs, rule=rule)

        normal_vectors = torch.cross(src_vector, trg_vector)

        if aggr == 'mean' or aggr == 'sum':
            
            vertex_vectors = scatter(
                src=normal_vectors,
                index=scatter_idx,
                dim=0,
                reduce=aggr # TODO ablation study
            )
            
        elif aggr == 'weight':
            
            src_vector_d = src_vector / torch.norm(src_vector, dim=-1).unsqueeze(-1)
            trg_vector_d = trg_vector / torch.norm(trg_vector, dim=-1).unsqueeze(-1)
            cos_weight = (src_vector_d * trg_vector_d).sum(dim=-1)
            
            normal_vectors = normal_vectors * cos_weight.unsqueeze(-1)
            
            vertex_vectors = scatter(
                src=normal_vectors,
                index=scatter_idx,
                dim=0,
                reduce="sum" # TODO ablation study
            )
            
        # plane_normal_vectors = normal_vectors[sorted_nbrs[:, 2].long()]
        # return plane_normal_vectors
        
        return vertex_vectors
        
    def __call__(self, data: Data) -> Data:
        
        xyz = data.pos
        
        edge_index = radius_graph(xyz, r=self.cutoff)
        
        data.edge_index = edge_index
        
        data.nv = NVTransform.get_nv_j(xyz, edge_index, self.An2, self.aggr)
        
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(aggr={self.aggr}, cutoff={self.cutoff})')