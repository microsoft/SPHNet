import torch
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch_geometric.nn.models.schnet import GaussianSmearing, ShiftedSoftplus
import torch.nn as nn
from .torchmdnet.models.utils import (
    NeighborEmbedding,
    CosineCutoff,
    Distance,
    rbf_class_mapping,
    act_class_mapping,
    vec_layernorm,
    max_min_norm,
    norm
)
from .utils import *
from torch_geometric.utils import softmax
import copy
import math


class LongShortIneractModel_distance(MessagePassing):
    r'''
    This is the long term model to capture the relationship b/w center node and long term groups
    First, perform a pointTransformer within each group to obtain the representation of each group.
    Second, MP is performed on a bipartite graph of size #nodes * #groups.
    Below is an disection of the architecture of the model

    in_channels_node ---- (PointTransformer) ----- > in channels group

    in_channels_group --- linear --- num_filters -|    |-- > out_channels
                                                  | MP |
    in_channels_node  --- linear --- num_filters -| -> |-- > out_channels
                                                  |    
    num_gaussians     --- linear --- num_filters -|    

    @param in_channels_node: The size of node features
    @param in_chnnels_right: The size of group features
    @param out_channels: The size of output node features and group features (unified output size)
    @param num_filters: number of filters 
    @param num_gaussians: number of gaussians used to expand edge weight
    @param cutoff: long term cutoff distance
    ''' 
    def __init__(self, hidden_channels,num_gaussians, cutoff ,max_group_num = 3,act = "silu",**kwargs):
        super().__init__()

        self.act1 = nn.SiLU()
        self.cutoff = cutoff
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.max_group_num  = max_group_num
        
        self.act = nn.SiLU()
        self.mlp_1 = nn.Sequential(
            nn.Linear(num_gaussians, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, hidden_channels)
        # self.lin4 = nn.Linear(out_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        pass
        # torch.nn.init.xavier_uniform_(self.lin.weight)
        # self.lin.bias.data.fill_(0)
        # Get the center position
        
        
        # if self.group_center == 'geometric':
        #     group_pos = scatter(node_pos, labels, reduce='mean', dim=0)
        # elif self.group_center == 'center_of_mass':
        #     group_pos = scatter(node_pos * data.atomic_numbers, labels, reduce='sum', dim=0) / scatter(data.atomic_numbers, labels, reduce='sum', dim=0)
        # else:
        #     raise NotImplementedError

    def forward(self, edge_index, node_embedding, node_pos,group_embedding,group_pos,**kwargs):
        '''
        grouping_graph_edge_idx = data.grouping_graph # Grouping graph (intra group complete graph; inter group disconnected)
        edge_idx = data.interaction_graph # Bipartite graph
        
        '''

        nodes = edge_index[0]
        groups = edge_index[1]

        # Distance Expansion
        edge_weight = norm((node_pos[nodes] - group_pos[groups]), dim = -1)
        edge_attr = self.distance_expansion(edge_weight)

        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        
        C = 0.5 * (torch.cos(edge_weight * torch.pi / self.cutoff) + 1.0) #cutoff function
        W = self.mlp_1(edge_attr) * C.view(-1, 1)

        node_embedding = self.lin1(node_embedding)
        group_embedding = self.lin2(group_embedding)
        # Message flow from group to node
        node_embedding = self.propagate(edge_index.flip(0), size = (num_groups, num_nodes), x=(group_embedding, node_embedding), W=W)/self.max_group_num
        node_embedding = self.lin3(node_embedding)
        
        return node_embedding,None
        
    def message(self, x_j, W):
        return x_j * W
    
    

# class CFVectorConvBipartite(MessagePassing):
class LongShortIneractModel_dis_direct(MessagePassing):
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm = False,act = "silu",num_heads=8,**kwargs):
        super().__init__(aggr='add', node_dim = 0) # currently only node embedding is computed and updated, only group message flow to node
        self.act = act_class_mapping[act]()
        self.norm = norm
        self.layernorm_node = nn.LayerNorm(hidden_channels)
        self.layernorm_group = nn.LayerNorm(hidden_channels)
        self.layernorm_node_vec = nn.LayerNorm(hidden_channels)
        self.layernorm_group_vec = nn.LayerNorm(hidden_channels)
        self.model_2 = nn.ModuleDict({
            # 'mlp_edge_attr': nn.Sequential(
            #     nn.Linear(num_gaussians, hidden_channels),
            #     self.act,
            #     nn.Linear(hidden_channels, hidden_channels),),
            'q': nn.Linear(hidden_channels, hidden_channels),
            'k': nn.Linear(hidden_channels, hidden_channels),
            'val': nn.Linear(hidden_channels, hidden_channels),
            'mlp_scalar_pos': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
            'mlp_scalar_vec': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
            'linears': nn.ModuleList([nn.Linear(hidden_channels, hidden_channels, bias=False) for _ in range(6)])    
        })
        self.model_1 = None
        # self.model_1 = copy.deepcopy(self.model_2)
        self.num_heads = num_heads
        self.attn_channels = hidden_channels // num_heads
        self.reset_parameters()

    def reset_parameters(self):
        '''
        Create Xavier Unifrom for all linear modules
        '''
        self.layernorm_node.reset_parameters()
        self.layernorm_group.reset_parameters()
        # self.attn_layers.reset_parameters()
        for model in [self.model_1, self.model_2]:
            if model is None:continue
            for _, value in model.items():
                if isinstance(value, nn.ModuleList):
                    for m in value.modules():
                        if isinstance(m, nn.Linear):
                            torch.nn.init.xavier_uniform_(m.weight)
                elif isinstance(value, nn.Linear):
                    torch.nn.init.xavier_uniform_(value.weight)
                    value.bias.data.fill_(0)
                else:
                    pass
            
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec):
        """_summary_

        Args:
            edge_index (_type_): 2*edge_num, [0] is node id,[1] is group id
            x_node (_type_): _description_
            x_group (_type_): _description_
            v_node (_type_): _description_
            v_group (_type_): _description_
            r_node (_type_): _description_
            r_group (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            group_embedding = self.layernorm_group(group_embedding)
        # v_node = self.layernorm_node_vec(v_node)
        # v_group = self.layernorm_group_vec(v_group)
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        # Message flow from node to group
        # relative_vec_embed = node_vec[edge_index[0]] - group_vec[edge_index[1]]

        # Message flow from group to node
        attn_2, val_2 = self.calculate_attention(node_embedding, group_embedding, edge_index[0], edge_index[1], edge_attr, self.model_2, "silu")
        m_s_node, m_v_node = self.propagate(edge_index.flip(0),
                                size = (num_groups, num_nodes),
                                x =(group_embedding, node_embedding), 
                                v = group_vec[edge_index[1]],
                                u_ij = -edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = attn_2, 
                                val = val_2[edge_index[1]],
                                mode = 'group_to_node')    
        
        # vec to scalar, ||dot((W1*v), (W2*w))||
        # if self.select == 3:
        #     v_node_1 = self.model_2['linears'][2](node_vec)
        #     v_node_2 = self.model_2['linears'][3](node_vec)
        #     dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node)
        #     dv_node = m_v_node
        # elif self.select == 2:
        #     v_node_1 = self.model_2['linears'][2](node_vec)
        #     v_node_2 = self.model_2['linears'][3](node_vec)
        #     dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node)
        #     dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        # elif self.select == 1:
        #     v_node_1 = self.model_2['linears'][2](node_vec)
        #     v_node_2 = self.model_2['linears'][3](node_vec)
        #     dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        #     dv_node = m_v_node
        # elif self.select == 0:
        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        return dx_node, dv_node
    
    def calculate_attention(self,x_1, x_2, x1_index, x2_index, expanded_edge_weight, model, attn_type):
        r'''
        Calculate attention value for each edge.
        x_1: embedding for query. target node embedding.
        x_2: embedding for key value. source node embedding.
        edge_index: graph to calculate attention
        expanded_edge_weight: the expanded edge weight
        model: the model for calculating attention, be a dictionary with keys: q, k, val, mlp_edge_attr, mlp_scalar_pos, mlp_scalar_vec
        '''
        __supported_attn__ = ['softmax', 'silu']
        q = model['q'](x_1).reshape(-1, self.num_heads, self.attn_channels) # num_groups x num_heads x attn_channels
        k = model['k'](x_2).reshape(-1, self.num_heads, self.attn_channels) # num_nodes x num_heads x attn_channels
        val = model['val'](x_2).reshape(-1, self.num_heads, self.attn_channels) 

        q_i = q[x1_index]
        k_j = k[x2_index]

        expanded_edge_weight = expanded_edge_weight.reshape(-1, self.num_heads, self.attn_channels)
        attn = q_i * k_j * expanded_edge_weight
        attn = attn.sum(dim = -1) / math.sqrt(self.attn_channels)
        # attn = attn.sum(dim = -1) / self.attn_channels
        # attn = attn.sum(dim = -1)

            
        if attn_type == 'softmax':
            attn = softmax(attn, x1_index, dim = 0)
        elif attn_type == 'silu':
            attn = act_class_mapping['silu']()(attn)
        else:
            raise NotImplementedError(f'Attention type {attn_type} is not supported, supported types are {__supported_attn__}')
        return attn, val      

    def message(self, x_i, x_j, v, u_ij, d_ij, attn_score, val, mode):
        '''
        Calculate the message from node j to node i
        Return: Scalar message from node j to node i, Vector message from node j to node i
        @param x_j: Node feature of node j
        @param x_i: Node feature of node i
        @param smeared_distance: Smearing distance between node i and node j
        @param v: Embedding of the group th
        @param mode: 'node_to_group' or 'group_to_node'
        '''

        
        # distance gated attention, larger the distance, smaller the attention should be.
        # There is the intuition, but we gave the model flexibility to learn on its own.
        if mode == 'node_to_group':
            model = self.model_1
        else:
            model = self.model_2
        

        # a_ij = self.act(q_i * k_j * edge_attr.reshape(-1, self.num_heads, self.attn_channels) / (self.attn_channels ** 0.5))
        # a_ij = a_ij.sum(dim = -1, keepdim = True)
        # num_nodes, num_heads, attn_channels = q_i.shape
        
        # a_ij = q_i * k_j *  / (self.attn_channels ** 0.5)
        # a_ij = a_ij.(dim = -1, keepdim = True)
        
        m_s_ij = val * attn_score.unsqueeze(-1) # scalar message
        m_s_ij = m_s_ij.reshape(-1, self.num_heads * self.attn_channels)
        m_v_ij = model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1) \
        + model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * (v) # vector message
        return  m_s_ij, m_v_ij
        
    def aggregate(self, features, index, ptr, dim_size):
        # x, vec, w = features
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        # w = scatter(w, index, dim=self.node_dim, dim_size=dim_size)
        # return x, vec, w
        return x, vec    


class LongShortIneractModel_dis_direct_vector(LongShortIneractModel_dis_direct):
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm = False,act = "silu",num_heads=8,**kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm,act,num_heads) # currently only node embedding is computed and updated, only group message flow to node
        
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec):
        """_summary_

        Args:
            edge_index (_type_): 2*edge_num, [0] is node id,[1] is group id
            x_node (_type_): _description_
            x_group (_type_): _description_
            v_node (_type_): _description_
            v_group (_type_): _description_
            r_node (_type_): _description_
            r_group (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
            group_embedding = self.layernorm_group(group_embedding)
        # v_node = self.layernorm_node_vec(v_node)
        # v_group = self.layernorm_group_vec(v_group)
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        # Message flow from node to group
        # relative_vec_embed = node_vec[edge_index[0]] - group_vec[edge_index[1]]

        # Message flow from group to node
        attn_2, val_2 = self.calculate_attention(node_embedding, group_embedding, edge_index[0], edge_index[1], edge_attr, self.model_2, "silu")
        m_s_node, m_v_node = self.propagate(edge_index.flip(0),
                                size = (num_groups, num_nodes),
                                x=(group_embedding, node_embedding), 
                                v = group_vec[edge_index[1]],
                                u_ij = -edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = attn_2, 
                                val = val_2[edge_index[1]],
                                mode = 'group_to_node')    
        

        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        return dx_node, dv_node




class LongShortIneractModel_dis_direct_vector2(LongShortIneractModel_dis_direct):
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm = False,act = "silu",num_heads=8,**kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm,act,num_heads) # currently only node embedding is computed and updated, only group message flow to node
        
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec):
        """_summary_

        Args:
            edge_index (_type_): 2*edge_num, [0] is node id,[1] is group id
            x_node (_type_): _description_
            x_group (_type_): _description_
            v_node (_type_): _description_
            v_group (_type_): _description_
            r_node (_type_): _description_
            r_group (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
            group_embedding = self.layernorm_group(group_embedding)
            group_vec = vec_layernorm(group_vec, max_min_norm)

        # v_node = self.layernorm_node_vec(v_node)
        # v_group = self.layernorm_group_vec(v_group)
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        # Message flow from node to group
        # relative_vec_embed = node_vec[edge_index[0]] - group_vec[edge_index[1]]

        # Message flow from group to node
        attn_2, val_2 = self.calculate_attention(node_embedding, group_embedding, edge_index[0], edge_index[1], edge_attr, self.model_2, "silu")
        m_s_node, m_v_node = self.propagate(edge_index.flip(0),
                                size = (num_groups, num_nodes),
                                x=(group_embedding, node_embedding), 
                                v = group_vec[edge_index[1]],
                                u_ij = -edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = attn_2, 
                                val = val_2[edge_index[1]],
                                mode = 'group_to_node')    
        

        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        return dx_node, dv_node
    


class LongShortIneractModel_dis_direct_vector2_drop(LongShortIneractModel_dis_direct):
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm = False,act = "silu",num_heads=8, p =0.1,**kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm,act,num_heads) # currently only node embedding is computed and updated, only group message flow to node
        self.dropout_s = nn.Dropout(p)
        self.dropout_v = nn.Dropout(p)
        self.p = p
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec):
        """_summary_

        Args:
            edge_index (_type_): 2*edge_num, [0] is node id,[1] is group id
            x_node (_type_): _description_
            x_group (_type_): _description_
            v_node (_type_): _description_
            v_group (_type_): _description_
            r_node (_type_): _description_
            r_group (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
            group_embedding = self.layernorm_group(group_embedding)
            group_vec = vec_layernorm(group_vec, max_min_norm)
        if self.p>0:
            group_embedding = self.dropout_s(group_embedding)
            group_vec = self.dropout_v(group_vec)
        
        # v_node = self.layernorm_node_vec(v_node)
        # v_group = self.layernorm_group_vec(v_group)
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        # Message flow from node to group
        # relative_vec_embed = node_vec[edge_index[0]] - group_vec[edge_index[1]]

        # Message flow from group to node
        attn_2, val_2 = self.calculate_attention(node_embedding, group_embedding, edge_index[0], edge_index[1], edge_attr, self.model_2, "silu")
        m_s_node, m_v_node = self.propagate(edge_index.flip(0),
                                size = (num_groups, num_nodes),
                                x=(group_embedding, node_embedding), 
                                v = group_vec[edge_index[1]],
                                u_ij = -edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = attn_2, 
                                val = val_2[edge_index[1]],
                                mode = 'group_to_node')    
        

        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        return dx_node, dv_node
    
    
        
class LongShortIneractModel_dis_direct_vector3(LongShortIneractModel_dis_direct):
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm = False,act = "silu",num_heads=8,**kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm,act,num_heads) # currently only node embedding is computed and updated, only group message flow to node
        
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec):
        """_summary_

        Args:
            edge_index (_type_): 2*edge_num, [0] is node id,[1] is group id
            x_node (_type_): _description_
            x_group (_type_): _description_
            v_node (_type_): _description_
            v_group (_type_): _description_
            r_node (_type_): _description_
            r_group (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
            # group_embedding = self.layernorm_group(group_embedding)
            # group_vec = vec_layernorm(group_vec, max_min_norm)

        # v_node = self.layernorm_node_vec(v_node)
        # v_group = self.layernorm_group_vec(v_group)
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        # Message flow from node to group
        # relative_vec_embed = node_vec[edge_index[0]] - group_vec[edge_index[1]]

        # Message flow from group to node
        attn_2, val_2 = self.calculate_attention(node_embedding, group_embedding, edge_index[0], edge_index[1], edge_attr, self.model_2, "silu")
        m_s_node, m_v_node = self.propagate(edge_index.flip(0),
                                size = (num_groups, num_nodes),
                                x=(group_embedding, node_embedding), 
                                v = group_vec[edge_index[1]],
                                u_ij = -edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = attn_2, 
                                val = val_2[edge_index[1]],
                                mode = 'group_to_node')    
        

        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        return dx_node, dv_node
    






class LongShortIneractModel_dis_direct_two_way(MessagePassing):
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm = False,act = "silu",num_heads=8,**kwargs):
        super().__init__(aggr='add', node_dim = 0) # currently only node embedding is computed and updated, only group message flow to node
        self.act = act_class_mapping[act]()
        self.norm = norm
        self.layernorm_node = nn.LayerNorm(hidden_channels)
        self.layernorm_group = nn.LayerNorm(hidden_channels)
        self.layernorm_node_vec = nn.LayerNorm(hidden_channels)
        self.layernorm_group_vec = nn.LayerNorm(hidden_channels)
        self.model_2 = nn.ModuleDict({
            # 'mlp_edge_attr': nn.Sequential(
            #     nn.Linear(num_gaussians, hidden_channels),
            #     self.act,
            #     nn.Linear(hidden_channels, hidden_channels),),
            'q': nn.Linear(hidden_channels, hidden_channels),
            'k': nn.Linear(hidden_channels, hidden_channels),
            'val': nn.Linear(hidden_channels, hidden_channels),
            'mlp_scalar_pos': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
            'mlp_scalar_vec': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
            'linears': nn.ModuleList([nn.Linear(hidden_channels, hidden_channels, bias=False) for _ in range(6)])    
        })
        # self.model_1 = None
        self.model_1 = nn.ModuleDict({
            'mlp_scalar_pos': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
            'mlp_scalar': nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
            'mlp_scalar_vec': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
        })
        self.num_heads = num_heads
        self.attn_channels = hidden_channels // num_heads
        self.reset_parameters()

    def reset_parameters(self):
        '''
        Create Xavier Unifrom for all linear modules
        '''
        self.layernorm_node.reset_parameters()
        self.layernorm_group.reset_parameters()
        # self.attn_layers.reset_parameters()
        for model in [self.model_1, self.model_2]:
            if model is None:continue
            for _, value in model.items():
                if isinstance(value, nn.ModuleList):
                    for m in value.modules():
                        if isinstance(m, nn.Linear):
                            torch.nn.init.xavier_uniform_(m.weight)
                elif isinstance(value, nn.Linear):
                    torch.nn.init.xavier_uniform_(value.weight)
                    value.bias.data.fill_(0)
                else:
                    pass
            
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec):
        """_summary_

        Args:
            edge_index (_type_): 2*edge_num, [0] is node id,[1] is group id
            x_node (_type_): _description_
            x_group (_type_): _description_
            v_node (_type_): _description_
            v_group (_type_): _description_
            r_node (_type_): _description_
            r_group (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
        # v_node = self.layernorm_node_vec(v_node)
        # v_group = self.layernorm_group_vec(v_group)
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        # Message flow from node to group
        # relative_vec_embed = node_vec[edge_index[0]] - group_vec[edge_index[1]]
        m_s_group, m_v_group = self.propagate(edge_index,
                                size = (num_nodes, num_groups,),
                                x =(node_embedding, group_embedding, ), 
                                v = node_vec[edge_index[0]],
                                u_ij = edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = None, 
                                val = None,
                                mode = 'node_to_group')
        
        group_embedding = group_embedding + m_s_group
        group_vec = group_vec + m_v_group 
        
        # Message flow from group to node
        attn_2, val_2 = self.calculate_attention(node_embedding, group_embedding, edge_index[0], edge_index[1], edge_attr, self.model_2, "silu")
        m_s_node, m_v_node = self.propagate(edge_index.flip(0),
                                size = (num_groups, num_nodes),
                                x =(group_embedding, node_embedding), 
                                v = group_vec[edge_index[1]],
                                u_ij = -edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = attn_2, 
                                val = val_2[edge_index[1]],
                                mode = 'group_to_node')       
        
        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        return dx_node, dv_node
    
    def calculate_attention(self,x_1, x_2, x1_index, x2_index, expanded_edge_weight, model, attn_type):
        r'''
        Calculate attention value for each edge.
        x_1: embedding for query. target node embedding.
        x_2: embedding for key value. source node embedding.
        edge_index: graph to calculate attention
        expanded_edge_weight: the expanded edge weight
        model: the model for calculating attention, be a dictionary with keys: q, k, val, mlp_edge_attr, mlp_scalar_pos, mlp_scalar_vec
        '''
        __supported_attn__ = ['softmax', 'silu']
        q = model['q'](x_1).reshape(-1, self.num_heads, self.attn_channels) # num_groups x num_heads x attn_channels
        k = model['k'](x_2).reshape(-1, self.num_heads, self.attn_channels) # num_nodes x num_heads x attn_channels
        val = model['val'](x_2).reshape(-1, self.num_heads, self.attn_channels) 

        q_i = q[x1_index]
        k_j = k[x2_index]

        expanded_edge_weight = expanded_edge_weight.reshape(-1, self.num_heads, self.attn_channels)
        attn = q_i * k_j * expanded_edge_weight
        attn = attn.sum(dim = -1) / math.sqrt(self.attn_channels)
        # attn = attn.sum(dim = -1) / self.attn_channels
        # attn = attn.sum(dim = -1)

            
        if attn_type == 'softmax':
            attn = softmax(attn, x1_index, dim = 0)
        elif attn_type == 'silu':
            attn = act_class_mapping['silu']()(attn)
        else:
            raise NotImplementedError(f'Attention type {attn_type} is not supported, supported types are {__supported_attn__}')
        return attn, val      

    def message(self, x_i, x_j, v, u_ij, d_ij, attn_score, val, mode):
        '''
        Calculate the message from node j to node i
        Return: Scalar message from node j to node i, Vector message from node j to node i
        @param x_j: Node feature of node j
        @param x_i: Node feature of node i
        @param smeared_distance: Smearing distance between node i and node j
        @param v: Embedding of the group th
        @param mode: 'node_to_group' or 'group_to_node'
        '''

        
        # distance gated attention, larger the distance, smaller the attention should be.
        # There is the intuition, but we gave the model flexibility to learn on its own.
        if mode == 'node_to_group':
            model = self.model_1
            m_s_ij = model['mlp_scalar'](torch.cat([x_i, x_j], dim = -1))
            m_v_ij = model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * v + \
            model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1)
            return m_s_ij, m_v_ij
        else:
            model = self.model_2
        

        # a_ij = self.act(q_i * k_j * edge_attr.reshape(-1, self.num_heads, self.attn_channels) / (self.attn_channels ** 0.5))
        # a_ij = a_ij.sum(dim = -1, keepdim = True)
        # num_nodes, num_heads, attn_channels = q_i.shape
        
        # a_ij = q_i * k_j *  / (self.attn_channels ** 0.5)
        # a_ij = a_ij.(dim = -1, keepdim = True)
        
        m_s_ij = val * attn_score.unsqueeze(-1) # scalar message
        m_s_ij = m_s_ij.reshape(-1, self.num_heads * self.attn_channels)
        m_v_ij = model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1) \
        + model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * (v) # vector message
        return  m_s_ij, m_v_ij
        
    def aggregate(self, features, index, ptr, dim_size):
        # x, vec, w = features
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        # w = scatter(w, index, dim=self.node_dim, dim_size=dim_size)
        # return x, vec, w
        return x, vec
    

class LongShortIneractModel_dis_direct_new(MessagePassing):
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm = False,
                 act = "silu",num_heads=8,**kwargs):
        super().__init__(aggr='add', node_dim = 0) # currently only node embedding is computed and updated, only group message flow to node
        self.act = act_class_mapping[act]()
        self.norm = norm
        self.layernorm_node = nn.LayerNorm(hidden_channels)
        self.layernorm_group = nn.LayerNorm(hidden_channels)
        self.layernorm_node_vec = nn.LayerNorm(hidden_channels)
        self.layernorm_group_vec = nn.LayerNorm(hidden_channels)
        self.model_2 = nn.ModuleDict({
            'q': nn.Linear(hidden_channels, hidden_channels),
            'k': nn.Linear(hidden_channels, hidden_channels),
            'val': nn.Linear(hidden_channels, hidden_channels),
            'mlp_scalar': nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels)),
            'mlp_scalar_pos': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
            'mlp_scalar_vec': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
            'linears': nn.ModuleList([nn.Linear(hidden_channels, hidden_channels, bias = False) for _ in range(3)]),
            'linears_scalar': nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(3)]),
            'vec_linear': VecLinear2(hidden_channels)    
        })
        self.cutoff_func = CosineCutoff(cutoff)
        self.model_1 = None
        # self.model_1 = nn.ModuleDict({
        #     'mlp_scalar_pos': nn.Sequential(
        #         nn.Linear(hidden_channels, hidden_channels),
        #         self.act,
        #         nn.Linear(hidden_channels, hidden_channels),),
        #     'mlp_scalar': nn.Sequential(
        #         nn.Linear(2 * hidden_channels, hidden_channels),
        #         self.act,
        #         nn.Linear(hidden_channels, hidden_channels),),
        #     'mlp_scalar_vec': nn.Sequential(
        #         nn.Linear(hidden_channels, hidden_channels),
        #         self.act,
        #         nn.Linear(hidden_channels, hidden_channels),),
        # })
        self.num_heads = num_heads
        self.attn_channels = hidden_channels // num_heads
        self.reset_parameters()

    def reset_parameters(self):
        '''
        Create Xavier Unifrom for all linear modules
        '''
        self.layernorm_node.reset_parameters()
        self.layernorm_group.reset_parameters()
        # self.attn_layers.reset_parameters()
        for model in [self.model_1, self.model_2]:
            if model is None:continue
            for _, value in model.items():
                if isinstance(value, nn.ModuleList):
                    for m in value.modules():
                        if isinstance(m, nn.Linear):
                            torch.nn.init.xavier_uniform_(m.weight)
                elif isinstance(value, nn.Linear):
                    torch.nn.init.xavier_uniform_(value.weight)
                    value.bias.data.fill_(0)
                else:
                    pass
            
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec):
        """_summary_

        Args:
            edge_index (_type_): 2*edge_num, [0] is node id,[1] is group id
            x_node (_type_): _description_
            x_group (_type_): _description_
            v_node (_type_): _description_
            v_group (_type_): _description_
            r_node (_type_): _description_
            r_group (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
        # v_node = self.layernorm_node_vec(v_node)
        # v_group = self.layernorm_group_vec(v_group)
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        
        # Message flow from group to node
        attn_2, val_2 = self.calculate_attention(node_embedding, group_embedding, edge_index[0], edge_index[1], edge_attr, self.model_2, "silu", edge_weight)
        m_s_node, m_v_node = self.propagate(edge_index.flip(0),
                                size = (num_groups, num_nodes),
                                x =(group_embedding, node_embedding), 
                                v = group_vec[edge_index[1]],
                                u_ij = -edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = attn_2, 
                                val = val_2[edge_index[1]],
                                mode = 'group_to_node')       
        
        v_node_1 = self.model_2['linears'][0](node_vec)
        v_node_2 = self.model_2['linears'][1](node_vec)
        dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears_scalar'][0](m_s_node) + self.model_2['linears_scalar'][1](m_s_node)
        dv_node = m_v_node + self.model_2['linears_scalar'][2](m_s_node).unsqueeze(1) * self.model_2['linears'][2](node_vec)
        return dx_node, dv_node
    
    def calculate_attention(self,x_1, x_2, x1_index, x2_index, expanded_edge_weight, model, attn_type, edge_weight = None):
        r'''
        Calculate attention value for each edge.
        x_1: embedding for query. target node embedding.
        x_2: embedding for key value. source node embedding.
        edge_index: graph to calculate attention
        expanded_edge_weight: the expanded edge weight
        model: the model for calculating attention, be a dictionary with keys: q, k, val, mlp_edge_attr, mlp_scalar_pos, mlp_scalar_vec
        '''
        __supported_attn__ = ['softmax', 'silu']
        q = model['q'](x_1).reshape(-1, self.num_heads, self.attn_channels) # num_groups x num_heads x attn_channels
        k = model['k'](x_2).reshape(-1, self.num_heads, self.attn_channels) # num_nodes x num_heads x attn_channels
        val = model['val'](x_2).reshape(-1, self.num_heads, self.attn_channels) 

        q_i = q[x1_index]
        k_j = k[x2_index]

        expanded_edge_weight = expanded_edge_weight.reshape(-1, self.num_heads, self.attn_channels)
        attn = q_i * k_j * expanded_edge_weight
        if edge_weight is not None:
            attn = attn.sum(dim = -1) 
        else:
            attn = attn.sum(dim = -1)

            
        if attn_type == 'softmax':
            attn = softmax(attn, x1_index, dim = 0) * self.cutoff_func(edge_weight).unsqueeze(1)
        elif attn_type == 'silu':
            attn = act_class_mapping['silu']()(attn) * self.cutoff_func(edge_weight).unsqueeze(1)
        else:
            raise NotImplementedError(f'Attention type {attn_type} is not supported, supported types are {__supported_attn__}')
        return attn, val      

    def message(self, x_i, x_j, v, u_ij, d_ij, attn_score, val, mode):
        '''
        Calculate the message from node j to node i
        Return: Scalar message from node j to node i, Vector message from node j to node i
        @param x_j: Node feature of node j
        @param x_i: Node feature of node i
        @param smeared_distance: Smearing distance between node i and node j
        @param v: Embedding of the group th
        @param mode: 'node_to_group' or 'group_to_node'
        '''

        
        # distance gated attention, larger the distance, smaller the attention should be.
        # There is the intuition, but we gave the model flexibility to learn on its own.
        if mode == 'node_to_group':
            model = self.model_1
            m_s_ij = model['mlp_scalar'](torch.cat([x_i, x_j], dim = -1))
            m_v_ij = model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * v + \
            model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1)
            return m_s_ij, m_v_ij
        else:
            model = self.model_2
        

        # a_ij = self.act(q_i * k_j * edge_attr.reshape(-1, self.num_heads, self.attn_channels) / (self.attn_channels ** 0.5))
        # a_ij = a_ij.sum(dim = -1, keepdim = True)
        # num_nodes, num_heads, attn_channels = q_i.shape
        
        # a_ij = q_i * k_j *  / (self.attn_channels ** 0.5)
        # a_ij = a_ij.(dim = -1, keepdim = True)

        m_s_ij = val * attn_score.unsqueeze(-1) # scalar message
        m_s_ij = m_s_ij.reshape(-1, self.num_heads * self.attn_channels)
        v, s = model['vec_linear'](v)        
        m_s_ij = model['mlp_scalar'](torch.cat([s, m_s_ij], dim = -1))
        m_v_ij = model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1) \
        + model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * (v) # vector message
        return  m_s_ij, m_v_ij
        
    def aggregate(self, features, index, ptr, dim_size):
        # x, vec, w = features
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        # w = scatter(w, index, dim=self.node_dim, dim_size=dim_size)
        # return x, vec, w
        return x, vec
    
