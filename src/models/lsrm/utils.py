import torch
import torch.nn as nn
from .torchmdnet.models.utils import (
    act_class_mapping,
)
from .torchmdnet.models.utils import norm
PI = 3.141592653589793

class LinearBiasSmall(nn.Module):
    def __init__(self, in_channels, out_channels, bias_norm = 1e-6):
        super(LinearBiasSmall, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias_norm_param = bias_norm
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels, 3))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        sum_of_square = torch.sum(self.bias ** 2, dim = 1)
        # if torch.any(sum_of_square < self.bias_norm_param):
        bais_norm = torch.sqrt(sum_of_square + 1e-8)
        if torch.any(bais_norm < self.bias_norm_param):
            bais_norm = torch.where(bais_norm < self.bias_norm_param, self.bias_norm_param * torch.ones_like(bais_norm), bais_norm)
        return torch.matmul(x, self.weight.t()) + (self.bias / bais_norm.unsqueeze(1)).unsqueeze(0).transpose(2, 1).contiguous()

class WOperator(nn.Module):
    def __init__(self, hidden_channels):
        super(WOperator, self).__init__()
        self.hidden_channels = hidden_channels
        self.linear = LinearBiasSmall(hidden_channels, hidden_channels)
        self.weight = nn.Parameter(torch.Tensor(1, 3, hidden_channels))
        
    def forward(self, vec):
        return self.linear(vec.cross(self.weight, dim = 1)), self.linear(vec).sum(dim = 1)

class VecLinear(nn.Module):
    def __init__(self, hidden_channels):
        super(VecLinear, self).__init__()
        self.hidden_channels = hidden_channels
        self.w = WOperator(hidden_channels)
        self.linear_1 = LinearBiasSmall(hidden_channels, hidden_channels)
        self.linear_2 = LinearBiasSmall(2 * hidden_channels, hidden_channels)
        
    def forward(self, v):
        vout_1, s = self.w(v)
        vout_2 = self.linear_1(v)
        v_out = self.linear_2(torch.cat([vout_1, vout_2], dim = 2))
        return v_out, s

class VecLinear2(nn.Module):
    def __init__(self, hidden_channels):
        super(VecLinear2, self).__init__()
        self.hidden_channels = hidden_channels
        self.linear_1 = nn.Linear(hidden_channels, 2 * hidden_channels, bias = False)
        self.linear_2 = nn.Linear(2 * hidden_channels, hidden_channels, bias = False)
        
    def forward(self, v):
        v1, v2 = torch.split(self.linear_1(v), self.hidden_channels, dim = 2)
        v3 = v1.cross(v2, dim = 1)
        v_out = self.linear_2(torch.cat([v, v3], dim = 2))
        return v_out, (v1 * v2).sum(dim = 1)
        
    
class VisInterAction(nn.Module):
    '''
    This class performs interaction between vector and scalar features
    '''
    def __init__(self, hidden_channels, act) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.act = act_class_mapping[act]()
        self.vec_linear = VecLinear(hidden_channels)
        self.linear_2 = nn.Linear(hidden_channels, hidden_channels, bias = False)
        
    
    def forward(self, x, v):
        '''
        x: scalar features, shape: [num_nodes, hidden_channels]
        v: vector features, shape: [num_nodes, 3, hidden_channels]
        '''
        vout_1, s = self.vec_linear(v)
        vout_2 = self.linear_1(v)
        v_out = self.linear_2(torch.cat([vout_1, vout_2], dim = 2)), s
        return v_out, s




class Dropout1d(torch.nn.Module):
    '''
    Dropout channel wise
    '''
    def __init__(self, p=0.5):
        super(Dropout1d, self).__init__()
        self.p = p

    def forward(self, x):
        # x: batch_size x 3 x channel
        if self.training:
            mask = torch.empty(x.size(0), 1, x.size(2), device=x.device).bernoulli_(1 - self.p)
            return x * mask / (1 - self.p)
        return x

    def extra_repr(self):
        return 'p={}'.format(self.p)

def get_distance(source_pos,target_pos,edge_index):
    """_summary_

    Args:
        edge_index 2*edge_num: [source, target]

    Returns:
        _type_: _description_
    """

    edge_vec = source_pos[edge_index[0]] - target_pos[edge_index[1]]
    mask = edge_index[0] == edge_index[1]
    edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
    edge_weight[~mask] = torch.norm(edge_vec[~mask], p=2, dim=1)
    
    # if(source_pos.shape[0]==target_pos.shape[1]):
    #     mask = edge_index[0] != edge_index[1]
    #     edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
    #     edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
    # else:
    #     edge_weight = torch.norm(edge_vec, dim=-1)
    # edge_weight = torch.norm(edge_vec, dim=-1) if not torch.all(edge_vec == 0) else torch.zeros(edge_vec.size(0), device=edge_vec.device)
    
    # if self.loop:
    #     # mask out self loops when computing distances because
    #     # the norm of 0 produces NaN gradients
    #     # NOTE: might influence force predictions as self loop gradients are ignored
    #     mask = edge_index[0] != edge_index[1]
    #     edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
    #     edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
    # else:
    #     edge_weight = torch.norm(edge_vec, dim=-1)
    # lower_mask = edge_weight >= self.cutoff_lower
    # edge_index = edge_index[:, lower_mask]
    # edge_weight = edge_weight[lower_mask]

    # if self.return_vecs:
    #     edge_vec = edge_vec[lower_mask]
    #     return edge_index, edge_weight, edge_vec
    # # TODO: return only `edge_index` and `edge_weight` once
    # # Union typing works with TorchScript (https://github.com/pytorch/pytorch/pull/53180)
    # return edge_index, edge_weight, None
    
    return edge_index,edge_weight,edge_vec
    