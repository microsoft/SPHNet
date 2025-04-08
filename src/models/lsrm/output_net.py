import torch
from torch import nn
from torch_scatter import scatter
from .torchmdnet.models.utils import act_class_mapping

class OutputNet(nn.Module):
    def __init__(self, hidden_channels, act = 'silu', dipole = False, mean = None, std = None, atomref = None, scale = None, mean_std_adder = 'molecule_level') -> None:
        '''
        Output Module 
        If mean_std_adder is 'atom_level', then the mean and std are added to the output at the atom level.
        If mean_std_adder is 'molecule_level', then the mean and std are added to the output at the molecular level.
        '''
        __MEAN_STD_ADDER__ = ['atom_level', 'molecule_level']
        super().__init__()
        self.dipole = dipole
        self.scale = scale
        self.readout = 'sum'
        self.scale = scale
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.register_buffer('atomref', atomref)
        act_class = act_class_mapping[act]
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2 ),
            act_class(),
            nn.Linear(hidden_channels // 2 , 1),
        )   
        self.mean_std_adder = mean_std_adder
        assert self.mean_std_adder in __MEAN_STD_ADDER__, f"mean_std_adder must be one of {__MEAN_STD_ADDER__}"
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)
        
    def forward(self, h, v,data):
        z = data.atomic_numbers.long()
        pos = data.pos
        
        batch = data.batch
        h = self.output_network(h)
        if self.dipole:
            # Get center of mass.
            c = scatter(z * pos, batch, dim=0) / scatter(z, batch, dim=0)
            h = h * (pos - c.index_select(0, batch))


        if not self.dipole and self.mean is not None and self.std is not None and self.mean_std_adder == 'atom_level':
            h = h * self.std + self.mean
            
        # if not self.dipole and self.atomref is not None:
        #     h = h + self.atomref[z]
            
                       
        out = scatter(h, batch, dim=0, reduce=self.readout)

        if not self.dipole and self.mean is not None and self.std is not None and self.mean_std_adder == 'molecule_level':
            out = out * self.std + self.mean
        
        if self.atomref is not None:
            out = out + scatter(self.atomref[z],batch, dim=0, reduce='sum')
        
        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out
        
        return out


class EquivariantScalar(nn.Module):
    def __init__(self, hidden_channels, act="silu",dipole = False,  mean=None,std = None,atomref = None,scale = None,mean_std_adder = 'molecule_level'):
        super(EquivariantScalar, self).__init__()
        
        self.dipole = dipole
        self.scale = scale
        self.readout = 'sum'
        self.scale = scale
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.register_buffer('atomref', atomref)
        act_class = act_class_mapping[act]
        
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=act,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(hidden_channels // 2, 1, activation=act),
            ]
        )
        self.mean_std_adder = mean_std_adder
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x, v,data):
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch
        
        for layer in self.output_network:
            x, v = layer(x, v)
        # include v in output to make sure all parameters have a gradient
        h = x + v.sum() * 0
        

        if not self.dipole and self.mean is not None and self.std is not None and self.mean_std_adder == 'atom_level':
            h = h * self.std + self.mean
            
        # if not self.dipole and self.atomref is not None:
        #     h = h + self.atomref[z]
            
                       
        out = scatter(h, batch, dim=0, reduce=self.readout)

        if not self.dipole and self.mean is not None and self.std is not None and self.mean_std_adder == 'molecule_level':
            out = out * self.std + self.mean
        
                
        if self.atomref is not None:
            out = out + scatter(self.atomref[z],batch, dim=0, reduce=self.readout)
        
        
        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out
        
        return out
        
        
class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        activation="silu",
        scalar_activation=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        act_class = act_class_mapping[activation]
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            act_class(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = act_class() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v
    