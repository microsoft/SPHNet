from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.nn as snn
from .utils import (vec_layernorm, max_min_norm)

from .torchmd_et import TorchMD_ET
from torch.nn.functional import silu

__all__ = ["PaiNN", "PaiNNInteraction", "PaiNNMixing"]
from torch_scatter import scatter_add


class PaiNNInteraction(nn.Module):
    r"""PaiNN interaction block for modeling equivariant interactions of atomistic systems."""

    def __init__(self, n_atom_basis: int, activation: Callable):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            activation: if None, no activation function is used.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super(PaiNNInteraction, self).__init__()
        self.n_atom_basis = n_atom_basis

        self.interatomic_context_net = nn.Sequential(
            snn.Dense(n_atom_basis, n_atom_basis, activation=activation),
            snn.Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )

    def forward(
        self,
        q: torch.Tensor,
        mu: torch.Tensor,
        Wij: torch.Tensor,
        dir_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        n_atoms: int,
    ):
        """Compute interaction output.
        Args:
            q: scalar input values
            mu: vector input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j
        Returns:
            atom features after interaction
        """
        # inter-atomic
        x = self.interatomic_context_net(q)
        xj = x[idx_j]
        muj = mu[idx_j]
        x = Wij * xj

        dq, dmuR, dmumu = torch.split(x, self.n_atom_basis, dim=-1)
        dq = scatter_add(dq, idx_i, dim=0)
        dmu = dmuR * dir_ij[..., None] + dmumu * muj
        dmu = scatter_add(dmu, idx_i, dim = 0)

        q = q + dq
        mu = mu + dmu

        return q, mu


class PaiNNMixing(nn.Module):
    r"""PaiNN interaction block for mixing on atom features."""

    def __init__(self, n_atom_basis: int, activation: Callable, epsilon: float = 1e-8):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            activation: if None, no activation function is used.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super(PaiNNMixing, self).__init__()
        self.n_atom_basis = n_atom_basis

        self.intraatomic_context_net = nn.Sequential(
            snn.Dense(2 * n_atom_basis, n_atom_basis, activation=activation),
            snn.Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )
        self.mu_channel_mix = snn.Dense(
            n_atom_basis, 2 * n_atom_basis, activation=None, bias=False
        )
        self.epsilon = epsilon

    def forward(self, q: torch.Tensor, mu: torch.Tensor):
        """Compute intraatomic mixing.
        Args:
            q: scalar input values
            mu: vector input values
        Returns:
            atom features after interaction
        """
        ## intra-atomic
        mu_mix = self.mu_channel_mix(mu)
        mu_V, mu_W = torch.split(mu_mix, self.n_atom_basis, dim=-1)
        mu_Vn = torch.sqrt(torch.sum(mu_V**2, dim=-2, keepdim=True) + self.epsilon)

        ctx = torch.cat([q, mu_Vn], dim=-1)
        x = self.intraatomic_context_net(ctx)

        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.n_atom_basis, dim=-1)
        dmu_intra = dmu_intra * mu_W

        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)

        q = q + dq_intra + dqmu_intra
        mu = mu + dmu_intra
        return q, mu

class PaiNN(TorchMD_ET):
    """PaiNN - polarizable interaction neural network
    References:
    .. [#painn1] Schütt, Unke, Gastegger:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html
    """

    def __init__(
        self,
        hidden_channels=128,
        num_layers=6,
        num_rbf=50,
        rbf_type="expnorm",
        trainable_rbf=True,
        activation="silu",
        attn_activation="silu",
        neighbor_embedding=True,
        num_heads=8,
        distance_influence="both",
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_z=100,
        max_num_neighbors=32,
        otf_graph=True,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            activation: activation function
            shared_interactions: if True, share the weights across
                interaction blocks.
            shared_interactions: if True, share the weights across
                filter-generating networks.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super(PaiNN, self).__init__(128, num_layers, num_rbf, 
                                    rbf_type, trainable_rbf, activation, 
                                    attn_activation, neighbor_embedding, 
                                    num_heads, distance_influence, 
                                    cutoff_lower, cutoff_upper, max_z, 
                                    max_num_neighbors, otf_graph)
        self.interactions = nn.ModuleList()
        self.mixings = nn.ModuleList()
        self.edge_proj = nn.Linear(num_rbf, 3 * self.hidden_channels)
        self.layer_norm = nn.ModuleList()
        for i in range(self.num_layers):
            self.interactions.append(PaiNNInteraction(self.hidden_channels, silu))
            self.mixings.append(PaiNNMixing(self.hidden_channels, silu))
            self.layer_norm.append(nn.LayerNorm(self.hidden_channels))


    def forward(self,
                data,
                **kwargs        
                ):
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch
        if z.dim() == 2: # if z of shape num_atoms x 1
            z = z.squeeze() # squeeze to num_atoms
        x = self.embedding(z).squeeze()
        n_atoms = x.size(0)
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        assert (
            edge_vec is not None
        ), "Distance module did not return directional information"

        edge_attr = self.distance_expansion(edge_weight)
        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)

        # if self.neighbor_embedding is not None:
        #     x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)

        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)
        edge_attr = self.edge_proj(edge_attr)
        x = x.unsqueeze(1)
        edge_attr = edge_attr.unsqueeze(1)
        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixings)):

            x, vec = interaction(x, vec, edge_attr, edge_vec, edge_index[0], edge_index[1], n_atoms)
            x, vec = mixing(x, vec)
            x = self.layer_norm[i](x)
            
        x = x.squeeze(1)
        x = self.out_norm(x)

        return x, vec, z, pos, batch