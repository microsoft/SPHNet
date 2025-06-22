# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from .QHNet_utils import *


class ExpertPairNetLayerSymmetry(PairNetLayer_symmetry):
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
                 nonlinear='ssp'):
        super().__init__(irrep_in_node, irrep_bottle_hidden, irrep_out, sh_irrep, edge_attr_dim, node_attr_dim, resnet, invariant_layers, invariant_neurons, tp_mode, nonlinear)

    # Modify the argument list of the original forward function
    def forward(self, edge_index, edge_attr, node_attr, node_pair_attr=None):
        dst, src = edge_index
        node_attr_0 = self.linear_node_pair_inner(node_attr)
        s0 = self.inner_product(node_attr_0[dst], node_attr_0[src])[:, self.irrep_in_node.slices()[0].stop:]
        s0 = torch.cat([0.5*node_attr_0[dst][:, self.irrep_in_node.slices()[0]]+
                        0.5*node_attr_0[src][:, self.irrep_in_node.slices()[0]], s0], dim=-1)

        node_attr = self.norm_gate_pre(node_attr)
        node_attr = self.linear_node_pair_input(node_attr)

        node_pair = self.tp_node_pair(node_attr[src], node_attr[dst],
            self.fc_node_pair(edge_attr) * self.fc(s0))

        node_pair = self.norm_gate(node_pair)
        node_pair = self.node_residual(node_pair)

        if self.resnet and node_pair_attr is not None:
            node_pair = node_pair + node_pair_attr
        return node_pair


class ExpertPairNetLayer(PairNetLayer):
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
                 nonlinear='ssp'):
        super().__init__(irrep_in_node, irrep_bottle_hidden, irrep_out, sh_irrep, edge_attr_dim, node_attr_dim, resnet, invariant_layers, invariant_neurons, tp_mode, nonlinear)

    # Modify the argument list of the original forward function
    def forward(self, edge_index, edge_attr, node_attr, node_pair_attr=None):
        dst, src = edge_index
        node_attr_0 = self.linear_node_pair_inner(node_attr)
        s0 = self.inner_product(node_attr_0[dst], node_attr_0[src])[:, self.irrep_in_node.slices()[0].stop:]
        s0 = torch.cat([node_attr_0[dst][:, self.irrep_in_node.slices()[0]],
                        node_attr_0[src][:, self.irrep_in_node.slices()[0]], s0], dim=-1)

        node_attr = self.norm_gate_pre(node_attr)
        node_attr = self.linear_node_pair_n(node_attr)

        node_pair = self.tp_node_pair(node_attr[src], node_attr[dst],
            self.fc_node_pair(edge_attr) * self.fc(s0))

        node_pair = self.norm_gate(node_pair)
        node_pair = self.linear_node_pair(node_pair)

        if self.resnet and node_pair_attr is not None:
            node_pair = node_pair + node_pair_attr
        return node_pair

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, expert_model, num_experts, hidden_size, noisy_gating=True, k=4):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList([expert_model for _ in range(num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(hidden_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(hidden_size, num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, data, node_features, x, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](data, node_features, x) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss


class MoESelfLayer(MoE):
    def __init__(self,
                 num_experts, 
                 gate_hidden_size, 
                 noisy_gating, 
                 k,  
                 irrep_in_node,
                 irrep_bottle_hidden,
                 irrep_out,
                 sh_irrep,
                 edge_attr_dim,
                 node_attr_dim,
                 resnet: bool = True,
                 tp_mode = "uuu",
                 nonlinear='ssp'):
        self.gate_hidden_size = gate_hidden_size
        expert_model = SelfNetLayer(
            irrep_in_node,
            irrep_bottle_hidden,
            irrep_out,
            sh_irrep,
            edge_attr_dim,
            node_attr_dim,
            resnet=resnet,
            tp_mode=tp_mode,
            nonlinear=nonlinear
        )
        super().__init__(expert_model, num_experts, gate_hidden_size, noisy_gating, k)
    

    def forward(self, data, x, fii = None,  loss_coef=0.01):
        # get atom embeddings
        # get the type zero node features
        # potential bugs: x might not have type zero node features
        x_gating = x[:, :self.gate_hidden_size]
        gates, load = self.noisy_top_k_gating(x_gating, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        x_dispached = dispatcher.dispatch(x)
        if fii is not None:
            fii_dispatched = dispatcher.dispatch(fii)
        else:
            fii_dispatched = [None for _ in range(self.num_experts)]

        gates = dispatcher.expert_to_gates()
        # parrallel computation of experts
        expert_outputs = [self.experts[i](data, x_dispached[i], fii_dispatched[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        data['MoESelfLoss'] = loss
        return y


class MoEPairLayer(MoE):
    def __init__(self,
                 num_experts, 
                 gate_hidden_size, 
                 noisy_gating, 
                 k,  
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
                 nonlinear='ssp'):
        
        expert_model = ExpertPairNetLayer(
            irrep_in_node,
            irrep_bottle_hidden,
            irrep_out,
            sh_irrep,
            edge_attr_dim,
            node_attr_dim,
            resnet=resnet,
            invariant_layers=invariant_layers,
            invariant_neurons=invariant_neurons,
            tp_mode=tp_mode,
            nonlinear=nonlinear
        )
        super().__init__(expert_model, num_experts, gate_hidden_size, noisy_gating, k)
    

    def forward(self, data, node_features, fij = None,  loss_coef=0.01):
        edge_index = data.full_edge_index
        edge_features = data.full_edge_attr
        # get atom embeddings
        gates, load = self.noisy_top_k_gating(edge_features, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        # the dispatcher accept batch dimension as the first dimension, hence we need to transpose the edge_index
        edge_indexs_dispatched_transposed = dispatcher.dispatch(edge_index.transpose(0, 1))
        edge_features_dispatched = dispatcher.dispatch(edge_features)
        if fij is not None:
            fij_dispatched = dispatcher.dispatch(fij)
        else:
            fij_dispatched = [None for _ in range(self.num_experts)]

        gates = dispatcher.expert_to_gates()
        # parrallel computation of experts
        expert_outputs = [self.experts[i](edge_indexs_dispatched_transposed[i].transpose(0,1), edge_features_dispatched[i], node_features.clone(), fij_dispatched[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        data['MoEPairLoss'] = loss
        return y
    

class MoEPairLayerSymmetry(MoE):
    def __init__(self,
                 num_experts, 
                 gate_hidden_size, 
                 noisy_gating, 
                 k,  
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
                 nonlinear='ssp'):
        
        expert_model = ExpertPairNetLayerSymmetry(
            irrep_in_node,
            irrep_bottle_hidden,
            irrep_out,
            sh_irrep,
            edge_attr_dim,
            node_attr_dim,
            resnet=resnet,
            invariant_layers=invariant_layers,
            invariant_neurons=invariant_neurons,
            tp_mode=tp_mode,
            nonlinear=nonlinear
        )
        super().__init__(expert_model, num_experts, gate_hidden_size, noisy_gating, k)

    def forward(self, data, node_features, fij = None,  loss_coef=0.01):
        edge_index = data.full_edge_index
        edge_features = data.full_edge_attr
        # get atom embeddings
        gates, load = self.noisy_top_k_gating(edge_features, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        # the dispatcher accept batch dimension as the first dimension, hence we need to transpose the edge_index
        edge_indexs_dispatched_transposed = dispatcher.dispatch(edge_index.transpose(0, 1))
        edge_features_dispatched = dispatcher.dispatch(edge_features)
        if fij is not None:
            fij_dispatched = dispatcher.dispatch(fij)
        else:
            fij_dispatched = [None for _ in range(self.num_experts)]

        gates = dispatcher.expert_to_gates()
        # parrallel computation of experts
        expert_outputs = [self.experts[i](edge_indexs_dispatched_transposed[i].transpose(0,1), edge_features_dispatched[i], node_features.clone(), fij_dispatched[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        data['MoEPairLoss'] = loss
        return y
        