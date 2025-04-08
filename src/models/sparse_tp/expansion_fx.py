import math
import torch
import torch.nn as nn
import torch.fx as fx
from typing import List, Optional, Union
import random
from e3nn import o3
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode
from e3nn.o3._tensor_product._instruction import Instruction


def prod(shape):
    """ e3nn.util.prod """
    p = 1
    for s in shape:
        p *= s
    return p

def codegen_expansion_fx(
    irreps_in: o3.Irreps,
    irreps_out_1: o3.Irreps,
    irreps_out_2: o3.Irreps,
    instructions: List[Instruction],
    # has_internal_weights: bool,
):

    graph = fx.Graph()
    tracer = fx.proxy.GraphAppendingTracer(graph)

    x_in_proxy = fx.Proxy(graph.placeholder("x_in", torch.Tensor), tracer=tracer)
    external_w_proxy = fx.Proxy(graph.placeholder("external_weights", torch.Tensor), tracer=tracer)
    external_b_proxy = fx.Proxy(graph.placeholder("external_bias_weights", torch.Tensor), tracer=tracer)

    batch_num = x_in_proxy.shape[0]

    if len(irreps_in) == 1:
        x_in_s = [x_in_proxy.reshape(batch_num, irreps_in[0].mul, irreps_in[0].ir.dim)]
    else:
        x_in_s = [
            x_in_proxy[:, i].reshape(batch_num, mul_ir.mul, mul_ir.ir.dim)
        for i, mul_ir in zip(irreps_in.slices(), irreps_in)]

    outputs = {}
    flat_weight_index = 0
    bias_weight_index = 0

    for ins in instructions:
        # parse
        i_in = ins[0]
        i_out1 = ins[1]
        i_out2 = ins[2]
        # has_w = ins[3]
        # shape = ins[-1]
        # path_weight = ins[4]

        mul_ir_in = irreps_in[i_in]
        mul_ir_out1 = irreps_out_1[i_out1]
        mul_ir_out2 = irreps_out_2[i_out2]

        x1 = x_in_s[i_in]
        x1 = x1.reshape(batch_num, mul_ir_in.mul, mul_ir_in.ir.dim)

        w3j_attr_name = f"w3j_{i_out1}_{i_out2}_{i_in}"
        get_attr_node = graph.get_attr(w3j_attr_name)
        w3j_matrix = fx.Proxy(get_attr_node, tracer=tracer)
        
        weight = external_w_proxy[:, flat_weight_index:flat_weight_index + prod(ins[-1])].reshape([-1] + ins[-1])
        result = torch.einsum(f"bwuv, bwk-> buvk", weight, x1)
        if ins[0] == 0 and external_b_proxy is not None:
            bias_weight = external_b_proxy[:,bias_weight_index:bias_weight_index + prod(ins[-1][1:])].reshape([-1] + ins[-1][1:])
            bias_weight_index += prod(ins[-1][1:])
            result = result + bias_weight.unsqueeze(-1)

        result = torch.einsum(f"ijk, buvk->buivj", w3j_matrix, result) / mul_ir_in.mul
        flat_weight_index += prod(ins[-1])

        result = result.reshape(batch_num, mul_ir_out1.dim, mul_ir_out2.dim)
        key = (ins[1], ins[2])
        if key in outputs.keys():
            outputs[key] = outputs[key] + result
        else:
            outputs[key] = result

    
    rows = []
    for i in range(len(irreps_out_1)):
        blocks = []
        for j in range(len(irreps_out_2)):
            if (i, j) not in outputs.keys():
                blocks += [torch.zeros((x_in_proxy.shape[0], irreps_out_1[i].dim, irreps_out_2[j].dim),
                                        device=x_in_proxy.device).type(x_in_proxy.type())]
            else:
                blocks += [outputs[(i, j)]]
        rows.append(torch.cat(blocks, dim=-1))
    output = torch.cat(rows, dim=-2)

    graph.output(output.node)
    # check graphs
    graph.lint()


    module = nn.Module()

    for ins in instructions:
        i_in = ins[0]
        i_out1 = ins[1]
        i_out2 = ins[2]
        w3j_attr_name = f"w3j_{i_out1}_{i_out2}_{i_in}"
        module.register_buffer(w3j_attr_name, o3.wigner_3j(i_out1, i_out2, i_in))  # 仅演示

    return fx.GraphModule(module, graph)


@compile_mode("script")
class ExpansionFX(CodeGenMixin, nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out_1: o3.Irreps,
        irreps_out_2: o3.Irreps,
        # instructions,
    ):
        super().__init__()
        self.irrep_in = irreps_in
        self.irrep_out_1 = irreps_out_1
        self.irrep_out_2 = irreps_out_2
        
        self.instructions = self.get_expansion_path(irreps_in, irreps_out_1, irreps_out_2)

        self.num_path_weight = sum(prod(ins[-1]) for ins in self.instructions if ins[3])
        self.num_bias = sum([prod(ins[-1][1:]) for ins in self.instructions if ins[0] == 0])
        self.num_weights = self.num_path_weight + self.num_bias

        graphmod_expansion = codegen_expansion_fx(
            self.irrep_in, 
            self.irrep_out_1, 
            self.irrep_out_2,
            self.instructions
        )

        self._codegen_register({"_compiled_expansion": graphmod_expansion})

    def forward(self, x_in, weights=None, bias_weights=None, use_sparse_tp=False):

        return self._compiled_expansion(x_in, weights, bias_weights)


    @property
    def device(self):
        return next(self.parameters()).device

    def __repr__(self):
        return (
            f"ExpansionFX(irrep_in={self.irrep_in}, irrep_out_1={self.irrep_out_1}, "
            f"irrep_out_2={self.irrep_out_2}, num_weights={self.num_weights}, num_bias={self.num_bias})"
        )

    def get_expansion_path(self, irrep_in, irrep_out_1, irrep_out_2):
        instructions = []
        for  i, (num_in, ir_in) in enumerate(irrep_in):
            for  j, (num_out1, ir_out1) in enumerate(irrep_out_1):
                for k, (num_out2, ir_out2) in enumerate(irrep_out_2):
                    if ir_in in ir_out1 * ir_out2:
                        instructions.append([i, j, k, True, 1.0, [num_in, num_out1, num_out2]])
        return instructions

