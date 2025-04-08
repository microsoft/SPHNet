import numpy as np
import torch
from torch.autograd import Function
from argparse import Namespace


convention_dict = {
    'pyscf_631G': Namespace(
        atom_to_orbitals_map={1: 'ss', 6: 'ssspp', 7: 'ssspp', 8: 'ssspp', 9: 'ssspp'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd':
                          [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1], 6: [0, 1, 2, 3, 4], 7:  [0, 1, 2, 3, 4],
            8:  [0, 1, 2, 3, 4], 9:  [0, 1, 2, 3, 4]
        },
    ),
    'pyscf_def2svp': Namespace(
        atom_to_orbitals_map={1: 'ssp', 6: 'sssppd', 7: 'sssppd', 8: 'sssppd', 9: 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [1, 2, 0], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2], 6: [0, 1, 2, 3, 4, 5], 7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5], 9: [0, 1, 2, 3, 4, 5]
        },
    ),
    'back2pyscf': Namespace(
        atom_to_orbitals_map={1: 'ssp', 6: 'sssppd', 7: 'sssppd', 8: 'sssppd', 9: 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2], 6: [0, 1, 2, 3, 4, 5], 7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5], 9: [0, 1, 2, 3, 4, 5]
        },
    ),
}


def matrix_transform(hamiltonian, atoms, convention='pyscf_def2svp'):
    conv = convention_dict[convention]
    orbitals = ''
    orbitals_order = []
    for a in atoms:
        offset = len(orbitals_order)
        orbitals += conv.atom_to_orbitals_map[a]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[a]]

    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        map_sign = conv.orbital_sign_map[orb]
        transform_indices.append(torch.tensor(map_idx) + offset)
        transform_signs.append(torch.tensor(map_sign))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    transform_indices = torch.hstack(transform_indices).to(torch.long).to(hamiltonian.device)
    transform_signs = torch.hstack(transform_signs).to(hamiltonian.device)

    hamiltonian_new = hamiltonian[..., transform_indices, :]
    hamiltonian_new = hamiltonian_new[..., :, transform_indices]
    hamiltonian_new = hamiltonian_new * transform_signs[:, None]
    hamiltonian_new = hamiltonian_new * transform_signs[None, :]

    return hamiltonian_new


eng_threshold = 1e-16

class power_iteration_once(Function):
    @staticmethod
    def forward(ctx, M, v_k, num_iter=19):
        '''
        :param ctx: used to save meterials for backward.
        :param M: n by n matrix.
        :param v_k: initial guess of leading vector.
        :return: v_k1 leading vector.
        '''
        ctx.num_iter = num_iter
        ctx.save_for_backward(M, v_k)
        return v_k

    @staticmethod
    def backward(ctx, grad_output):
        M, v_k = ctx.saved_tensors
        dL_dvk = grad_output
        I = torch.eye(M.shape[-1], out=torch.empty_like(M))
        numerator = I - v_k.mm(torch.t(v_k))
        denominator = torch.norm(M.mm(v_k)).clamp(min=1.e-5)
        ak = numerator / denominator
        term1 = ak
        q = M / denominator
        for i in range(1, ctx.num_iter + 1):
            ak = q.mm(ak)
            term1 += ak
        dL_dM = torch.mm(term1.mm(dL_dvk), v_k.t())
        return dL_dM, ak

def ED_PI_Layer(input, n_eigens, reverse, pi_iter=19):
    p = input
    dtype = p.dtype
    p = p.cpu() #SVD is faster on CPU
    eig_diag, eig_vec = torch.linalg.eigh(p)
    eig_diag = eig_diag.to(input.device)
    eig_vec = eig_vec.to(input.device)
    eig_diag[eig_diag <= eng_threshold] = eng_threshold
    eig_diag = eig_diag.diag_embed().type(dtype)

    new_eig_vecs = []
    idxes = list(range(n_eigens-1, -1, -1)) if reverse else list(range(n_eigens))
    xxt = input
    for i in idxes:
        new_eig_vec = power_iteration_once.apply(xxt, eig_vec[:, :, i], pi_iter)[..., None] # B, N, 1
        new_eig_vecs.append(new_eig_vec) 
        counter += 1
        xxt = xxt - torch.mm(torch.mm(xxt, new_eig_vec), new_eig_vec.t())

    new_eig_vecs = torch.stack(new_eig_vecs, dim=-1)
    if reverse:
        new_eig_vecs = new_eig_vecs.flip(2).permute(0, 2, 1)
    # TODO: check new_eig_vecs == eig_vec

    return eig_diag, new_eig_vecs

def truncated_gradients(s, trunc_factor, num_orb):
    dim = s.size(1)
    p = 1 / (s.unsqueeze(-1) - s.unsqueeze(-2))
    p[:, torch.arange(0, dim), torch.arange(0, dim)] = 0
    batch_std = p[:, :num_orb, :num_orb].std(dim=(1, 2)).mean()
    threshold = trunc_factor * batch_std
    p[p > threshold] = threshold
    p[p < -threshold] = -threshold
    return p

#ED Step
class ED_trunc(Function):
    @staticmethod
    def forward(ctx, input, trunc_factor=3, num_orb=-1):
        p = input
        p = p.cpu() #SVD is faster on CPU
        eig_diag, eig_vec = torch.linalg.eigh(p)
        eig_diag = eig_diag.to(input)
        eig_vec = eig_vec.to(input)
        trunc_factor = torch.tensor(trunc_factor).to(input)
        if num_orb == -1:
            num_orb = p.shape[-1]
        num_orb = torch.tensor(num_orb).to(device=input.device, dtype=torch.int64)
        # uniquely determine a set of eigenvectors
        # max_abs_cols = torch.argmax(eig_vec.abs(), dim=1)
        # signs = torch.sign(eig_vec.gather(1, max_abs_cols[:, None])) # [b, 1, n]
        # eig_vec = eig_vec * signs # [b, 1, n]
        # eig_diag[eig_diag <= eng_threshold] = eng_threshold
        ctx.save_for_backward(eig_diag, eig_vec, trunc_factor, num_orb)
        return eig_diag, eig_vec
    
    @staticmethod
    def backward(ctx, dvals, dvecs):
        # breakpoint()
        eig_diag, eig_vec, trunc_factor, num_orb = ctx.saved_tensors
        k = truncated_gradients(eig_diag, trunc_factor, num_orb)
        grad_input=(k.transpose(1, 2).conj() * (eig_vec.transpose(1,2).conj() @ dvecs)) + torch.diag_embed(dvals)
        #Gradient Overflow Check
        # grad_input[grad_input == float('inf')] = grad_input[grad_input != float('inf')].max()
        # grad_input[grad_input == float('-inf')] = grad_input[grad_input != float('-inf')].min()
        grad_input = eig_vec @ grad_input @ eig_vec.transpose(1,2).conj()
        # Gradient Overflow Check
        # grad_input[grad_input == float('inf')] = grad_input[grad_input != float('inf')].max()
        # grad_input[grad_input == float('-inf')] = grad_input[grad_input != float('-inf')].min()
        grad_input = (grad_input + grad_input.transpose(1,2).conj()) * 0.5
        return grad_input, None, None
    


def pad_symmetric_matrices(matrices):
    """
    Pads a list of symmetric matrices with zeros to match the size of the largest matrix in the list.
    For symmetric matrices, only adds 1s along the diagonal to preserve eigenvalues.
    
    Parameters:
    - matrices (list of torch.Tensor): A list of symmetric matrices (2D tensors) of varying sizes.

    Returns:
    - list of torch.Tensor: A list of padded symmetric matrices, all of the same size.
    """
    # Determine the max dimension (since they are symmetric, we only need to consider one dimension)
    max_size = max(max(matrix.shape) for matrix in matrices)
    
    padded_matrices = []
    for matrix in matrices:
        size = matrix.shape[0]  # Assuming square matrices
        # Calculate padding needed
        padding_size = max_size - size
        # Pad the matrix symmetrically and add 1s on the new diagonal elements if needed
        padded_matrix = torch.nn.functional.pad(matrix, (0, padding_size, 0, padding_size), "constant", 0)
        for i in range(size, max_size):
            padded_matrix[i, i] = 1.0  # Add 1s on the new diagonal elements
        padded_matrices.append(padded_matrix)
    
    return torch.stack(padded_matrices)