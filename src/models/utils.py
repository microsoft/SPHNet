from torch_cluster import radius_graph
import torch
from torch_geometric.data import Data
from argparse import Namespace
import numpy as np

CONVENTION_DICT = {
        'def2-tzvp': Namespace(
            atom_to_orbitals_map={1: 'sssp', 6: 'ssssspppddf', 7: 'ssssspppddf', 8: 'ssssspppddf',
                                9: 'ssssspppddf', 15: 'ssssspppppddf', 16: 'ssssspppppddf',
                                17: 'ssssspppppddf'},
            # as 17 is the atom with biggest orbitals, 5*1+5*3+2*5+1*7, thus s ~[0,5) p~[5,5+15) d~[20,20+2*5) f~[30,37)
            # thus max_block_size is 37
            str2idx = {"s":0,"p":0+5*1,"d":0+5*1+5*3,"f":0+5*1+5*3+2*5},
            max_block_size= 37,
            orbital_idx_map={'s': np.array([0]), 'p': np.array([2, 0, 1]), 
                             'd':np.array( [0, 1, 2, 3, 4]), 'f': np.array([0, 1, 2, 3, 4, 5, 6])},
        ),
        'def2-svp': Namespace(
            atom_to_orbitals_map={1: 'ssp', 6: 'sssppd', 7: 'sssppd', 8: 'sssppd',
                                9: 'sssppd'},
            # as 17 is the atom with biggest orbitals, 5*1+5*3+2*5+1*7, thus s ~[0,5) p~[5,5+15) d~[20,20+2*5) f~[30,37)
            # thus max_block_size is 37
            str2idx = {"s":0,"p":0+3*1,"d":0+3*1+2*3},
            max_block_size= 14,
            orbital_idx_map={'s': np.array([0]), 'p': np.array([2, 0, 1]), 
                             'd':np.array( [0, 1, 2, 3, 4])},
        ),
        
    }

def block2matrix(Z,diag,non_diag,mask_lin,max_block_size,sym = False):
    if isinstance(Z,torch.Tensor):
        if not isinstance(mask_lin[1],torch.Tensor):
            for key in mask_lin:
                mask_lin[key] = torch.from_numpy(mask_lin[key])
        Z = Z.reshape(-1)
        n_atom = len(Z)
        atom_orbitals = []
        for i in range(n_atom):
            atom_orbitals.append(i*max_block_size+mask_lin[Z[i].item()])
        atom_orbitals = torch.cat(atom_orbitals,dim= 0)

        rebuild_fock = torch.zeros((n_atom,n_atom,max_block_size,max_block_size)).to(Z.device)
        
        
        if sym:
            ## down
            rebuild_fock[torch.eye(n_atom)==1] = diag
            unit_matrix = torch.ones((n_atom,n_atom))
            down_triangular_matrix = unit_matrix- torch.triu(unit_matrix)
            rebuild_fock[down_triangular_matrix==1] = 2*non_diag
            rebuild_fock = (rebuild_fock + torch.permute(rebuild_fock,(1,0,3,2)))/2
        else:
            # no sym
            rebuild_fock[torch.eye(n_atom)==1] = diag
            unit_matrix = torch.ones((n_atom,n_atom))
            matrix_noeye = unit_matrix - torch.eye(len(Z))
            rebuild_fock[matrix_noeye==1] = non_diag
            rebuild_fock = (rebuild_fock + torch.permute(rebuild_fock,(1,0,3,2)))/2
        
        rebuild_fock = torch.permute(rebuild_fock,(0,2,1,3))
        rebuild_fock = rebuild_fock.reshape((n_atom*max_block_size,n_atom*max_block_size))
        rebuild_fock = rebuild_fock[atom_orbitals][:,atom_orbitals]
        return rebuild_fock
        
    else:
        Z = Z.reshape(-1)
        n_atom = len(Z)
        atom_orbitals = []
        for i in range(n_atom):
            atom_orbitals.append(i*max_block_size+mask_lin[Z[i]])
        atom_orbitals = np.concatenate(atom_orbitals,axis= 0)
        rebuild_fock = np.zeros((n_atom,n_atom,max_block_size,max_block_size))
        
        
        if sym:
            ## down
            rebuild_fock[np.eye(n_atom)==1] = diag
            unit_matrix = np.ones((n_atom,n_atom))
            down_triangular_matrix = unit_matrix- np.triu(unit_matrix)
            rebuild_fock[down_triangular_matrix==1] = 2*non_diag
            rebuild_fock = (rebuild_fock + rebuild_fock.transpose(1,0,3,2))/2
        else:
            # no sym
            rebuild_fock[np.eye(n_atom)==1] = diag
            unit_matrix = np.ones((n_atom,n_atom))
            matrix_noeye = unit_matrix - np.eye(len(Z))
            rebuild_fock[matrix_noeye==1] = non_diag
        
        rebuild_fock = rebuild_fock.transpose(0,2,1,3)
        rebuild_fock = rebuild_fock.reshape((n_atom*max_block_size,n_atom*max_block_size))
        rebuild_fock = rebuild_fock[atom_orbitals][:,atom_orbitals]
        return rebuild_fock
    
def get_conv_variable(basis = "def2-tzvp"):
    # str2order = {"s":0,"p":1,"d":2,"f":3}
    conv = CONVENTION_DICT[basis]
    mask = {}
    for atom in conv.atom_to_orbitals_map:
        mask[atom] = []
        orb_id = 0
        visited_orbital = set()
        for s in conv.atom_to_orbitals_map[atom]:
            if s not in visited_orbital:
                visited_orbital.add(s)
                orb_id = conv.str2idx[s]

            mask[atom].extend(conv.orbital_idx_map[s]+orb_id)
            orb_id += len(conv.orbital_idx_map[s])
    for key in mask:
        mask[key] = np.array(mask[key])
    return conv, None, mask,None

def get_full_graph(batch_data):
    full_edge_index = []
    # radius_graph(batch_data.pos, 1000, batch_data.batch,max_num_neighbors=1000)
    # batch_data["non_diag_hamiltonian"] = batch_data["non_diag_hamiltonian"][full_edge_index[0]>full_edge_index[1]]
    # batch_data['non_diag_mask'] = batch_data["non_diag_mask"][full_edge_index[0]>full_edge_index[1]]
    # full_edge_index = full_edge_index[:,full_edge_index[0]>full_edge_index[1]]
    atom_start = 0
    for n_atom in batch_data["molecule_size"].reshape(-1):
        n_atom = n_atom.item()
        full_graph = torch.stack([torch.arange(n_atom).reshape(-1,1).repeat(1,n_atom),torch.arange(n_atom).reshape(1,-1).repeat(n_atom,1)],axis = 0).reshape(2,-1)
        full_graph = full_graph[:,full_graph[0]!=full_graph[1]]
        full_edge_index.append(atom_start+full_graph)
        atom_start = atom_start + n_atom

    return torch.concat(full_edge_index,dim = 1).to(batch_data["molecule_size"].device)

def get_transpose_index(data, full_edges):
    start_edge_index = 0
    all_transpose_index = []
    for graph_idx in range(data.ptr.shape[0] - 1):
        num_nodes = data.ptr[graph_idx +1] - data.ptr[graph_idx]
        graph_edge_index = full_edges[:, start_edge_index:start_edge_index+num_nodes*(num_nodes-1)]
        sub_graph_edge_index = graph_edge_index - data.ptr[graph_idx]
        bias = (sub_graph_edge_index[0] < sub_graph_edge_index[1]).type(torch.int)
        transpose_index = sub_graph_edge_index[0] * (num_nodes - 1) + sub_graph_edge_index[1] - bias
        transpose_index = transpose_index + start_edge_index
        all_transpose_index.append(transpose_index)
        start_edge_index = start_edge_index + num_nodes*(num_nodes-1)
    return torch.cat(all_transpose_index, dim=-1)

def get_transpose_index(data, full_edges):
    start_edge_index = 0
    all_transpose_index = []
    for graph_idx in range(data.ptr.shape[0] - 1):
        num_nodes = data.ptr[graph_idx +1] - data.ptr[graph_idx]
        graph_edge_index = full_edges[:, start_edge_index:start_edge_index+num_nodes*(num_nodes-1)]
        sub_graph_edge_index = graph_edge_index - data.ptr[graph_idx]
        bias = (sub_graph_edge_index[0] < sub_graph_edge_index[1]).type(torch.int)
        transpose_index = sub_graph_edge_index[0] * (num_nodes - 1) + sub_graph_edge_index[1] - bias
        transpose_index = transpose_index + start_edge_index
        all_transpose_index.append(transpose_index)
        start_edge_index = start_edge_index + num_nodes*(num_nodes-1)
    return torch.cat(all_transpose_index, dim=-1)

def get_toy_data(n_atom = 100):
    out = {}
    neighbors = 30
    atom_max_orbital = 14  # def2-svp 14 def2-tzvp 37
    pos = torch.randn(n_atom,3)
    atomic_numbers = torch.randint(low = 0,high=20,size = (n_atom,))

    edge_index = radius_graph(pos, 10,max_num_neighbors=neighbors)
    energy = torch.randn(1)
    forces = torch.randn(n_atom,3)
    diag_hamiltonian = torch.randn((n_atom,atom_max_orbital,atom_max_orbital))
    non_diag_hamiltonian = torch.randn((n_atom*(n_atom-1),atom_max_orbital,atom_max_orbital))
    diag_mask = torch.ones((n_atom,atom_max_orbital,atom_max_orbital))
    non_diag_mask = torch.ones((n_atom*(n_atom-1),atom_max_orbital,atom_max_orbital))

    out.update({
        "molecule_size":torch.Tensor([n_atom]).int(),
        "pos":pos,
        "atomic_numbers":atomic_numbers,
        "edge_index":edge_index,
        "energy":energy,
        "forces":forces,
        "diag_hamiltonian":diag_hamiltonian,
        "non_diag_hamiltonian":non_diag_hamiltonian,
        "diag_mask":diag_mask,
        "non_diag_mask":non_diag_mask
    })
    return out
    # Data(edge_index=[2, 32], pos=[9, 3], interaction_graph=[2, 9], labels=[9], 
    #      num_nodes=9, num_labels=2, label_batch=[2], diag_hamiltonian=[9, 14, 14], 
    #      mask_l1=[32], non_diag_mask=[32, 14, 14], batch=[9], energy=[2], 
    #      diag_mask=[9, 14, 14], atomic_numbers=[9], molecule_size=[2], forces=[9, 3], 
    #      non_diag_hamiltonian=[32, 14, 14])
def construct_o3irrps(dim,order):
    string = []
    for l in range(order+1):
        string.append(f"{dim}x{l}e" if l%2==0 else f"{dim}x{l}o")
    return "+".join(string)

def to_torchgeometric_Data(data:dict):
    torchgeometric_data = Data()
    for key in data.keys():
        torchgeometric_data[key] = data[key]
    return torchgeometric_data

def construct_o3irrps_base(dim,order):
    string = []
    for l in range(order+1):
        string.append(f"{dim}x{l}e")
    return "+".join(string)