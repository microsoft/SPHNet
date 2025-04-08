from tqdm import tqdm
import torch
import os
import lmdb
import pickle  
from torch_geometric.data import Data
from torch_geometric.transforms.radius_graph import RadiusGraph

import sys
from build_label import build_label

sys.path.append('..')
import numpy as np
from utility.pyscf import get_pyscf_obj_from_dataset
from argparse import Namespace
import shutup
shutup.please()

chemical_symbols = ["n", "H", "He" ,"Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", 
            "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
            "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
            "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", 
            "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", 
            "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", 
            "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", 
            "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]

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
        transform_indices.append(np.array(map_idx) + offset)
        transform_signs.append(np.array(map_sign))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    transform_indices = np.concatenate(transform_indices).astype(np.int32)
    transform_signs = np.concatenate(transform_signs)

    hamiltonian_new = hamiltonian[...,transform_indices, :]
    hamiltonian_new = hamiltonian_new[...,:, transform_indices]
    hamiltonian_new = hamiltonian_new * transform_signs[:, None]
    hamiltonian_new = hamiltonian_new * transform_signs[None, :]

    return hamiltonian_new

def cord2xyz(atom_types, atom_cords):
    xyz = ""
    for i in range(len(atom_cords)):
        xyz += f"{atom_types[i]} {' '.join([str(j) for j in atom_cords[i]])}\n"
    return xyz

def cal_initH(Z, R):

    pos = R
    atomic_numbers = Z
    mol, mf,factory = get_pyscf_obj_from_dataset(pos,atomic_numbers, basis='def2-svp', 
                                                    xc='pbe', gpu=False, verbose=1)
    dm0 = mf.init_guess_by_minao()
    init_h = mf.get_fock(dm=dm0)

    return init_h

def create_lmdb(file_path, data):  
    # 创建 LMDB 环境
    env = lmdb.open(file_path, map_size=80 * 1024 * 1024 * 1024)  
  
    with env.begin(write=True) as txn:  
        # 存储数据库长度  
        txn.put("length".encode("ascii"), pickle.dumps(len(data)))  
  
        for idx, data_dict in enumerate(data):  
            key = idx.to_bytes(length=4, byteorder='big')  
            value = pickle.dumps(data_dict)  
            txn.put(key, value)  
  
    env.close()  

name = 'malondialdehyde'
dataset = torch.load(f'/data/datasets/md17/{name}/processed/data.pt')

if name == 'water':
    atoms = [8, 1, 1]
elif name == 'ethanol':
    atoms = [6, 6, 8, 1, 1, 1, 1, 1, 1]
elif name == 'malondialdehyde':
    atoms = [6, 6, 6, 8, 8, 1, 1, 1, 1]
elif name == 'uracil':
    atoms = [6, 6, 7, 6, 7, 6, 8, 8, 1, 1, 1, 1]
elif name == 'aspirin':
    atoms = [6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8,
                1, 1, 1, 1, 1, 1, 1, 1]

data = []
total_mol = dataset[0].energy.shape[0]
atoms_num = len(atoms)
focks = dataset[0].hamiltonian.reshape(total_mol, dataset[0].hamiltonian.shape[-1], dataset[0].hamiltonian.shape[-1])
for i in tqdm(range(total_mol)):
    init_h = cal_initH(atoms,dataset[0].pos[i*atoms_num:(i+1)*atoms_num])
    trans_H = matrix_transform(focks[i], atoms, convention='back2pyscf')

    data_lsr = Data()
    data_lsr.num_nodes = atoms_num
    data_lsr.pos = dataset[0].pos[i*atoms_num:(i+1)*atoms_num]
    neighbor_finder = RadiusGraph(r = 3)
    data_lsr = neighbor_finder(data_lsr)
    min_nodes_foreachGroup = 3
    build_label(data_lsr, num_labels = int(atoms_num/min_nodes_foreachGroup),method = 'kmeans')

    data_dict = {  
        "id":i,
        "pos": np.array(dataset[0].pos[i*atoms_num:(i+1)*atoms_num]), 
        "atoms": np.array(atoms).astype(np.int32),           
        "edge_index": data_lsr['edge_index'], 
        "labels": data_lsr['labels'], 
        'num_nodes':atoms_num,
        "Ham": np.array(trans_H),
        "Ham_init":init_h
    }  
    data.append(data_dict) 

# create mdb file
lmdb_path = f'/data/datasets/md17/{name}/processed'  
if not os.path.exists(lmdb_path):  
    os.makedirs(lmdb_path)  

create_lmdb(lmdb_path, data)  