'''
This file is an example script about how to make your dataset suitable for SPHNet model.
This script will finally output a mdb format file.
You can modify the data reading part according to your data format.
'''



import torch
import os
import os.path as osp
import lmdb
import pickle
import argparse
import sys
import numpy as np
import requests
import tarfile

from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.transforms.radius_graph import RadiusGraph

from build_label import build_label
sys.path.append('src')
from utility.pyscf import get_pyscf_obj_from_dataset
from argparse import Namespace
from apsw import Connection
import gdown
import random
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip, Data)

BOHR2ANG = 1.8897259886

GoogleDriveLink = 'https://drive.google.com/drive/u/0/folders/1LXTC8uaOQzmb76FsuGfwSocAbK5Hshfj'


sys.path.append('..')

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

class QH9Stable(InMemoryDataset):
    url = 'https://drive.google.com/file/d/1LcEJGhB8VUGkuyb0oQ_9ANJdSkky9xMS/view?usp=sharing'
    def __init__(self, root='datasets/', split='random', transform=None, pre_transform=None, pre_filter=None):
        self.folder = osp.join(root, 'QH9Stable')
        self.split = split
        self.db_dir = os.path.join(self.folder, 'processed')
        self.full_orbitals = 14
        self.orbital_mask = {}
        idx_1s_2s_2p = torch.tensor([0, 1, 3, 4, 5])
        orbital_mask_line1 = idx_1s_2s_2p
        orbital_mask_line2 = torch.arange(self.full_orbitals)
        for i in range(1, 11):
            self.orbital_mask[i] = orbital_mask_line1 if i <= 2 else orbital_mask_line2

        super(QH9Stable, self).__init__(self.folder, transform, pre_transform, pre_filter)
        self.train_mask, self.val_mask, self.test_mask = torch.load(self.processed_paths[0])
        self.slices = {
            'id': torch.arange(self.train_mask.shape[0] + self.val_mask.shape[0] + self.test_mask.shape[0] + 1)}

    def download(self):
        try:
            print(f"Downloading the QH9Stable dataset to through {self.url}")
            gdown.download(self.url, output=self.raw_paths[0], fuzzy=True)
        except:
            print(f"Downloading failed! Please download the QH9Stable dataset to {self.raw_paths[0]} through {self.url}")
            print(f"Or you can try to download the zip file through {GoogleDriveLink}")
            raise FileNotFoundError(f"QH9Stable needs to be downloaded.")

    @property
    def raw_file_names(self):
        return [f'QH9Stable.db']

    @property
    def processed_file_names(self):
        if self.split == 'random':
            return ['processed_QH9Stable_random_12.pt', 'QH9Stable.lmdb/data.mdb']
        elif self.split == 'size_ood':
            return ['processed_QH9Stable_size_ood.pt', 'QH9Stable.lmdb/data.mdb']

    def process(self):
        if self.split == 'size_ood':
            num_nodes_list = []
        
        for raw_file_name in self.raw_file_names:
            connection = Connection(os.path.join(self.root, "raw", raw_file_name))
            cursor = connection.cursor()
            data = cursor.execute("select * from data").fetchall()
            if not os.path.isdir(os.path.join(self.processed_dir, 'QH9Stable.lmdb')):
                # dataloader with lmdb
                db_env = lmdb.open(os.path.join(self.processed_dir, 'QH9Stable.lmdb'), map_size=1048576000000)
                for row in tqdm(data):
                    with db_env.begin(write=True) as txn:
                        ori_data_dict = {
                            'id': row[0],
                            'num_nodes': row[1],
                            'atoms': row[2],
                            'pos': row[3], # ang
                            'Ham': row[4]
                        }
                        data_dict = pickle.dumps(ori_data_dict)
                        txn.put(ori_data_dict['id'].to_bytes(length=4, byteorder='big'), data_dict)
                db_env.close()
                print('Saving lmdb database...')
            else:
                print("lmdb database exists. Jump the lmdb database creation step.")
            
        if self.split == 'random':
            print('Random splitting...')
            data_ratio = [0.8, 0.1, 0.1]
            data_split = [int(len(data) * data_ratio[0]), int(len(data) * data_ratio[1])]
            data_split.append(len(data) - sum(data_split))
            indices = np.random.RandomState(seed=43).permutation(len(data))
            train_mask = indices[:data_split[0]]
            val_mask = indices[data_split[0]:data_split[0] + data_split[1]]
            test_mask = indices[data_split[0] + data_split[1]:]

        elif self.split == 'size_ood':
            print('Size OOD splitting...')
            num_nodes_list = [row[1] for row in data]
            num_nodes_array = np.array(num_nodes_list)
            train_indices = np.where(num_nodes_array <= 20)
            val_condition = np.logical_and(num_nodes_array >= 21, num_nodes_array <= 22)
            val_indices = np.where(val_condition)
            test_indices = np.where(num_nodes_array >= 23)
            train_mask = train_indices[0].astype(np.int64)
            val_mask = val_indices[0].astype(np.int64)
            test_mask = test_indices[0].astype(np.int64)

        torch.save((train_mask, val_mask, test_mask), self.processed_paths[0])

    def get_mol(self, atoms, pos, Ham):
        data = Data(
            pos=torch.tensor(pos, dtype=torch.float64),
            atoms=torch.tensor(atoms, dtype=torch.int64).view(-1, 1),
            Ham=torch.tensor(Ham, dtype=torch.float64)
        )
        return data

    def get(self, idx):
        db_env = lmdb.open(os.path.join(self.processed_dir, 'QH9Stable.lmdb'), readonly=True, lock=False)
        with db_env.begin() as txn:
            data_dict = txn.get(int(idx).to_bytes(length=4, byteorder='big'))
            data_dict = pickle.loads(data_dict)
            _, num_nodes, atoms, pos, Ham = \
                data_dict['id'], data_dict['num_nodes'], \
                np.frombuffer(data_dict['atoms'], np.int32), \
                np.frombuffer(data_dict['pos'], np.float64), \
                np.frombuffer(data_dict['Ham'], np.float64)
            pos = pos.reshape(num_nodes, 3)
            num_orbitals = sum([5 if atom <= 2 else 14 for atom in atoms])
            Ham = Ham.reshape(num_orbitals, num_orbitals)
            data = self.get_mol(atoms, pos, Ham)
        db_env.close()
        return data

class QH9Dynamic(InMemoryDataset):
    url = {
        '100k': 'https://drive.google.com/file/d/1SNWk0GD6Nt96qNAJJU2uedwWDQ4bbB1w/view?usp=sharing',
        '300k': 'https://drive.google.com/file/d/1f3SOZ1ite5mDbvhybkTk0dWUmwevKUxg/view?usp=sharing'
    }
    def __init__(self, root='datasets/', task='', split='geometry', version='300k', transform=None, pre_transform=None,
                 pre_filter=None):
        assert task in ['']
        self.version = version
        if self.version == '300k':
            self.folder = osp.join(root, 'QH9Dynamic_300k')
        elif self.version == '100k':
            self.folder = osp.join(root, 'QH9Dynamic_100k')
            print("It is recommended to use the 300k version which contains longer MD trajectories.")
        else:
            print(f"Current version parameter is {version}, which is not included in [100k, 300k].")
            print(f"Using 300k version instead...")
            self.folder = osp.join(root, 'QH9Dynamic_300k')
            self.version = '300k'

        self.split = split
        self.db_dir = os.path.join(self.folder, 'processed')
        self.full_orbitals = 14
        self.orbital_mask = {}
        idx_1s_2s_2p = torch.tensor([0, 1, 3, 4, 5])
        orbital_mask_line1 = idx_1s_2s_2p
        orbital_mask_line2 = torch.arange(self.full_orbitals)
        for i in range(1, 11):
            self.orbital_mask[i] = orbital_mask_line1 if i <= 2 else orbital_mask_line2

        super(QH9Dynamic, self).__init__(self.folder, transform, pre_transform, pre_filter)
        self.train_mask, self.val_mask, self.test_mask \
            = torch.load(self.processed_paths[0])
        self.slices = {
            'id': torch.arange(self.train_mask.shape[0] + self.val_mask.shape[0] + self.test_mask.shape[0] + 1)}


    def download(self):
        try:
            print(f"Downloading the QH9Dynamic_{self.version} dataset through {self.url}")
            gdown.download(self.url[self.version], output=self.raw_paths[0], fuzzy=True)
        except:
            print(f"Downloading failed! Please download the QH9Dynamic_{self.version} dataset to {self.raw_paths[0]} "
                  f"through {self.url[self.version]}")
            print(f"Or you can try to download the zip file through {GoogleDriveLink}")
            raise FileNotFoundError(f"QH9Dynamic_{self.version} needs to be downloaded.")


    @property
    def raw_file_names(self):
        if self.version == '300k':
            return [f'QH9Dynamic_300k.db']
        elif self.version == '100k':
            return [f'QH9Dynamic_100k.db']

    @property
    def processed_file_names(self):
        if self.split == 'geometry':
            return ['processed_QH9Dynamic_geometry.pt', 'QH9Dynamic.lmdb/data.mdb']
        elif self.split == 'mol':
            return ['processed_QH9Dynamic_mol.pt', 'QH9Dynamic.lmdb/data.mdb']

    def process(self, num_geometry_per_mol=100, num_train_geometry_per_mol=80, num_val_geometry_per_mol=10, num_mol=2998):
        if self.version == '300k':
            num_mol = 2998
        elif self.version == '100k':
            num_mol = 999
            
        if not os.path.isdir(os.path.join(self.processed_dir, 'QH9Dynamic.lmdb')):
            SAVE_TO_LMDB = True
            print("lmdb database is created.")
        else:
            SAVE_TO_LMDB = False
            print("lmdb database exists. Jump the lmdb database creation step.")
        
        db_env = lmdb.open(os.path.join(self.processed_dir, 'QH9Dynamic.lmdb'), map_size=1048576000000)
        if self.split == 'geometry':
            print('Geometry-wise splitting...')
            train_mask = np.array([], dtype=np.int64)
            val_mask = np.array([], dtype=np.int64)
            test_mask = np.array([], dtype=np.int64)
            cur_index = 0
            for raw_file_name in self.raw_file_names:
                connection = Connection(os.path.join(self.root, "raw", raw_file_name))
                cursor = connection.cursor()
                data = cursor.execute("select * from data").fetchall()
                for ind, row in enumerate(tqdm(data)):
                    # dataloader with lmdb
                    if SAVE_TO_LMDB:
                        with db_env.begin(write=True) as txn:
                            ori_data_dict = {
                                'id': int(np.frombuffer(row[0], dtype=np.int64)),
                                'geo_id': row[1],
                                'num_nodes': row[2],
                                'atoms': row[3],
                                'pos': row[4], # Bhor, and then transfer to ang
                                'Ham': row[9]
                            }
                            data_dict = pickle.dumps(ori_data_dict)
                            txn.put(ind.to_bytes(length=4, byteorder='big'), data_dict)
                

                    if (ind + 1) % num_geometry_per_mol == 0:  # Finish traversing one molecule
                        indices = np.random.RandomState(seed=ind).permutation(
                            num_geometry_per_mol)  # Different random split for different molecules
                        train_mask_cur_mol = cur_index + indices[:num_train_geometry_per_mol]
                        val_mask_cur_mol = cur_index + indices[
                                                       num_train_geometry_per_mol:num_train_geometry_per_mol + num_val_geometry_per_mol]
                        test_mask_cur_mol = cur_index + indices[num_train_geometry_per_mol + num_val_geometry_per_mol:]
                        train_mask = np.concatenate((train_mask, train_mask_cur_mol))
                        val_mask = np.concatenate((val_mask, val_mask_cur_mol))
                        test_mask = np.concatenate((test_mask, test_mask_cur_mol))
                        cur_index += num_geometry_per_mol

        elif self.split == 'mol':
            print('Molecule-wise splitting...')
            for raw_file_name in self.raw_file_names:
                connection = Connection(os.path.join(self.root, "raw", raw_file_name))
                cursor = connection.cursor()
                data = cursor.execute("select * from data").fetchall()
                if SAVE_TO_LMDB:
                    for ind, row in enumerate(tqdm(data)):
                        # dataloader with lmdb
                        with db_env.begin(write=True) as txn:
                            ori_data_dict = {
                                'id': int(np.frombuffer(row[0], dtype=np.int64)),
                                'geo_id': row[1],
                                'num_nodes': row[2],
                                'atoms': row[3],
                                'pos': row[4], # Bhor, and then transfer to ang
                                'Ham': row[9]
                            }
                            data_dict = pickle.dumps(ori_data_dict)
                            txn.put(ind.to_bytes(length=4, byteorder='big'), data_dict)

            if SAVE_TO_LMDB:
                db_env.close()

            mol_id_list = [i for i in range(num_mol) for _ in range(num_geometry_per_mol)]
            data_ratio = [0.8, 0.1, 0.1]
            index_list = [i for i in range(max(mol_id_list) + 1)]
            random.seed(43)
            random.shuffle(index_list)
            train_mol_ids = np.array(index_list[:int(len(index_list) * data_ratio[0])])
            val_mol_ids = np.array(index_list[
                                   int(len(index_list) * data_ratio[0]):int(len(index_list) * data_ratio[0]) + int(
                                       len(index_list) * data_ratio[1])])
            test_mol_ids = np.array(
                index_list[int(len(index_list) * data_ratio[0]) + int(len(index_list) * data_ratio[1]):])
            mol_id_array = np.array(mol_id_list)
            train_mask = np.where(np.isin(mol_id_array, train_mol_ids))[0].astype(np.int64)
            val_mask = np.where(np.isin(mol_id_array, val_mol_ids))[0].astype(np.int64)
            test_mask = np.where(np.isin(mol_id_array, test_mol_ids))[0].astype(np.int64)
        print('Saving...')

        torch.save((train_mask, val_mask, test_mask), self.processed_paths[0])

    def get_mol(self, atoms, pos, Ham):
        data = Data(
            pos=torch.tensor(pos, dtype=torch.float64),
            atoms=torch.tensor(atoms, dtype=torch.int64).view(-1, 1),
            Ham=torch.tensor(Ham, dtype=torch.float64)
        )
        return data

    def get(self, idx):
        db_env = lmdb.open(os.path.join(self.processed_dir, 'QH9Dynamic.lmdb'), readonly=True, lock=False)
        with db_env.begin() as txn:
            data_dict = txn.get(int(idx).to_bytes(length=4, byteorder='big'))
            data_dict = pickle.loads(data_dict)
            _, num_nodes, atoms, pos, Ham =\
                data_dict['id'], data_dict['num_nodes'], \
                np.frombuffer(data_dict['atoms'], np.int32), \
                np.frombuffer(data_dict['pos'], np.float64), \
                np.frombuffer(data_dict['Ham'], np.float64)
            pos = pos.reshape(num_nodes, 3)
            pos = pos / BOHR2ANG  # transfer the unit back to ang
            # pos = pos
            num_orbitals = sum([5 if atom <= 2 else 14 for atom in atoms])
            Ham = Ham.reshape(num_orbitals, num_orbitals)
            data = self.get_mol(atoms, pos, Ham)
        db_env.close()
        return data

def matrix_transform(hamiltonian, atoms, convention='pyscf_def2svp'):
    '''
    The order of orbital in the Hamiltonian matrix calculated by different methods are different. 
    This function is to transform the orbital order.
    The supported transform pairs are list in above convention_dict.
    '''
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
    '''
    This function calculate the initial guess of Hamiltonian matrix. This should be very quick.
    '''
    pos = R
    atomic_numbers = Z
    mol, mf, factory = get_pyscf_obj_from_dataset(pos,atomic_numbers, basis='def2-svp', 
                                                    xc='pbe', gpu=False, verbose=1)
    dm0 = mf.init_guess_by_minao()
    init_h = mf.get_fock(dm=dm0)

    return init_h

def create_lmdb(file_path, data):  
    # create LMDB environment
    env = lmdb.open(file_path, map_size=80 * 1024 * 1024 * 1024)  
  
    with env.begin(write=True) as txn:   
        txn.put("length".encode("ascii"), pickle.dumps(len(data)))  
  
        for idx, data_dict in enumerate(data):  
            key = idx.to_bytes(length=4, byteorder='big')  
            value = pickle.dumps(data_dict)  
            txn.put(key, value)  
  
    env.close()  

def main():
    parser = argparse.ArgumentParser(description="Arguments for loading dataset")

    # Add command-line arguments
    parser.add_argument("--data_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the generated LMDB file")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the original dataset file")

    # Parse arguments
    args = parser.parse_args()

    # Use the arguments
    print(f"Dataset name: {args.data_name}")
    print(f"LMDB path: {args.output_path}")
    print(f"Dataset path: {args.input_path}")

    # download dataset
    if args.data_name.split('_')[1] == 'stable' and args.data_name.split('_')[2] == 'random':
        dataset = QH9Stable(root=args.input_path, split='random')
    elif args.data_name.split('_')[1] == 'stable' and args.data_name.split('_')[2] == 'ood':
        dataset = QH9Stable(root=args.input_path, split='size_ood')
    elif args.data_name.split('_')[1] == 'dynamic' and args.data_name.split('_')[2] == 'geometry':
        dataset = QH9Dynamic(root=args.input_path, split='geometry', version='100k')
    elif args.data_name.split('_')[1] == 'dynamic' and args.data_name.split('_')[2] == 'mol':
        dataset = QH9Dynamic(root=args.input_path, split='mol', version='100k')

    # preprocess dataset
    data = []
    for i in tqdm(range(len(dataset))):

        atoms = dataset[i].atoms.squeeze(-1).numpy().astype(np.int32)
        atoms_num = atoms.shape[0]
        # calculate the initial guess and transform the Hamiltonian matrix
        init_h = cal_initH(atoms,dataset[i].pos)

        # calculate the short range and long range edge index
        data_lsr = Data()
        data_lsr.num_nodes = atoms_num
        data_lsr.pos = dataset[i].pos
        neighbor_finder = RadiusGraph(r = 3)
        data_lsr = neighbor_finder(data_lsr)
        min_nodes_foreachGroup = 3
        build_label(data_lsr, num_labels = int(atoms_num/min_nodes_foreachGroup),method = 'kmeans')

        data_dict = {
            "id": i,
            "pos": np.array(dataset[i].pos),
            "atoms": atoms,
            "edge_index": data_lsr['edge_index'], 
            "labels": data_lsr['labels'], 
            'num_nodes': atoms_num,
            "Ham": np.array(dataset[i].Ham),
            "Ham_init": init_h
        }
        data.append(data_dict)

    # create mdb file
    output_path = args.output_path  
    if not os.path.exists(output_path):  
        os.makedirs(output_path)  

    create_lmdb(output_path, data)

if __name__ == "__main__":
    main() 
