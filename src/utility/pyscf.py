from pyscf import gto, scf, dft
import numpy as np
import torch
from argparse import Namespace
HATREE_TO_KCAL = 627.5096

convention_dict = {
    'pyscf_631G': Namespace(
        atom_to_orbitals_map={1: 'ss', 6: 'ssspp', 7: 'ssspp', 8: 'ssspp', 9: 'ssspp'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd':
            [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1], 6: [0, 1, 2, 3, 4], 7: [0, 1, 2, 3, 4],
            8: [0, 1, 2, 3, 4], 9: [0, 1, 2, 3, 4]
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

    hamiltonian_new = hamiltonian[..., transform_indices, :]
    hamiltonian_new = hamiltonian_new[..., :, transform_indices]
    hamiltonian_new = hamiltonian_new * transform_signs[:, None]
    hamiltonian_new = hamiltonian_new * transform_signs[None, :]

    return hamiltonian_new

def transform_h_into_pyscf(hamiltonian: np.ndarray, mol: gto.Mole)-> np.ndarray:
    """
    Transforms the given Hamiltonian matrix into the PySCF format based on the atomic orbital (AO) type order.

    Args:
        hamiltonian (np.ndarray): The input Hamiltonian matrix.
        mol (gto.Mole): The PySCF Mole object representing the molecular system.

    Returns:
        np.ndarray: The transformed Hamiltonian matrix in the PySCF format.
    """
    # Get ao type list, format [atom_idx, atom_type, ao_type]
    ao_type_list = mol.ao_labels()
    order_list = []
    for idx, labels in enumerate(ao_type_list):
        _, _, ao_type = labels.split(' ')[:3]
        # for p orbitals, the order is px, py, pz, which means the order should transform 
        # from [0, 1, 2], to [2, 0, 1], thus [+2, -1, -1]
        if 'px' in ao_type:
            order_list.append(idx+2)
        elif 'py' in ao_type:
            order_list.append(idx-1)
        elif 'pz' in ao_type:
            order_list.append(idx-1)
        else:
            order_list.append(idx)
       
    # Transform hamiltonian
    hamiltonian_pyscf = hamiltonian[..., order_list, :]
    hamiltonian_pyscf = hamiltonian_pyscf[..., :, order_list]

    return hamiltonian_pyscf

    
def get_pyscf_obj_from_dataset(pos,atomic_numbers,  basis: str="def2-svp",unit=None, xc: str="b3lyp5", gpu=False,verbose=1):
    """
    Get the PySCF Mole and KS objects from a dataset.

    Args:
        data (dict): The dataset containing the molecular data.
        idx (int): The index of the molecular data to retrieve.
        basis (str, optional): The basis set to use. Defaults to "def2-svp".
        xc (str, optional): The exchange-correlation functional to use. Defaults to "b3lyp5".
        gpu (bool, optional): Whether to use GPU acceleration. Defaults to False.

    Returns:
        tuple: A tuple containing the PySCF Mole and KS objects.

    """

    mol = gto.Mole()
    mol.atom = ''.join([f"{atomic_numbers[i]} {pos[i][0]} {pos[i][1]} {pos[i][2]}\n" for i in range(len(atomic_numbers))])
    mol.basis = basis
    mol.verbose = verbose
    mol.build(unit=unit)
    # mf = dft.KS(mol, xc=xc)
    mf = scf.RKS(mol)
    mf.grids.level=3
    mf.max_cycle=20
    mf.direct_scf_tol=1e-10 # previous is 1e-12
    mf.conv_tol_grad=3.16e-5
    mf.conv_tol=1e-8   # 1e-13
    mf.xc = "b3lyp5"
    factory = None
    if gpu:
        try:
            from madft.cuda_factory import CUDAFactory
            from madft.cuda_factory import CUDAParams
            params = CUDAParams()
            params.gpu_id_list = list(range(int(gpu)))
            factory = CUDAFactory()
            factory.params = params
            mf = factory.generate_cuda_instance(mf)
            return mol, mf, factory
        except:
            print("CUDA is not available, falling back to CPU")
            aaa=111
    return mol, mf, factory

def get_psycf_obj_from_xyz(file_name: str, basis: str='def2-svp', xc: str='b3lyp5', gpu=False):
    """
    Create a PySCF Mole and DFT object from an XYZ file.

    Args:
        file_name (str): The path to the XYZ file.
        basis (str, optional): The basis set to use. Defaults to 'def2-svp'.
        xc (str, optional): The exchange-correlation functional to use. Defaults to 'b3lyp5'.
        gpu (bool, optional): Whether to use GPU acceleration. Defaults to False.

    Returns:
        tuple: A tuple containing the PySCF Mole object and the DFT object.
    """
    with open(file_name, 'r') as f:
        lines = f.readlines()
    
    num_atoms = int(lines[0])
    charge, multiplicity = map(int, lines[1].split())
    
    atom_lines = lines[2:2+num_atoms]
    atom_info = [line.split() for line in atom_lines]
    
    mol = gto.Mole()
    mol.atom = ''.join([f"{info[0]} {info[1]} {info[2]} {info[3]}\n" for info in atom_info])
    mol.basis = basis
    mol.verbose = 4
    mol.charge = charge
    mol.spin = multiplicity - 1
    mol.build()
    mf = dft.KS(mol, xc=xc)
    if gpu:
        try:
            from madft.cuda_factory import CUDAFactory
            factory = CUDAFactory()
            mf = factory.generate_cuda_instance(mf)
        except:
            print("CUDA is not available, falling back to CPU")
    return mol, mf

class SCFCallback:
    def __init__(self,scfiter_log):
        self.iter_count = 0
        self.scfiter_log = scfiter_log

    def __call__(self, envs):
        self.iter_count += 1
        if self.scfiter_log:
            print(f"pyscf cycle is : {envs['cycle']}, e is: {envs['e_tot']}")

    def count(self):
        return self.iter_count

def fock_to_dm(mf: scf.RHF, fock: np.ndarray, s1e: np.ndarray = None):
    if s1e is None:
        s1e = mf.get_ovlp()
    mo_energy, mo_coeff = mf.eig(fock, s1e)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    return dm
        
def run_scf_fromh(mf: scf.RHF, h: np.ndarray,conv_tol=1e-5,scfiter_log = False):
    """
    Calculates the energy for a given mean-field object and model.

    Args:
        mf (scf.RHF): The mean-field object.
        model (torch.nn.Module optional): The model used for prediction.
        data (dict, optional): Additional data for the model. Defaults to None.

    Returns:
        (float, int): The energy and the iteration step.
    """
    dm= fock_to_dm(mf, h)
    mf.callback = SCFCallback(scfiter_log)
    mf.conv_tol = conv_tol
    mf.kernel(dm0 = dm) 
    return mf.e_tot, mf.callback.count()

def get_energy_from_h(mf: scf.RHF, h: np.ndarray):
    """
    Calculates the energy for a given mean-field object and Hamiltonian matrix.

    Args:
        mf (scf.RHF): The mean-field object.
        h (np.ndarray): The Hamiltonian matrix.

    Returns:
        float: The energy.
    """
    dm = fock_to_dm(mf, h)
    e_tot = mf.energy_tot(dm=dm)
    return e_tot


def get_homo_lumo_from_h(mf: scf.RHF, h: np.ndarray, s1e: np.ndarray=None):
    if s1e is None:
        s1e = mf.get_ovlp()
    mo_energy, mo_coeff = mf.eig(h, s1e)
    e_idx = np.argsort(mo_energy)
    e_sort = mo_energy[e_idx]
    nocc = mf.mol.nelectron // 2
    homo, lumo = e_sort[nocc-1], e_sort[nocc]
    return homo-lumo, mo_coeff[:, :nocc], mo_energy[:nocc]



# TODO: 
# 1. Init Model class from arguments and them load model from checkout point
# 2. Add energy check in the test step.

# Add test for energy and homo-lumo

