import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.nn.functional import mse_loss, l1_loss,huber_loss
from collections import defaultdict
from transformers import get_polynomial_decay_schedule_with_warmup

from pytorch_lightning import LightningModule
from ..models.model import create_model
from ..utility.pyscf import get_pyscf_obj_from_dataset, get_homo_lumo_from_h, get_energy_from_h
from ..dataset.buildblock import get_conv_variable_lin,block2matrix
from ..utility.eigen_solver import *
from functools import partial
import torch_geometric.transforms as T
import random
HATREE_TO_KCAL = 627.5096

class FloatCastDatasetWrapper(T.BaseTransform):
    """A transform that casts all floating point tensors to a given dtype.
    tensors to a given dtype.
    """

    def __init__(self, dtype=torch.float64):
        super(FloatCastDatasetWrapper, self).__init__()
        self._dtype = dtype

    def forward(self, data):
        for key, value in data:
            if torch.is_tensor(value) and torch.is_floating_point(value):
                setattr(data, key, value.to(self._dtype))
        return data

class ErrorMetric():
    def __init__(self,loss_weight):
        # if loss_weight == 0:
        #     raise ValueError(f"loss weight is 0, please check your each loss weight")
        pass
    def get_loss_from_diff(self, diff,metric):
        if metric == "mae":
            loss  =  torch.mean(torch.abs(diff))
        elif metric == "ae":
            loss  =  torch.sum(torch.abs(diff))
        elif metric == "mse":
            loss =  torch.mean(diff**2)
        elif metric == "se":
            loss =  torch.sum(diff**2)
        elif metric == "rmse":
            loss  = torch.sqrt(torch.mean(diff**2))
        elif (metric == "maemse") or (metric == "msemae"):
            mae = torch.mean(torch.abs(diff))
            mse = torch.mean(diff**2)
            loss =  mae+mse
        elif (metric == "maemse") or (metric == "msemae"):
            mae = torch.mean(torch.abs(diff))
            mse = torch.mean(diff**2)
            loss =  mae+mse
        elif metric == 'huber':
            loss = huber_loss(diff, 0, reduction="mean", delta=1.0)
        else:
            raise ValueError(f"loss not support metric: {metric}")
        return loss
    
    def cal_loss(self,batch_data,error_dict = {},metric = None):
        pass
    
class EnergyError(ErrorMetric):
    def __init__(self, loss_weight, metric = "mae"):
        super().__init__(loss_weight)
        self.loss_weight = loss_weight      
        self.metric = metric
        self.name = "energy_loss"
        
    def cal_loss(self,batch_data,error_dict = {},metric = None):
        error_dict["loss"] = error_dict.get("loss",0)
        metric = self.metric if metric is None else metric
        diff = batch_data["energy"]-batch_data["pred_energy"]
        loss = self.get_loss_from_diff(diff,metric)

        error_dict["loss"] += self.loss_weight*loss
        error_dict[f"energy_loss_{metric}"] = loss
        return error_dict
        
class ForcesError(ErrorMetric):
    def __init__(self, loss_weight, metric = "mae"):
        super().__init__(loss_weight)

        self.metric = metric
        self.loss_weight = loss_weight
        self.name = "force_loss"
        
    def cal_loss(self,batch_data,error_dict = {},metric = None):
        error_dict["loss"] = error_dict.get("loss",0)
        metric = self.metric if metric is None else metric
        diff = batch_data["forces"]-batch_data["pred_forces"]
        loss = self.get_loss_from_diff(diff,metric)

        error_dict["loss"] += self.loss_weight*loss
        error_dict[f"forces_loss_{metric}"] = loss.detach()
        
class HamiltonianError(ErrorMetric):
    def __init__(self, loss_weight, metric = "mae", sparse = False, sparse_coeff = 1e-5, hami_name = 'HamiHead'):
        super().__init__(loss_weight)
        self.metric = metric
        self.loss_weight = loss_weight
        self.name = "hamiltonian_loss"
        self.sparse = sparse
        self.sparse_coeff = sparse_coeff
        self.symmetry = 'symmetry' in hami_name.lower()


    def cal_loss(self, batch_data,error_dict = {},metric = None):
        error_dict["loss"] = error_dict.get("loss",0)
        metric = self.metric if metric is None else metric
        
        if self.symmetry:
            mask = torch.cat((batch_data['diag_mask'],batch_data['non_diag_mask'],batch_data['non_diag_mask']))
            # predict = torch.cat((predictions['hamiltonian_diagonal_blocks'],predictions['hamiltonian_non_diagonal_blocks']*mask_l1.unsqueeze(-1).unsqueeze(-1)))
            predict = torch.cat((batch_data['pred_hamiltonian_diagonal_blocks'],batch_data['pred_hamiltonian_non_diagonal_blocks'],batch_data['pred_hamiltonian_non_diagonal_blocks']))  # no distance norm
            target = torch.cat((batch_data['diag_hamiltonian'],batch_data['non_diag_hamiltonian'],batch_data['non_diag_hamiltonian'])) #the label is ground truth minus initial guess
        else:
            mask = torch.cat((batch_data['diag_mask'],batch_data['non_diag_mask']))
            # predict = torch.cat((predictions['hamiltonian_diagonal_blocks'],predictions['hamiltonian_non_diagonal_blocks']*mask_l1.unsqueeze(-1).unsqueeze(-1)))
            predict = torch.cat((batch_data['pred_hamiltonian_diagonal_blocks'],batch_data['pred_hamiltonian_non_diagonal_blocks']))  # no distance norm
            target = torch.cat((batch_data['diag_hamiltonian'],batch_data['non_diag_hamiltonian'])) #the label is ground truth minus initial guess
        if self.sparse:
            # target geq to sparse coeff is considered as non-zero
            sparse_mask = torch.abs(target).ge(self.sparse_coeff).float()
            target = target*sparse_mask
        diff = (predict-target)*mask
        
        weight = (mask.numel() / mask.sum())
        if metric == 'multi_head_mae':
            error_dict[f'hami_loss_mae'] = weight * torch.mean(torch.abs(diff)).detach()
            indices_list = batch_data['multi_head_indices']
            weight_list = [1,2,6,20]
            loss = 0
            for i in range(len(indices_list)):
                diff = batch_data['non_diag_hamiltonian'][indices_list[i]] - batch_data['pred_hamiltonian_non_diagonal_blocks'][indices_list[i]]
                mae = weight_list[i] * torch.mean(torch.abs(diff))
                error_dict[f'hami_loss_mae_non_diag_{i}'] = mae.detach()
                loss += mae
            diag_diff = batch_data['diag_hamiltonian'] - batch_data['pred_hamiltonian_diagonal_blocks']
            diag_mae = torch.mean(torch.abs(diag_diff))
            error_dict[f'hami_loss_mae_diag'] = diag_mae.detach()
            loss += diag_mae
        else:
            loss = self.get_loss_from_diff(diff,metric)
        if metric == "rmse":
            loss = loss*weight**0.5
        else:
            loss = loss*weight
        if metric in ["msemae","maemse"]:
            error_dict[f'hami_loss_mae'] = weight*torch.mean(torch.abs(diff.detach()))
            error_dict[f'hami_loss_mse'] = weight*torch.mean((diff.detach())**2)
            
        error_dict['loss']  += loss*self.loss_weight
        error_dict[f'hami_loss_{metric}'] = loss.detach()
        # print(f"==============hami_loss_{metric}, {loss.detach()}")

def build_final_matrix(batch_data, basis, sym=True):
    atom_start = 0
    atom_pair_start = 0
    rebuildfocks = []
    conv,_,mask_lin,_ = get_conv_variable_lin(basis)
    for idx,n_atom in enumerate(batch_data.molecule_size.reshape(-1)):
        n_atom = n_atom.item()
        Z = batch_data.atomic_numbers[atom_start:atom_start+n_atom]
        diag = batch_data.diag_hamiltonian[atom_start:atom_start+n_atom]
        if sym:
            non_diag = batch_data.non_diag_hamiltonian[atom_pair_start:atom_pair_start+n_atom*(n_atom-1)//2]
        else:
            non_diag = batch_data.non_diag_hamiltonian[atom_pair_start:atom_pair_start+n_atom*(n_atom-1)]

        atom_start += n_atom
        atom_pair_start += n_atom*(n_atom-1)//2
        
        rebuildfock = block2matrix(Z,diag,non_diag,mask_lin,conv.max_block_size, sym=sym)
        rebuildfocks.append(rebuildfock)
    # batch_data["pred_hamiltonian"] = rebuildfocks
    return rebuildfocks

class EnergyHamiError(ErrorMetric):
    def __init__(self, loss_weight, trainer = None,metric="mae", 
                    basis="def2-svp", transform_h=False, scaled=False, normalization=False):
        super().__init__(loss_weight)
        
        self.trainer = trainer
        self.metric = metric
        self.loss_weight = loss_weight
        self.name = "energy_hami_loss"
        self.basis = basis
        self.transform_h = transform_h
        self.scaled = scaled
        self.normalization = normalization
        if normalization:
            self.mean_diag = torch.zeros(37, 37)
            self.mean_non_diag = torch.zeros(37, 37)
            self.std_diag = torch.load('std_diag.pt')
            self.std_non_diag = torch.load('std_non_diag.pt')
    
    def _batch_energy_hami(self, batch_data):
        batch_size = batch_data['idx'].shape[0]
        energy = batch_data['energy'] if 'energy' in batch_data.keys() else batch_data['idx']
        if self.normalization:
            batch_data['pred_hamiltonian_diagonal_blocks'] = \
                batch_data['pred_hamiltonian_diagonal_blocks'] * \
                    self.std_diag[None, :, :].to(batch_data['pred_hamiltonian_diagonal_blocks'].device) + \
                    self.mean_diag[None, :, :].to(batch_data['pred_hamiltonian_diagonal_blocks'].device)
            batch_data['pred_hamiltonian_non_diagonal_blocks'] = \
                batch_data['pred_hamiltonian_non_diagonal_blocks'] * \
                self.std_non_diag[None, :, :].to(batch_data['pred_hamiltonian_non_diagonal_blocks'].device) + \
                self.mean_non_diag[None, :, :].to(batch_data['pred_hamiltonian_non_diagonal_blocks'].device)
        elif self.scaled:
            diag_mask, non_diag_mask = batch_data['diag_mask'], batch_data['non_diag_mask']
            diag_target, non_diag_target = batch_data['diag_hamiltonian'], batch_data['non_diag_hamiltonian']

            sample_weight = diag_mask.size(1) * diag_mask.size(2) / diag_mask.sum(axis=(1,2))
            mean_target = diag_target.abs().mean(axis=(1,2)) * sample_weight
            batch_data['pred_hamiltonian_diagonal_blocks'] = batch_data['pred_hamiltonian_diagonal_blocks'] * mean_target[:, None, None]
            sample_weight = non_diag_mask.size(1) * non_diag_mask.size(2) / non_diag_mask.sum(axis=(1,2))
            mean_target = non_diag_target.abs().mean(axis=(1,2)) * sample_weight
            batch_data['pred_hamiltonian_non_diagonal_blocks'] = mean_target[:, None, None] * batch_data['pred_hamiltonian_non_diagonal_blocks']
        
        self.trainer.model.hami_model.build_final_matrix(batch_data) 
        full_hami = batch_data['pred_hamiltonian']
        hami_energy = torch.zeros_like(energy,dtype=torch.float64)
        target_energy = torch.zeros_like(energy,dtype=torch.float64)
        hami_humo_lumo = torch.zeros_like(energy,dtype=torch.float64)
        target_humo_lumo = torch.zeros_like(energy,dtype=torch.float64)
        hami_coeff = torch.zeros_like(energy,dtype=torch.float64)

        target_hami = batch_data["hamiltonian"]

        for i in range(batch_size):
            start , end = batch_data['ptr'][i],batch_data['ptr'][i+1]
            pos = batch_data['pos'][start:end].detach().cpu().numpy()
            atomic_numbers = batch_data['atomic_numbers'][start:end].detach().cpu().numpy()
            mol, mf,factory = get_pyscf_obj_from_dataset(pos,atomic_numbers, basis=self.basis, 
                                                         xc='b3lyp5', gpu=False, verbose=1)
            dm0 = mf.init_guess_by_minao()
            init_h = mf.get_fock(dm=dm0)

            if self.trainer.hparams.remove_init:
                f_hi = full_hami[i].detach().cpu().numpy()+init_h
                f_gti = target_hami[i].detach().cpu().numpy()+init_h
            else:
                f_hi = full_hami[i].detach().cpu().numpy()
                f_gti = target_hami[i].detach().cpu().numpy()

            hami_energy[i] = get_energy_from_h(mf, f_hi)
            target_energy[i] = get_energy_from_h(mf, f_gti)

            hami_humo_lumo[i], hami_mo_coeff, mo_energy_pred = get_homo_lumo_from_h(mf, f_hi)
            target_humo_lumo[i], target_mo_coeff, mo_energy_target = get_homo_lumo_from_h(mf, f_gti)
            target_energy[i] = torch.mean(torch.abs(torch.tensor(mo_energy_pred - mo_energy_target)))

            hami_coeff[i] = torch.cosine_similarity(torch.tensor(hami_mo_coeff), torch.tensor(target_mo_coeff), dim=0).abs().mean()

            if factory is not None:factory.free_resources()

        return hami_energy, target_energy, hami_humo_lumo-target_humo_lumo, hami_coeff
    
    def cal_loss(self, batch_data, error_dict = {}, metric = None):
        metric = self.metric if metric is None else metric
        
        predict, target, humo_lumo_gap_diff, mo_coeff = self._batch_energy_hami(batch_data)
        error_dict['mo_coefficient'] = mo_coeff.mean()
        error_dict['energy_mae'] = torch.mean(target)


class OrbitalEnergyError(ErrorMetric):
    def __init__(self, loss_weight, trainer = None, metric="mae", 
                basis="def2-svp", transform_h=False, ed_type = 'naive', pi_iter = 19, orbital_matrix_gt=False):
        super().__init__(loss_weight)
        self.trainer = trainer
        self.metric = metric
        self.loss_weight = loss_weight
        self.name = "orbital_energy_loss"
        self.basis = basis
        self.transform_h = transform_h
        self.ed_type = ed_type
        self.orbital_matrix_gt = orbital_matrix_gt
        if ed_type == 'naive':
            self.ed_layer = torch.linalg.eigh
        elif ed_type == 'trunc':
            self.ed_layer = ED_trunc.apply
        elif ed_type == 'power_iteration':
            self.ed_layer = partial(ED_PI_Layer, pi_iter=pi_iter)
        else:
            raise NotImplementedError()

        self.loss_type = trainer.hparams.enable_hami_orbital_energy
    
        # >>> A = torch.randn(2, 2, dtype=torch.complex128)
        # >>> A = A + A.T.conj()  # creates a Hermitian matrix
        # >>> A
        # tensor([[2.9228+0.0000j, 0.2029-0.0862j],
        #         [0.2029+0.0862j, 0.3464+0.0000j]], dtype=torch.complex128)
        # >>> L, Q = torch.linalg.eigh(A)
        # >>> L
        # tensor([0.3277, 2.9415], dtype=torch.float64)
        # >>> Q
        # tensor([[-0.0846+-0.0000j, -0.9964+0.0000j],
        #         [ 0.9170+0.3898j, -0.0779-0.0331j]], dtype=torch.complex128)
        # >>> torch.dist(Q @ torch.diag(L.cdouble()) @ Q.T.conj(), A)
        # tensor(6.1062e-16, dtype=torch.float64)

        # >>> A = torch.randn(3, 2, 2, dtype=torch.float64)
        # >>> A = A + A.mT  # creates a batch of symmetric matrices
        # >>> L, Q = torch.linalg.eigh(A)
        # >>> torch.dist(Q @ torch.diag_embed(L) @ Q.mH, A)
    @staticmethod
    def eigen_solver(full_hamiltonian, overlap_matrix, atoms, ed_layer=torch.linalg.eigh, ed_type="naive", eng_threshold = 1e-8):
        eig_success = True
        degenerate_eigenvals = False
        # try:
        # eigvals, eigvecs = torch.linalg.eigh(overlap_matrix)
        n_eigens = overlap_matrix.shape[-1]
        if ed_type == 'power_iteration':
            eigvals, eigvecs = ed_layer(overlap_matrix, n_eigens, reverse=True)
        else:
            eigvals, eigvecs = ed_layer(overlap_matrix)
        eps = eng_threshold * torch.ones_like(eigvals)
        eigvals = torch.where(eigvals > eng_threshold, eigvals, eps)
        frac_overlap = eigvecs / torch.sqrt(eigvals).unsqueeze(-2)

        Fs = torch.bmm(torch.bmm(frac_overlap.transpose(-1, -2), full_hamiltonian), frac_overlap)
        num_orb = sum(atoms) // 2
        # orbital_energies, orbital_coefficients = torch.linalg.eigh(Fs)
        if ed_type == 'power_iteration':
            orbital_energies, orbital_coefficients = ed_layer(Fs, num_orb, reverse=False)
        elif ed_type == 'trunc':
            orbital_energies, orbital_coefficients = ed_layer(Fs, 1, num_orb)
        else:
            orbital_energies, orbital_coefficients = ed_layer(Fs)

        _, counts = torch.unique_consecutive(orbital_energies, return_counts=True)
        if torch.any(counts>1): #will give NaNs in backward pass
            degenerate_eigenvals = True #will give NaNs in backward pass
        orbital_coefficients = torch.bmm(frac_overlap, orbital_coefficients)
        # except RuntimeError: #catch convergence issues with symeig
        #     eig_success = False
        #     orbital_energies = None
        #     orbital_coefficients = None
        
        return eig_success, degenerate_eigenvals, orbital_energies, orbital_coefficients,frac_overlap


    def cal_loss(self, batch_data, error_dict = {}, metric = None):
        self.trainer.model.hami_model.build_final_matrix(batch_data) # construct full_hamiltonian
        if self.loss_type != 33:
            self.cal_homolumo_occupied(batch_data, error_dict, metric)
        if self.loss_type in [1,2,3]:
            return self.cal_loss_winit(batch_data, error_dict, metric)
        elif self.loss_type == 5:
            return self.cal_loss_woinit(batch_data, error_dict, metric)
        elif self.loss_type == 15:
            return self.cal_loss_ws_woinit15(batch_data, error_dict, metric)
        elif self.loss_type in [20,21,22]:
            return self.cal_loss_ws_woinit20(batch_data, error_dict, metric)
    
    def cal_homolumo_occupied(self, batch_data, error_dict = {}, metric = None):
        error_dict["loss"] = error_dict.get("loss",0)
        metric = self.metric if metric is None else metric
        self.metric = metric

        batch_size = batch_data['energy'].shape[0]
        full_hami_pred = batch_data['pred_hamiltonian']
        full_hami = batch_data['hamiltonian']
        error1,error2,error3=[],[],[]
        error4,error5,error6=[],[],[]
        for i in range(batch_size):
            start , end = batch_data['ptr'][i],batch_data['ptr'][i+1]
            atomic_numbers = batch_data['atomic_numbers'][start:end]
            number_of_electrons = (atomic_numbers).sum()
            number_of_occ_orbitals = number_of_electrons // 2

            # get the overlap matrix
            if 's1e' in batch_data:
                overlap_matrix = batch_data['s1e'][i] 
                # transfer overlap_matrix to the device of full_hami
                overlap_matrix = torch.from_numpy(overlap_matrix).to(full_hami_pred[i].device)
            else:
                raise ValueError("overlap matrix is not provided")
            
            # get the initial guess fock matrix
            if 'init_fock'  in batch_data:
                init_fock = batch_data['init_fock'][i]
                # transfer init_fock to the device of full_hami
                init_fock = torch.from_numpy(init_fock).to(full_hami_pred[i].device)
                full_hami_pred_i = (full_hami_pred[i] + init_fock).detach()
                full_hami_i = (full_hami[i] + init_fock).detach()
            else:
                full_hami_pred_i = full_hami_pred[i].detach()
                full_hami_i = full_hami[i].detach()
            norb = full_hami_i.shape[-1]

            # get the ground truth occupied orbital energies and calculate the loss
            symeig_success, degenerate_eigenvalues, orbital_energies, orbital_coefficients,frac_overlap = self.eigen_solver(full_hami_i.unsqueeze(0),
                                                                                                                overlap_matrix.unsqueeze(0),
                                                                                                                atomic_numbers,
                                                                                                                self.ed_layer,
                                                                                                                self.ed_type)
            
            # get the ground truth occupied orbital energies and calculate the loss
            _, _, orbital_energies_pred, orbital_coefficients_pred,_ = self.eigen_solver(full_hami_pred_i.unsqueeze(0),
                                                                                                                overlap_matrix.unsqueeze(0),
                                                                                                                atomic_numbers,
                                                                                                                self.ed_layer,
                                                                                                                self.ed_type)
                
            e = orbital_energies.squeeze(0)
            e_NN = orbital_energies_pred.squeeze(0)
            homo_error = torch.mean(torch.abs(e[number_of_occ_orbitals-1] - e_NN[number_of_occ_orbitals-1]))
            lumo_error = torch.mean(torch.abs(e[number_of_occ_orbitals] - e_NN[number_of_occ_orbitals]))
            gap_error = torch.mean(torch.abs(e[number_of_occ_orbitals-1]-e[number_of_occ_orbitals]-e_NN[number_of_occ_orbitals-1]+e_NN[number_of_occ_orbitals]))
            occ_error =  torch.mean(torch.abs(e[:number_of_occ_orbitals] - e_NN[:number_of_occ_orbitals]))
            orb_error = torch.mean(torch.abs(e - e_NN))
            correlation = torch.cosine_similarity(orbital_coefficients[..., :number_of_occ_orbitals], 
                                            orbital_coefficients_pred[..., :number_of_occ_orbitals], 
                                            dim=1).abs().mean()
            error1.append(homo_error)
            error2.append(lumo_error)
            error3.append(gap_error)
            error4.append(occ_error)
            error5.append(orb_error)
            error6.append(correlation)
        error_dict[f'homo_err_kcalmol'] = HATREE_TO_KCAL*torch.mean(torch.stack(error1))
        error_dict[f'lumo_err_kcalmol'] = HATREE_TO_KCAL*torch.mean(torch.stack(error2))
        error_dict[f'homolumogap_err_kcalmol'] = HATREE_TO_KCAL*torch.mean(torch.stack(error3))
        
        error_dict[f'occ_err_kcalmol'] = HATREE_TO_KCAL*torch.mean(torch.stack(error4))
        error_dict[f'orb_err_kcalmol'] = HATREE_TO_KCAL*torch.mean(torch.stack(error5))
        error_dict[f'cos_sim_eigenvec'] = torch.mean(torch.stack(error6))
        
            
    def cal_loss_ws_woinit20(self, batch_data, error_dict = {}, metric = None):
        error_dict["loss"] = error_dict.get("loss",0)
        metric = self.metric if metric is None else metric
        self.metric = metric

        batch_size = batch_data['energy'].shape[0]
        full_hami_pred = batch_data['pred_hamiltonian']
        full_hami = batch_data['hamiltonian']
        loss_aggregated = []
        loss_aggregated2 = []
        loss_aggregated3 = []
        for i in range(batch_size):
            start , end = batch_data['ptr'][i],batch_data['ptr'][i+1]
            atomic_numbers = batch_data['atomic_numbers'][start:end]
            number_of_electrons = (atomic_numbers).sum()
            number_of_occ_orbitals = number_of_electrons // 2
            # get the initial guess fock matrix
            # get the overlap matrix
            if 's1e' in batch_data:
                overlap_matrix = batch_data['s1e'][i] 
                # transfer overlap_matrix to the device of full_hami
                overlap_matrix = torch.from_numpy(overlap_matrix).to(full_hami_pred[i].device)
            else:
                pos = batch_data['pos'][start:end].detach().cpu().numpy()
                mol, mf,factory = get_pyscf_obj_from_dataset(pos,atomic_numbers, basis=self.basis, gpu=True)
                s1e = mf.get_ovlp()
                overlap_matrix = torch.from_numpy(s1e).float().to(full_hami_pred[i].device)
                # raise ValueError("overlap matrix is not provided")
            
            
            if 'init_fock'  in batch_data:
                init_fock = batch_data['init_fock'][i]
                # transfer init_fock to the device of full_hami
                init_fock = torch.from_numpy(init_fock).to(full_hami_pred[i].device)
                            
                # get the full hamiltonian by adding the initial guess
                full_hami_pred_i = full_hami_pred[i] + init_fock
                full_hami_i = full_hami[i] + init_fock
            else:
                # get the full hamiltonian by adding the initial guess
                full_hami_pred_i = full_hami_pred[i]
                full_hami_i = full_hami[i]
                
            norb = full_hami_i.shape[-1]

            # get the ground truth occupied orbital energies and calculate the loss
            symeig_success, degenerate_eigenvalues, orbital_energies, orbital_coefficients,frac_overlap = self.eigen_solver(full_hami_i.unsqueeze(0),
                                                                                                                overlap_matrix.unsqueeze(0),
                                                                                                                atomic_numbers,
                                                                                                                self.ed_layer,
                                                                                                                self.ed_type)
            


            w1 = 627    
            e_fockdelta = (w1*orbital_coefficients.permute(0,2,1)@full_hami[i].unsqueeze(0)@(orbital_coefficients))
            e_NN_fockdelta = (w1*orbital_coefficients.permute(0,2,1)@full_hami_pred[i].unsqueeze(0)@(orbital_coefficients))
            # weight = 10
            # e_fockdelta[:, number_of_occ_orbitals] *= weight
            # e_NN_fockdelta[:, number_of_occ_orbitals] *= weight

            # e_fockdelta[:, number_of_occ_orbitals, :] *= weight
            # e_fockdelta[:, :, number_of_occ_orbitals] *= weight
            # e_NN_fockdelta[:, number_of_occ_orbitals, :] *= weight
            # e_NN_fockdelta[:, :, number_of_occ_orbitals] *= weight

            # e_fockdelta[:, :, number_of_occ_orbitals-1] *= weight
            # e_fockdelta[:, number_of_occ_orbitals-1, :] *= weight
            # e_NN_fockdelta[:, :, number_of_occ_orbitals-1] *= weight
            # e_NN_fockdelta[:, number_of_occ_orbitals-1, :] *= weight
            
            # get the ground truth occupied orbital energies and calculate the loss
            flag1 = symeig_success and (not degenerate_eigenvalues)
            flag = flag1

            diff = (e_fockdelta-e_NN_fockdelta)[:,torch.eye(norb)==1] if flag else torch.zeros_like(orbital_energies)
            diff2 = (e_fockdelta-e_NN_fockdelta)[:,(1-torch.eye(norb))==1] if flag else torch.zeros_like(orbital_energies)
            if random.random()>0.95 and self.trainer.local_rank==0:     
                print("e_fockdelta energy , pred, diff:",e_fockdelta[:,torch.eye(norb)==1],
                                                        e_NN_fockdelta[:,torch.eye(norb)==1],
                                                        diff)
            loss = self.get_loss_from_diff(diff,metric)
            loss2 = self.get_loss_from_diff(diff2,metric)
            
            

            
            loss_aggregated.append(loss)
            loss_aggregated2.append(loss2)
            eigenvec_loss = self.get_loss_from_diff(
                100*(full_hami[i].unsqueeze(0)@(orbital_coefficients)-full_hami_pred[i].unsqueeze(0)@(orbital_coefficients)),
                metric)
            loss_aggregated3.append(eigenvec_loss)
        
        loss_aggregated = torch.stack(loss_aggregated).mean()
        loss_aggregated2 = torch.stack(loss_aggregated2).mean()
        loss_aggregated3 = torch.stack(loss_aggregated3).mean()
        
        error_dict[f'e_NN_diag_{self.metric}'] = loss.detach()
        error_dict[f'e_NN_nondiag_{self.metric}'] = loss2.detach()
        error_dict[f'e_NN_eigenvec_{self.metric}'] = eigenvec_loss.detach()
        if self.loss_type == 20:
            loss = loss_aggregated+loss_aggregated2
        if self.loss_type == 21:
            loss = loss_aggregated3
        if self.loss_type == 22:
            loss = loss_aggregated+loss_aggregated2 + loss_aggregated3



        error_dict[f'loss'] += loss*self.loss_weight
        return error_dict
    
    def cal_loss_ws_woinit15(self, batch_data, error_dict = {}, metric = None):
        error_dict["loss"] = error_dict.get("loss",0)
        metric = self.metric if metric is None else metric
        self.metric = metric

        batch_size = batch_data['energy'].shape[0]
        full_hami_pred = batch_data['pred_hamiltonian']
        full_hami = batch_data['hamiltonian']
        loss_aggregated = []
        loss_aggregated2 = []
        for i in range(batch_size):
            start , end = batch_data['ptr'][i],batch_data['ptr'][i+1]
            atomic_numbers = batch_data['atomic_numbers'][start:end]
            number_of_electrons = (atomic_numbers).sum()
            number_of_occ_orbitals = number_of_electrons // 2
            # get the initial guess fock matrix
            if 'init_fock'  in batch_data:
                init_fock = batch_data['init_fock'][i]
                # transfer init_fock to the device of full_hami
                init_fock = torch.from_numpy(init_fock).to(full_hami_pred[i].device)
            else:
                init_fock = torch.zeros_like(full_hami_pred[i])
            # get the overlap matrix
            if 's1e' in batch_data:
                overlap_matrix = batch_data['s1e'][i] 
                # transfer overlap_matrix to the device of full_hami
                overlap_matrix = torch.from_numpy(overlap_matrix).to(full_hami_pred[i].device)
            else:
                pos = batch_data['pos'][start:end].detach().cpu().numpy()
                mol, mf,factory = get_pyscf_obj_from_dataset(pos,atomic_numbers, basis=self.basis, gpu=True)
                s1e = mf.get_ovlp()
                overlap_matrix = torch.from_numpy(s1e).float().to(full_hami_pred[i].device)
                # raise ValueError("overlap matrix is not provided")
            # get the full hamiltonian by adding the initial guess
            full_hami_pred_i = full_hami_pred[i] # + init_fock
            full_hami_i = full_hami[i] # + init_fock
            norb = full_hami_i.shape[-1]

            # get the ground truth occupied orbital energies and calculate the loss
            symeig_success, degenerate_eigenvalues, orbital_energies, orbital_coefficients, frac_overlap = self.eigen_solver(full_hami_i.unsqueeze(0),
                                                                                                                overlap_matrix.unsqueeze(0),
                                                                                                                atomic_numbers,
                                                                                                                self.ed_layer,
                                                                                                                self.ed_type)
            

                
            # Fs_NN = torch.bmm(torch.bmm(frac_overlap.transpose(-1, -2), full_hami_pred_i.unsqueeze(0)), frac_overlap)
            e_NN = orbital_coefficients.permute(0,2,1)@full_hami_pred_i.unsqueeze(0)@orbital_coefficients

            
            
            # orbital energy shape is 1*orb, orbital_coefficients shape is 1*orb*orb
            # take only the occupied orbitals
            orbital_energies = orbital_energies #[:,:number_of_occ_orbitals]
            orbital_energies_pred = e_NN[:,torch.eye(norb)==1] #[:,:number_of_occ_orbitals]

            # get the ground truth occupied orbital energies and calculate the loss
            flag1 = symeig_success and (not degenerate_eigenvalues)
            flag = flag1

            diff = orbital_energies - orbital_energies_pred if flag else torch.zeros_like(orbital_energies)
            diff2 = e_NN[:,(1-torch.eye(norb))==1]
            if random.random()>0.95 and self.trainer.local_rank==0:     
                print("orb energy , pred, diff:",orbital_energies,orbital_energies_pred,diff)
            loss = self.get_loss_from_diff(diff,metric)
            loss2 = self.get_loss_from_diff(diff2,metric)

            loss_aggregated.append(loss)
            loss_aggregated2.append(loss2)
            
        loss_aggregated = torch.stack(loss_aggregated).mean()
        loss_aggregated2 = torch.stack(loss_aggregated2).mean()
        error_dict[f'e_NN_diag_{self.metric}'] = loss.detach()
        error_dict[f'e_NN_nondiag_{self.metric}'] = loss2.detach()

        loss = loss_aggregated+loss_aggregated2
        error_dict[f'loss'] += loss*self.loss_weight
        return error_dict
    

    def cal_loss_winit(self, batch_data, error_dict = {}, metric = None):
        error_dict["loss"] = error_dict.get("loss",0)
        metric = self.metric if metric is None else metric
        self.metric = metric

        batch_size = batch_data['energy'].shape[0]
        full_hami_pred = batch_data['pred_hamiltonian']
        full_hami = batch_data['hamiltonian']
        loss_aggregated = []
        loss_aggregated2 = []
        for i in range(batch_size):
            start , end = batch_data['ptr'][i],batch_data['ptr'][i+1]
            atomic_numbers = batch_data['atomic_numbers'][start:end]
            number_of_electrons = (atomic_numbers).sum()
            number_of_occ_orbitals = number_of_electrons // 2
            # get the initial guess fock matrix
            if 'init_fock'  in batch_data:
                init_fock = batch_data['init_fock'][i]
                # transfer init_fock to the device of full_hami
                init_fock = torch.from_numpy(init_fock).to(full_hami_pred[i].device)
            else:
                init_fock = torch.zeros_like(full_hami_pred[i])
            # get the overlap matrix
            if 's1e' in batch_data:
                overlap_matrix = batch_data['s1e'][i] 
                # transfer overlap_matrix to the device of full_hami
                overlap_matrix = torch.from_numpy(overlap_matrix).to(full_hami_pred[i].device)
            else:
                pos = batch_data['pos'][start:end].detach().cpu().numpy()
                mol, mf,factory = get_pyscf_obj_from_dataset(pos,atomic_numbers, basis=self.basis, gpu=True)
                s1e = mf.get_ovlp()
                overlap_matrix = torch.from_numpy(s1e).float().to(full_hami_pred[i].device)
                # raise ValueError("overlap matrix is not provided")
            # get the full hamiltonian by adding the initial guess
            full_hami_pred_i = full_hami_pred[i]  + init_fock
            full_hami_i = full_hami[i]  + init_fock
            norb = full_hami_i.shape[-1]

            # get the ground truth occupied orbital energies and calculate the loss
            symeig_success, degenerate_eigenvalues, orbital_energies, orbital_coefficients,frac_overlap = self.eigen_solver(full_hami_i.unsqueeze(0),
                                                                                                                overlap_matrix.unsqueeze(0),
                                                                                                                atomic_numbers,
                                                                                                                self.ed_layer,
                                                                                                                self.ed_type)
            

                
            # Fs_NN = torch.bmm(torch.bmm(frac_overlap.transpose(-1, -2), full_hami_pred_i.unsqueeze(0)), frac_overlap)
            e_NN = orbital_coefficients.permute(0,2,1)@full_hami_pred_i.unsqueeze(0)@orbital_coefficients

            # get the ground truth occupied orbital energies and calculate the loss
            flag1 = symeig_success and (not degenerate_eigenvalues)
            flag = flag1
            orbital_energies_pred,diff,diff2,loss,loss2 = None,None,None,None,None,None
            if self.loss_type == 1:
                # orbital energy shape is 1*orb, orbital_coefficients shape is 1*orb*orb
                # take only the occupied orbitals
                orbital_energies = orbital_energies #[:,:number_of_occ_orbitals]
                orbital_energies_pred = e_NN[:,torch.eye(norb)==1] #[:,:number_of_occ_orbitals]

                diff = orbital_energies - orbital_energies_pred if flag else torch.zeros_like(orbital_energies)
                diff2 = e_NN[:,(1-torch.eye(norb))==1]
                loss = self.get_loss_from_diff(diff,metric)
                loss2 = self.get_loss_from_diff(diff2,metric)
            elif self.loss_type == 2:
                # orbital energy shape is 1*orb, orbital_coefficients shape is 1*orb*orb
                # take only the occupied orbitals
                orbital_energies = orbital_energies
                orbital_energies_pred = orbital_energies_pred[:,torch.eye(norb)==1]


                diff = (orbital_energies - orbital_energies_pred[:,torch.eye(norb)==1])[:,:number_of_occ_orbitals] if flag else torch.zeros_like(orbital_energies)
                diff2 = e_NN[:,:number_of_occ_orbitals,:number_of_occ_orbitals][:,(1-torch.eye(number_of_occ_orbitals))==1]
                loss = self.get_loss_from_diff(diff,metric)
                loss2 = self.get_loss_from_diff(diff2,metric)
            elif self.loss_type == 3:
                # orbital energy shape is 1*orb, orbital_coefficients shape is 1*orb*orb
                # take only the occupied orbitals
                orbital_energies = orbital_energies
                orbital_energies_pred = e_NN[:,torch.eye(norb)==1]


                diff = orbital_energies - orbital_energies_pred if flag else torch.zeros_like(orbital_energies)

                loss = 0.1*self.get_loss_from_diff(diff,metric) + \
                        0.9*self.get_loss_from_diff(diff[:,:number_of_occ_orbitals],metric)
                loss2 = 0.1*self.get_loss_from_diff(e_NN[:,(1-torch.eye(norb))==1],metric) + \
                        0.9*self.get_loss_from_diff(e_NN[:,:number_of_occ_orbitals,:number_of_occ_orbitals][:,(1-torch.eye(number_of_occ_orbitals))==1],metric)

                
            if random.random()>0.95 and self.trainer.local_rank==0:     
                print("orb energy , pred, diff:",orbital_energies,orbital_energies_pred,diff)


            loss_aggregated.append(loss)
            loss_aggregated2.append(loss2)
            
        loss_aggregated = torch.stack(loss_aggregated).mean()
        loss_aggregated2 = torch.stack(loss_aggregated2).mean()
        error_dict[f'e_NN_diag_{self.metric}'] = loss.detach()
        error_dict[f'e_NN_nondiag_{self.metric}'] = loss2.detach()

        loss = loss_aggregated+loss_aggregated2
        error_dict[f'loss'] += loss*self.loss_weight
        return error_dict
    

    def cal_loss_woinit(self, batch_data, error_dict = {}, metric = None):
        error_dict["loss"] = error_dict.get("loss",0)
        metric = self.metric if metric is None else metric
        self.metric = metric

        batch_size = batch_data['energy'].shape[0]
        full_hami_pred = batch_data['pred_hamiltonian']
        full_hami = batch_data['hamiltonian']
        loss_aggregated = []
        loss_aggregated2 = []
        for i in range(batch_size):
            start , end = batch_data['ptr'][i],batch_data['ptr'][i+1]
            atomic_numbers = batch_data['atomic_numbers'][start:end]
            number_of_electrons = (atomic_numbers).sum()
            number_of_occ_orbitals = number_of_electrons // 2
            
            # get the full hamiltonian by adding the initial guess
            full_hami_pred_i = full_hami_pred[i] #/HATREE_TO_KCAL # + init_fock/HATREE_TO_KCAL
            full_hami_i = full_hami[i] #/HATREE_TO_KCAL # + init_fock/HATREE_TO_KCAL
            norb = full_hami_i.shape[-1]

            
            if self.ed_type == 'power_iteration':
                orbital_energies, orbital_coefficients = self.ed_layer(full_hami_i.unsqueeze(0), norb, reverse=False)
            elif self.ed_type == 'trunc':
                orbital_energies, orbital_coefficients = self.ed_layer(full_hami_i.unsqueeze(0), 1, norb)
            else:
                orbital_energies, orbital_coefficients = self.ed_layer(full_hami_i.unsqueeze(0))
                
                
            # Fs_NN = torch.bmm(torch.bmm(frac_overlap.transpose(-1, -2), full_hami_pred_i.unsqueeze(0)), frac_overlap)
            e_NN = orbital_coefficients.permute(0,2,1)@full_hami_pred_i.unsqueeze(0)@orbital_coefficients

            
            
            # orbital energy shape is 1*orb, orbital_coefficients shape is 1*orb*orb
            # take only the occupied orbitals
            orbital_energies = orbital_energies #[:,:number_of_occ_orbitals]
            orbital_energies_pred = e_NN[:,torch.eye(norb)==1] #[:,:number_of_occ_orbitals]

            # get the ground truth occupied orbital energies and calculate the loss
            flag1 = True #symeig_success and (not degenerate_eigenvalues)
            flag = flag1

            diff = orbital_energies - orbital_energies_pred if flag else torch.zeros_like(orbital_energies)
            diff2 = e_NN[:,(1-torch.eye(norb))==1]
            if random.random()>0.95 and self.trainer.local_rank==0:     
                print("orb energy , pred, diff:",orbital_energies,orbital_energies_pred,diff)
            loss = self.get_loss_from_diff(diff,metric)
            loss2 = self.get_loss_from_diff(diff2,metric)

            loss_aggregated.append(loss)
            loss_aggregated2.append(loss2)
            
        loss_aggregated = torch.stack(loss_aggregated).mean()
        loss_aggregated2 = torch.stack(loss_aggregated2).mean()
        error_dict[f'e_NN_diag_{self.metric}'] = loss.detach()
        error_dict[f'e_NN_nondiag_{self.metric}'] = loss2.detach()

        loss = loss_aggregated+loss_aggregated2
        error_dict[f'loss'] += loss*self.loss_weight
        return error_dict

class OrbitalEnergyErrorV2(ErrorMetric):
    def __init__(self, loss_weight, trainer = None, metric="mae", basis="def2-svp", transform_h=False, ed_type = 'naive', pi_iter = 19):
        super().__init__(loss_weight)
        self.trainer = trainer
        self.metric = metric
        self.loss_weight = loss_weight
        self.name = "orbital_energy_loss"
        self.basis = basis
        self.transform_h = transform_h
        self.ed_type = ed_type
        if ed_type == 'naive':
            self.ed_layer = torch.linalg.eigh
        elif ed_type == 'trunc':
            self.ed_layer = ED_trunc.apply
        elif ed_type == 'power_iteration':
            self.ed_layer = partial(ED_PI_Layer, pi_iter=pi_iter)
        else:
            raise NotImplementedError()
    

    def eigen_solver(self, full_hamiltonian, overlap_matrix, atoms, ed_layer, ed_type, eng_threshold = 1e-8):
        eig_success = True
        degenerate_eigenvals = False
        try:
            # eigvals, eigvecs = torch.linalg.eigh(overlap_matrix)
            n_eigens = overlap_matrix.shape[-1]
            if ed_type == 'power_iteration':
                eigvals, eigvecs = ed_layer(overlap_matrix, n_eigens, reverse=True)
            else:
                eigvals, eigvecs = ed_layer(overlap_matrix)
            eps = eng_threshold * torch.ones_like(eigvals)
            eigvals = torch.where(eigvals > eng_threshold, eigvals, eps)
            frac_overlap = eigvecs / torch.sqrt(eigvals).unsqueeze(-2)

            Fs = torch.bmm(torch.bmm(frac_overlap.transpose(-1, -2), full_hamiltonian), frac_overlap)
            num_orb = sum(atoms) // 2
            # orbital_energies, orbital_coefficients = torch.linalg.eigh(Fs)
            if ed_type == 'power_iteration':
                orbital_energies, orbital_coefficients = ed_layer(Fs, num_orb, reverse=False)
            elif ed_type == 'trunc':
                orbital_energies, orbital_coefficients = ed_layer(Fs, 1, num_orb)
            else:
                orbital_energies, orbital_coefficients = ed_layer(Fs)

            _, counts = torch.unique_consecutive(orbital_energies, return_counts=True)
            if torch.any(counts>1): #will give NaNs in backward pass
                degenerate_eigenvals = True #will give NaNs in backward pass
            orbital_coefficients = torch.bmm(frac_overlap, orbital_coefficients)
        except RuntimeError: #catch convergence issues with symeig
            eig_success = False
            orbital_energies = None
            orbital_coefficients = None
        
        return eig_success, degenerate_eigenvals, orbital_energies, orbital_coefficients



    def _batch_orbital_energy_hami_loss(self, batch_data):
        batch_size = batch_data['energy'].shape[0]
        self.trainer.model.hami_model.build_final_matrix(batch_data) # construct full_hamiltonian
        full_hami_pred = batch_data['pred_hamiltonian']
        full_hami = batch_data['hamiltonian']
        loss_aggregated = []
        for i in range(batch_size):
            start , end = batch_data['ptr'][i],batch_data['ptr'][i+1]
            atomic_numbers = batch_data['atomic_numbers'][start:end]
            number_of_electrons = (atomic_numbers).sum()
            number_of_occ_orbitals = number_of_electrons // 2
            # get the initial guess fock matrix
            if 'init_fock'  in batch_data:
                init_fock = batch_data['init_fock'][i]
                # transfer init_fock to the device of full_hami
                init_fock = torch.from_numpy(init_fock).to(full_hami_pred[i].device)
            # get the overlap matrix
            if 's1e' in batch_data:
                overlap_matrix = batch_data['s1e'][i] / HATREE_TO_KCAL
                # transfer overlap_matrix to the device of full_hami
                overlap_matrix = torch.from_numpy(overlap_matrix).to(full_hami_pred[i].device)
            else:
                raise ValueError("overlap matrix is not provided")
            # get the full hamiltonian by adding the initial guess
            full_hami_pred_i = full_hami_pred[i]/HATREE_TO_KCAL + init_fock/HATREE_TO_KCAL
            full_hami_i = full_hami[i]/HATREE_TO_KCAL + init_fock/HATREE_TO_KCAL
            # solve the FC = SCe problem
            
            # get the ground truth occupied orbital energies and calculate the loss
            symeig_success, degenerate_eigenvalues, orbital_energies, orbital_coefficients = self.eigen_solver(full_hami_i.unsqueeze(0),
                                                                                                                overlap_matrix.unsqueeze(0),
                                                                                                                atomic_numbers,
                                                                                                                self.ed_layer,
                                                                                                                self.ed_type)
            pred = torch.einsum('bji, jk, bkl -> bil ', orbital_coefficients, full_hami_pred_i, orbital_coefficients)
            # create a diagonal matrix with the orbital energies
            diag_orbital_energies = torch.diag_embed(orbital_energies)
            # get the ground truth occupied orbital energies and calculate the loss
            flag = symeig_success and (not degenerate_eigenvalues)
            diff = pred - diag_orbital_energies if flag else torch.zeros_like(diag_orbital_energies)
            loss = self.get_loss_from_diff(diff)
            loss_aggregated.append(loss)
            
        loss_aggregated = torch.stack(loss_aggregated).mean()

        return loss_aggregated
    

    def get_loss_from_diff(self, diff):
        metric = self.metric 
        if metric == "mae":
            loss  =  torch.mean(torch.abs(diff))
        elif metric == "mse":
            loss =  torch.mean(diff**2)
        elif metric == "rmse":
            loss  = torch.sqrt(torch.mean(diff**2))
        elif (metric == "maemse") or (metric == "msemae"):
            mae = torch.mean(torch.abs(diff))
            mse = torch.mean(diff**2)
            loss =  mae+mse
        else:
            raise ValueError(f"loss not support metric: {metric}")
        return loss

    def cal_loss(self, batch_data, error_dict = {}, metric = None):
        error_dict["loss"] = error_dict.get("loss",0)
        metric = self.metric if metric is None else metric
        loss = self._batch_orbital_energy_hami_loss(batch_data)
        error_dict[f'loss'] += loss*self.loss_weight
        error_dict[f'orbital_energy_hami_loss_{metric}'] = loss.detach()
        return error_dict


class HomoLumoHamiError(ErrorMetric):
    def __init__(self, loss_weight, metric="mae", basis="def2-svp", transform_h=False):
        super().__init__(loss_weight)

        self.metric = metric
        self.loss_weight = loss_weight
        self.name = "homo_lumo_hami_loss"
        self.basis = basis
        self.transform_h = transform_h

    
class LNNP(LightningModule):
    def __init__(self, hparams, mean=None, std=None):
        super(LNNP, self).__init__()
        self.save_hyperparameters(hparams)

        self.model = create_model(self.hparams, mean, std)
        self.enable_energy = self.hparams.enable_energy
        self.enable_forces = self.hparams.enable_forces
        self.enable_hami = self.hparams.enable_hami
        self.enable_hami_orbital_energy = self.hparams.enable_hami_orbital_energy
        
        self.construct_loss_func_list()
        self._reset_losses_dict()

        dtype_mapping = {16: torch.float16, 32: torch.float, 64: torch.float64}
        self.data_transform = FloatCastDatasetWrapper(
            dtype_mapping[int(self.hparams.precision)]
        )

    def construct_loss_func_list(self,):
        self.loss_func_list_train = []
        if self.enable_energy:
            self.loss_func_list_train.append(EnergyError(self.hparams.energy_weight,self.hparams.energy_train_loss))
        if self.enable_forces:
            self.loss_func_list_train.append(ForcesError(self.hparams.forces_weight,self.hparams.forces_train_loss))
        if self.enable_hami:
            self.loss_func_list_train.append(HamiltonianError(self.hparams.hami_weight,self.hparams.hami_train_loss, self.hparams.sparse_loss, self.hparams.sparse_loss_coeff, self.hparams.hami_model.name))
        if self.enable_hami_orbital_energy:
            self.loss_func_list_train.append(OrbitalEnergyError(self.hparams.orbital_energy_weight,
                 self, self.hparams.orbital_energy_train_loss, self.hparams.basis, ed_type=self.hparams.ed_type))        
        
        self.loss_func_list_val = []
        if self.enable_energy:
            self.loss_func_list_val.append(EnergyError(self.hparams.energy_weight,self.hparams.energy_val_loss))
        if self.enable_forces:
            self.loss_func_list_val.append(ForcesError(self.hparams.forces_weight,self.hparams.forces_val_loss))
        if self.enable_hami:
            self.loss_func_list_val.append(HamiltonianError(self.hparams.hami_weight,self.hparams.hami_val_loss, self.hparams.hami_model.name))
        if self.enable_hami_orbital_energy:
            self.loss_func_list_val.append(OrbitalEnergyError(self.hparams.orbital_energy_weight,
                 self, self.hparams.orbital_energy_train_loss, self.hparams.basis, ed_type=self.hparams.ed_type))        
        
        # some real world / application level evaluation.
        # a little time consuming, thus, in data module, only 1 batch data is used.
        self.loss_func_list_val_realworld = [] #self.loss_func_list_val[:]#
        if self.enable_hami and self.hparams.enable_energy_hami_error:
            self.loss_func_list_val_realworld.append(EnergyHamiError(1,
                                                                     self,
                                                                self.hparams.energy_val_loss, 
                                                                self.hparams.basis, 
                                                                "qh9" in self.hparams.data_name.lower(),
                                                                self.hparams.hami_train_loss=="scaled",
                                                                self.hparams.hami_train_loss== "normalization"))


        self.loss_func_list_test = self.loss_func_list_val[:]
        if self.enable_hami and self.hparams.enable_energy_hami_error:
            self.loss_func_list_test.append(EnergyHamiError(1,
                                                            self,
                                                            self.hparams.energy_val_loss, 
                                                            self.hparams.basis, 
                                                            "qh9" in self.hparams.data_name.lower(),
                                                            self.hparams.hami_train_loss=="scaled",
                                                            self.hparams.hami_train_loss== "normalization"))
            # if self.hparams.test_homo_lumo_hami:
            #     self.loss_func_list_test.append(HomoLumoHamiError(self.hparams.energy_weight,
            #                                                       self.hparams.energy_val_loss, 
            #                                                       self.hparams.basis, 
            #                                                       "qh9" in self.hparams.data_name.lower()))
    

    def _reset_losses_dict(self,):
        self.losses = {"train":defaultdict(list),
                        "val":defaultdict(list),
                        "test":defaultdict(list)}
        
    def configure_optimizers(self):
        if not self.hparams.multi_para_group: 
            params = self.model.parameters()
        else:
            other_params = []
            pretrained_params = []
            hami_head = []
            hami_head_0 = []
            hami_head_1 = []
            hami_head_2 = []
            hami_head_3 = []
            for (name, param) in self.model.named_parameters():
                # load pretrain is not in key
                if self.hparams.model.load_pretrain != '':
                    if 'node_attr_encoder' in name: # in so2 model the node_attr_encoder is likely to be pretrained
                        pretrained_params.append(param)
                # elif 'LSRM_module' in name:
                #     pretrained_params.append(param)
                # elif 'e3_gnn_node_pair_layer' in name:
                #     pretrained_params.append(param)
                elif 'hami_model' in name:
                    if ('e3_gnn_node_pair_layer' in name) or ('fc_ij' in name) or ('expand_ij' in name):
                        if '_1.' in name:
                            hami_head_1.append(param)
                        elif '_2.' in name:
                            hami_head_2.append(param)
                        elif '_3.' in name:
                            hami_head_3.append(param)
                        else:
                            hami_head_0.append(param)
                    else:
                        hami_head.append(param)
                else:
                    other_params.append(param)
            params = [
                {'params': other_params},
                {'params': pretrained_params, 'lr': self.hparams.lr*0.5},
                {'params': hami_head, 'lr': self.hparams.lr*5},
                # {'params': hami_head_0, 'lr': self.hparams.lr*10 * 0.8},
                # {'params': hami_head_1, 'lr': self.hparams.lr*10 * 0.5},
                # {'params': hami_head_2, 'lr': self.hparams.lr*10 * 1.2},
                # {'params': hami_head_3, 'lr': self.hparams.lr*10 * 40},
                {'params': hami_head_0, 'lr': self.hparams.lr*5},
                {'params': hami_head_1, 'lr': self.hparams.lr*5},
                {'params': hami_head_2, 'lr': self.hparams.lr*5},
                {'params': hami_head_3, 'lr': self.hparams.lr*5},
            ]
        optimizer = AdamW(
            params,
            lr=self.hparams.lr,
            betas = (0.99,0.999),
            weight_decay=self.hparams.weight_decay,
            amsgrad=False
        )
        
        schedule_cfg = self.hparams["schedule"]
        #warm up is set in optimizer_step
        if schedule_cfg.lr_schedule == 'cosine':
            scheduler = CosineAnnealingLR(optimizer,  schedule_cfg.lr_cosine_length)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        elif schedule_cfg.lr_schedule == 'polynomial':
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=-1, 
                num_training_steps= self.hparams.max_steps,
                lr_end =  schedule_cfg.lr_min, power = 1.0, last_epoch = -1)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        elif schedule_cfg.lr_schedule == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                "min",
                factor= schedule_cfg.lr_factor,
                patience= schedule_cfg.lr_patience,
                min_lr= schedule_cfg.lr_min,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        else:
            raise ValueError(f"Unknown lr_schedule: {schedule_cfg.lr_schedule}")
        
        return [optimizer], [lr_scheduler]
    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        lr_warmup_steps = self.hparams["schedule"]["lr_warmup_steps"]
        if self.trainer.global_step < lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(lr_warmup_steps),
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        # # 在更新之前保存权重  
        # original_weights = {name: param.data.clone() for name, param in self.named_parameters() if param.requires_grad}  

        super().optimizer_step(*args, **kwargs)

        # # 计算更新Δw，并计算Δw和原始权重W之间的比例  
        # updates_ratios = {}  
        # for name, param in self.named_parameters():  
        #     if param.requires_grad:  
        #         # Δw  
        #         update = param.data - original_weights[name]  
        #         # 避免除以零，添加一个小的epsilon  
        #         epsilon = 1e-10  
        #         # 计算比例 |Δw| / (|W| + epsilon)  
        #         ratio = torch.abs(update) / (torch.abs(original_weights[name]) + epsilon)  
        #         updates_ratios[name] = ratio  

        optimizer.zero_grad()
        
    def forward(self,  batch_data):
        return self.model(batch_data)

    def training_step(self, batch_data, batch_idx):
        return self.step(batch_data,  "train", self.loss_func_list_train)

    def validation_step(self, batch_data, batch_idx,dataloader_idx=0):
        # validation step
        if dataloader_idx == 0:
            return self.step(batch_data, "val", self.loss_func_list_val)
        else:
            if self.loss_func_list_val_realworld:
                return self.step(batch_data, "val", self.loss_func_list_val_realworld)
        
        # # test step
        # return self.step(batch, l1_loss, "test")

    def test_step(self, batch_data, batch_idx):
        return self.step(batch_data, "test", self.loss_func_list_test)


    def step(self, batch_data, stage, loss_func_list=[]):
        batch_data = self.data_transform(batch_data)
        with torch.set_grad_enabled(stage == "train" or self.enable_forces):
            # TODO: the model doesn't necessarily need to return a derivative once
            # Union typing works under TorchScript (https://github.com/pytorch/pytorch/pull/53180)
            # fock, pred, noise_pred, deriv = self(batch.z, batch.pos, batch.batch)
            batch_data = self(batch_data)
        
        loss_func_list = self.loss_func_list_train if loss_func_list is [] else loss_func_list
        error_dict = {"loss":0}
        for loss_func in loss_func_list:
            loss_func.cal_loss(batch_data,error_dict)
            
        for key in error_dict:
            if key=='loss' and type(error_dict[key])==int:
                continue
            self.losses[stage][key].append(error_dict[key].detach())


        # Frequent per-batch logging for training
        if stage == 'train':
            train_metrics = {f"train_per_step/{k}": v for k, v in error_dict.items()}
            train_metrics['learningrate'] = self.trainer.optimizers[0].param_groups[0]["lr"]
            train_metrics['step'] = self.trainer.global_step 

            self.trainer.progress_bar_metrics["lr"] = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.trainer.progress_bar_metrics["loss"] = error_dict["loss"].detach().item()
            
            # print(train_metrics['lr_per_step'])
            # train_metrics['batch_pos_mean'] = batch_data.pos.mean().item()
            self.log_dict(train_metrics, sync_dist=True)
            # if  train_metrics['step']%10 == 0:
            # print(train_metrics)
        return error_dict["loss"]



    def on_train_epoch_end(self):
        dm = self.trainer.datamodule
        # if hasattr(dm, "test_dataset") and len(dm.test_dataset) > 0:
        #     should_reset = (
        #         self.current_epoch % self.hparams.test_interval == 0
        #         or (self.current_epoch - 1) % self.hparams.test_interval == 0
        #     )
        #     if should_reset:
        #         # reset validation dataloaders before and after testing epoch, which is faster
        #         # than skipping test validation steps by returning None
        #         self.trainer.reset_val_dataloader(self)

    # TODO(shehzaidi): clean up this function, redundant logging if dy loss exists.
    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            # construct dict of logged metrics
            result_dict = {}

            for stage in ["train","val","test"]:
                for key in self.losses[stage]:
                    if stage == "val" and key == "loss":
                        result_dict["val_loss"] = torch.stack(self.losses[stage][key]).mean()                
                    result_dict[f"{stage}/{key}"] = torch.stack(self.losses[stage][key]).mean()
            self.log_dict(result_dict, sync_dist=True)
            print(result_dict)
        self._reset_losses_dict()
        
    def on_test_epoch_end(self):
        if not self.trainer.sanity_checking:
            # construct dict of logged metrics
            result_dict = {}

            for stage in ["train","val","test"]:
                for key in self.losses[stage]:
                    if stage == "val" and key == "loss":
                        result_dict["val_loss"] = torch.stack(self.losses[stage][key]).mean()                
                    else:
                        result_dict[f"{stage}/{key}"] = torch.stack(self.losses[stage][key]).mean()
            self.log_dict(result_dict, sync_dist=True)
            print(result_dict)
        self._reset_losses_dict()
