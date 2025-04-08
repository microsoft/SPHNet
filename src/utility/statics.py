import torch
# import shutil
from torch_scatter import scatter
from tqdm import tqdm, trange

from urllib import request as request

import torch
import torch
import numpy as np
from torch.utils.data import DataLoader
# 1 hartree to eV = 27.2114 


class StatisticsAccumulator:
    def __init__(self, batch=False, atomistic=False):
        """
        Use the incremental Welford algorithm described in [1]_ to accumulate
        the mean and standard deviation over a set of samples.

        Args:
            batch: If set to true, assumes sample is batch and uses leading
                   dimension as batch size
            atomistic: If set to true, average over atom dimension

        References:
        -----------
        .. [1] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

        """
        # Initialize state variables
        self.count = 0  # Sample count
        self.mean = 0  # Incremental average
        self.M2 = 0  # Sum of squares of differences
        self.batch = batch
        self.atomistic = atomistic

    def add_sample(self, sample_value):
        """
        Add a sample to the accumulator and update running estimators.
        Differentiates between different types of samples.

        Args:
            sample_value (torch.Tensor): data sample
        """

        # Check different cases
        if not self.batch and not self.atomistic:
            self._add_sample(sample_value)
        elif not self.batch and self.atomistic:
            n_atoms = sample_value.size(0)
            for i in range(n_atoms):
                self._add_sample(sample_value[i].reshape(-1))
        elif self.batch and not self.atomistic:
            n_batch = sample_value.size(0)
            for i in range(n_batch):
                self._add_sample(sample_value[i].reshape(-1))
        else:
            n_batch = sample_value.shape[0]
            n_atoms = sample_value.shape[1]
            for i in range(n_batch):
                for j in range(n_atoms):
                    self._add_sample(sample_value[i, j].reshape(-1))

    def _add_sample(self, sample_value):
        # https://blog.csdn.net/u014485485/article/details/77679669
        # Update count
        self.count += 1
        delta_old = sample_value - self.mean
        # Difference to old mean
        self.mean += delta_old / self.count
        # Update mean estimate
        delta_new = sample_value - self.mean
        # Update sum of differences
        self.M2 += delta_old * delta_new
    
    
    def get_statistics(self):
        """
        Compute statistics of all data collected by the accumulator.

        Returns:
            torch.Tensor: Mean of data
            torch.Tensor: Standard deviation of data
        """
        # Compute standard deviation from M2
        mean = self.mean
        stddev = torch.sqrt(self.M2 / self.count)

        return mean, stddev

    def get_mean(self):
        return self.mean

    def get_stddev(self):
        return torch.sqrt(self.M2 / self.count)

def get_statistics(dataset,loader, prop_name, prop_divide_by_atoms, atomref=None):
    """
    Compute mean and variance of a property. Uses the incremental Welford
    algorithm implemented in StatisticsAccumulator

    Args:
        prop_name (str):  gather/compile statistic for given property. 
        prop_divide_by_atoms (True or False): divide by the number of atoms if True.
        prop_atomref: Use reference values for single atoms.
                                        e.g. COOH   Energy(COOH)-reference(C)-2*reference(O)-reference(H)

    Returns:
        mean: Mean value
        stddev: Standard deviation

    """
    
    # divide_by_atoms = dataset.divide_by_atoms
    # atomref = dataset.atomref
    # print('When statistic, use_atomref dict: ',use_atomref)
    
    ## just for statistic
    if loader is None:
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    statistics = StatisticsAccumulator(batch=True)
    if atomref is not None and isinstance(atomref,np.ndarray):
        atomref = torch.from_numpy(atomref).float()
        
    with torch.no_grad():
        print("statistics will be calculated...")
        for data in tqdm(loader, total = len(loader)):
            property_value = data[prop_name]
            
            # use atom as reference value!
            if atomref is not None:
                z = data["atomic_numbers"]
                p0 = atomref[z].reshape(-1,1)
                p0 = scatter(p0, data['batch'], dim=0).reshape(-1,1)
                property_value -= p0
            if prop_divide_by_atoms:
                try:
                    mask = torch.sum(data["_atom_mask"], dim=1, keepdim=True).view(
                        [-1, 1] + [1] * (property_value.dim() - 2)
                    )
                    property_value /= mask
                except:
                    counter = torch.ones_like(data['batch'])
                    mask = scatter(counter, data['batch'], dim=0)
                    # if mask.dim() == 1:
                    #     mask = mask.unsqueeze(1)
                    property_value = property_value.reshape(-1)/mask.reshape(-1)
                    # property_value = property_value.unsqueeze(1)

                
                
            statistics.add_sample(property_value)


    return statistics.get_mean(), statistics.get_stddev()






def get_atomref(dataset, prop_name, data_len = None, atomic_number_max = 60):
    '''
    prop_name: "energy"
    '''
    data_len = len(dataset) if data_len is None else data_len
    r = torch.zeros(data_len, atomic_number_max+1)
    energy = torch.zeros(data_len)
    for i in trange(data_len):
        data = dataset[i]
        counter = scatter(torch.ones(data.atomic_numbers.shape[0]),data.atomic_numbers.reshape(-1),dim = 0)
        energy[i] = data[prop_name].item()
        r[i,:len(counter)] = counter

    mask = torch.sum(r, dim = 0) > 0
    r = r[:, mask]
    legal_atomref = torch.linalg.lstsq(r, energy).solution
    print('in this dataset, available legal atomic numer is {}. its atomref is {}'.format(torch.where(mask)[0],legal_atomref))
    atomref = torch.zeros(atomic_number_max+1)
    atomref[mask] = legal_atomref
    return atomref