
import torch
import numpy as np
from abc import ABC
from torch_scatter import scatter
from torch_geometric.data import Data
from torch.utils.data import DataLoader,Dataset
from torch.utils.data import Dataset

from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from typing import Any, TypeVar
from .buildblock import *


def remap_values(remapping, x):
    index = torch.bucketize(x.ravel(), remapping[0])
    return remapping[1][index].reshape(x.shape)


def mapping_function(to_be_remapped, indice, remap_source, remap_target, remap_batch, max_num_nodes):
    r'''
    This function perform remapping.
    to_be_remapped: The edge index to be remapped.
    indice: The indicator variable indicating which batch the edge belongs to.
    remap_source: The domain of the mapping function
    remap_target: The range (induced value) of the mapping function
    remap_batch: The batch of the mapping function
    max_num_nodes: The maximum number of nodes in the dataset.
    ######################################################################################################################
    Below is an illustration of the mapping function.
    remap_source + remap_batch -> remap target (Predefined mapping function)
    to_be_remapped + to_be_remapped_indices -> mapped_target (Mapping function application)
    ######################################################################################################################
    '''

    hash_1 =  indice * (max_num_nodes + 1) + to_be_remapped
    hash_2 = remap_batch * (max_num_nodes + 1) + remap_source
    remapping = hash_2, remap_target
    return remap_values(remapping, hash_1)





T_co = TypeVar("T_co", covariant=True)

class DatasetWrapper(Dataset[T_co], ABC):
    def __getattr__(self, attr: str) -> Any:
        """Forward all other attributes to the underlying dataset."""
        return getattr(self._dataset, attr)


class InMemoryDataset(DatasetWrapper[T_co]):
    """Map-style PyTorch Dataset that caches items into CPU memory at initialization.

    This is thread-safe and can be used with multi-process data loading."""

    def __init__(
        self, dataset: Dataset[T_co],indices = None, pin_memory: bool = False, num_loading_threads: int = 4
    ):
        self._dataset = dataset
        self._pin_memory = pin_memory

        self.indices = range(len(dataset)) if indices is None else indices
        self._stored_data: List[T_co] = [None for _ in range(len(self.indices))]  # type: ignore
        with ThreadPool(processes=num_loading_threads) as pool:
            for _data in tqdm(
                pool.imap_unordered(self._load_item, range(len(self.indices))),
                total=len(self.indices),
                desc="Loading dataset into RAM",
                disable=None,  # Only shows in interactive environments.
            ):
                pass

        assert all(d is not None for d in self._stored_data)

    def _load_item(self, idx: int):
        # print(idx)
        data = self._dataset[self.indices[idx]]
        if self._pin_memory:
            data = torch.utils.data._utils.pin_memory.pin_memory(data)
        self._stored_data[idx] = data
        return data

    def __getitem__(self, idx: int) -> T_co:
        """Return a copy of the cached item."""
        example = self._stored_data[idx]
        return example

    def __len__(self) -> int:
        return len(self.indices)
        # return len(self._stored_data)

    # def __add__(self, other: Dataset[T_co]):
    #     return ConcatDataset([self, InMemoryDataset(other)])


def shard_discretizations(
    all_indexes:list[int],
    num_shards: int,
    shard_idx: int,
) -> list:

    assert (
        shard_idx < num_shards
    ), f"Shard index {shard_idx} is larger than the number of shards {num_shards}."
    assert num_shards > 0, f"Number of shards must be positive, got {num_shards}."

    # create a new copy of discretizations, sorted according to cost.
    # sorted_disc = sorted(discretizations, key=cost_fn)
    sharded_disc = all_indexes[shard_idx::num_shards]
    return sharded_disc



def collate_fn_unified(long_cutoff_upper = 9, unit = 1):

    '''
    collate function for pytorch geometric data objects
    Data object should have the following keys():
    num_nodes: number of nodes in the graph
    num_labels: number of labels in the graph
    x: Node features torch.Tensor of shape (num_nodes, num_features)
    pos: Node features torch.Tensor of shape (num_nodes, 3)
    y: output labels torch.Tensor of shape (num_targets)
    force: output labels torch.Tensor of shape (num_nodes * 3)
    edge_index: torch.LongTensor of shape (2, num_edges). This graph should be neighborhood expanded.
    grouping_graph: torch.LongTensor of shape (2, num_edges). This graph is used to group the nodes. 
        Intergroup connection are not allowed. Intragroup connection are constructed b/w every possible nodes (aka. Complete).
    interaction_graph: torch.LongTensor of shape (2, num_edges). This graph is used to construct the interaction graph of size num_nodes * num_groups.
    label: Labelling of the nodes for each group.
    '''
    def _collate_fn(list_of_data):
        processed = Data()
        bs_mol = len(list_of_data)
        for key in list_of_data[0].keys():
            if key in ['pos',"atomic_numbers"]:
                processed[key] = torch.cat([torch.tensor(list_of_data[i][key]) for i in range(bs_mol)],dim = 0)
            elif key in ['diag_hamiltonian','non_diag_hamiltonian','diag_hamiltonian_init','non_diag_hamiltonian_init']:
                processed[key] = torch.cat([torch.tensor(list_of_data[i][key]) for i in range(bs_mol)],dim = 0)
                processed[key] = processed[key]/unit
            elif key in['diag_mask','non_diag_mask',
                    "mask_l1"]:
                processed[key] = torch.cat([torch.tensor(list_of_data[i][key]) for i in range(bs_mol)],dim = 0)
            elif key in ['energy','forces','pyscf_energy']:
                processed[key] = torch.cat([torch.tensor(list_of_data[i][key]) for i in range(bs_mol)],dim = 0)
                processed[key] =  processed[key]/unit
                if key == "energy" or key == 'pyscf_energy':
                    processed[key].reshape(-1,1)
                else:
                    processed[key].reshape(-1,3)
            elif key in ['init_fock','h1e']:
                processed[key] = [list_of_data[i][key] for i in range(bs_mol)]
            elif key == "fock":
                H_block = []
                for i in range(bs_mol):
                    H_block.append(matrixtoblock_lin(list_of_data[i]["fock"],
                                        list_of_data[i]["atomic_numbers"],
                                        list_of_data[i]["buildblock_mask"],
                                        list_of_data[i]["max_block_size"]))
                processed.update({"diag_hamiltonian": torch.cat([torch.tensor(H_block[i][0]) for i in range(bs_mol)],dim = 0)/unit ,
                                "non_diag_hamiltonian":torch.cat([torch.tensor(H_block[i][1]) for i in range(bs_mol)],dim = 0)/unit  ,
                                "diag_mask":torch.cat([torch.tensor(H_block[i][2]) for i in range(bs_mol)],dim = 0)  ,
                                "non_diag_mask": torch.cat([torch.tensor(H_block[i][3]) for i in range(bs_mol)],dim = 0) ,})
            elif key in ['molecule_size','idx']: #,"orbitals"
                processed[key] = torch.Tensor([list_of_data[i][key] for i in range(bs_mol)]).int()
            # else:
            #     processed[key] = torch.Tensor([list_of_data[i][key] for i in range(bs_mol)])
        processed["batch"] = torch.concat([i*torch.ones(list_of_data[i]["pos"].shape[0], dtype=torch.int64) 
                                        for i in range(bs_mol)])

        if "labels" not in list_of_data[0].keys():
            return processed
        
        # for LSRM
        for d in list_of_data:
            if isinstance(d["labels"],torch.Tensor):
                continue
            else:
                d["labels"] = torch.tensor(d['labels'])
        labels = torch.cat([d["labels"] for d in list_of_data])
        

        pos = processed["pos"]
        batch = processed["batch"]
        # concatenate graph inside data object to get a single graph (block concatenation)
        # pair_index = torch.cat([d['intera'] for d in list_of_data], dim = -1)
        # grouping_graph = torch.cat([d['grouping_graph'] for d in list_of_data], dim = -1)
        edge_index = torch.cat([torch.tensor(list_of_data[i]["edge_index"]) for i in range(len(list_of_data))],dim = 1)
        node_id,group_id,inter_indices = [],[],[]
        for index,data in enumerate(list_of_data):
            count = 0
            group_pos = scatter(torch.tensor(data['pos'] * data['atomic_numbers'].reshape(-1,1)), 
                                data['labels'].to(torch.int64), reduce='sum', dim=0) / \
                                scatter(torch.tensor(data['atomic_numbers'].reshape(-1,1)),data['labels'].to(torch.int64), reduce='sum', dim=0)
            for i in range(data['pos'].shape[0]):
                for j in range(group_pos.shape[0]):
                    node_group_dis = torch.sqrt(torch.sum((pos[i]-group_pos[j])**2,dim=-1))
                    if node_group_dis<long_cutoff_upper:
                        node_id.append(i)
                        group_id.append(j)
                        count = count+1
            inter_indices.append(torch.zeros(count, dtype=torch.long).fill_(index))
        # interaction_graph = torch.cat([d['interaction_graph'] for d in list_of_data], dim = -1)
        interaction_graph = torch.stack((torch.tensor(node_id), torch.tensor(group_id)),dim=0)

        # indices are indicator varible indicating which batch does the current edge belongs to
        edgex_index_indices = torch.cat([torch.zeros(d['edge_index'].shape[1], dtype=torch.long).fill_(i) for i, d in enumerate(list_of_data)])
        # group_graph_indices = torch.cat([torch.zeros(d['grouping_graph'].shape[1], dtype=torch.long).fill_(i) for i, d in enumerate(list_of_data)])
        # interaction_graph_indices = torch.cat([torch.zeros(d['interaction_graph'].shape[1], dtype=torch.long).fill_(i) for i, d in enumerate(list_of_data)])
        interaction_graph_indices = torch.cat(inter_indices)
        # pair_index_indices = torch.cat([torch.zeros(d['intera'].shape[1], dtype=torch.long).fill_(i) for i, d in enumerate(list_of_data)])
        # concatenate the labels

        num_nodes = sum([d['pos'].shape[0] for d in list_of_data])
        max_num_nodes = max(d['pos'].shape[0] for d in list_of_data)
        num_labels = sum([d['labels'].unique().shape[0] for d in list_of_data])
        max_num_labels = max(d['labels'].unique().shape[0] for d in list_of_data)

        node_idx_mapping_source = torch.cat([torch.arange(d['pos'].shape[0], dtype=torch.long) for d in list_of_data])
        node_idx_mapping_target = torch.arange(num_nodes, dtype=torch.long)


        # remap the labels
        label_batch = torch.cat([torch.zeros(d['labels'].unique().shape[0], dtype=torch.long).fill_(i) for i, d in enumerate(list_of_data)])
        label_idx_remapped_source = torch.cat([torch.arange(d['labels'].unique().shape[0], dtype=torch.long) for d in list_of_data])
        label_idx_remapped_target = torch.arange(num_labels, dtype=torch.long)
        labels = mapping_function(labels, processed["batch"], label_idx_remapped_source, label_idx_remapped_target, label_batch, max_num_labels)
        # remap the short term graph
        edge_index = mapping_function(edge_index, edgex_index_indices, node_idx_mapping_source, node_idx_mapping_target, batch, max_num_nodes)
        # remap the pair-atom graph
        # pair_index = mapping_function(pair_index, pair_index_indices, edge_idx_mapping_source, edge_idx_mapping_target, edge_batch, max_num_edges)
        # remap the grouping graph
        # grouping_graph = mapping_function(grouping_graph, group_graph_indices, node_idx_mapping_source, node_idx_mapping_target, batch, max_num_nodes)
        # remap the interaction graph
        interaction_graph_src = mapping_function(interaction_graph[0], interaction_graph_indices, node_idx_mapping_source, node_idx_mapping_target, batch, max_num_nodes)
        interaction_graph_tgt = mapping_function(interaction_graph[1], interaction_graph_indices, label_idx_remapped_source, label_idx_remapped_target, label_batch, max_num_labels)
        interaction_graph = torch.stack([interaction_graph_src, interaction_graph_tgt], dim=0)
        return Data(
                    edge_index=edge_index, 
                    # grouping_graph=grouping_graph, 
                    interaction_graph=interaction_graph, 
                    labels=labels, num_nodes=num_nodes, num_labels=num_labels, 
                    label_batch = label_batch,
                    **processed)
    
    return _collate_fn  

def get_dataloader(dataset: Dataset,
                   batch_size:int,
                   num_workers = 0, 
                   pin_memory = False,
                   collate_fn = collate_fn_unified
                   ) -> DataLoader:

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    return data_loader
