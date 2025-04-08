import torch
import numpy as np
from torch_scatter import scatter
from sklearn.cluster import KMeans,SpectralClustering

# from lightnp.utils import MolGraph
from sklearn.cluster import KMeans,SpectralClustering,DBSCAN


def build_label(g, num_labels = 16,method = 'kmeans'):
    if num_labels == g.pos.shape[0]:
        g.labels = torch.arange(g.pos.shape[0]).long()
    elif num_labels == 1:
        g.labels = torch.zeros(g.pos.shape[0]).long()
    else:
        try:
            if method == 'kmeans':
                g.labels = torch.tensor(KMeans(n_clusters=num_labels, random_state=0).fit_predict(g.pos.numpy())).long()
            elif method == 'spectral':
                g.labels = torch.tensor(SpectralClustering(
                                                    n_clusters=num_labels,
                                                    eigen_solver="arpack",
                                                    affinity="nearest_neighbors",
                                                ).fit_predict(g.pos.numpy())).long()
        except:
            # Except for # nodes < num_labels
            g.labels = torch.arange(g.num_nodes).long()
    g.num_labels = num_labels
    

# node_ring = None
# step = 0
# interaction_graph = None
def build_label_two(g,num_labels = 16):
    pos = g.pos.numpy()
    # atomic_numbers = g.atomic_numbers.numpy().reshape(-1)

    # model = SpectralClustering(
    #                                     n_clusters=num_labels,
    #                                     eigen_solver="arpack",
    #                                     affinity="nearest_neighbors",
    #                                 )
    model = SpectralClustering(
                                        n_clusters=num_labels,
                                        eigen_solver="arpack",
                                        affinity="precomputed",
                                    )
    sim = np.sqrt(np.sum((pos.reshape(-1,1,3)-pos.reshape(1,-1,3))**2,axis = -1))
    # sim[np.arange(pos.shape[0]),np.arange(pos.shape[0])] = 1
    sim[sim>2] = 1000
    sim = 1./(sim+1)
    # sim[sim>0] = 1

    labels = model.fit_predict(sim)

    sc = SpectralClustering(2, affinity='precomputed', n_init=100,
                            assign_labels='discretize')
    node_ring = sc.fit_predict(sim)
    group_ring = scatter(torch.from_numpy(node_ring).float(), torch.from_numpy(labels).long(), reduce='mean', dim=0)
    group_ring = (group_ring>0.5).long().numpy()

    nid = np.tile(np.arange(node_ring.shape[0]).reshape(-1,1),[1,len(group_ring)])
    gid = np.tile(np.arange(group_ring.shape[0]).reshape(1,-1),[len(pos),1])

    interaction_graph = np.stack([nid.reshape(-1),gid.reshape(-1)],axis = 0)
    interaction_graph = interaction_graph[:,node_ring[interaction_graph[0]]==group_ring[interaction_graph[1]]]
    interaction_graph = torch.from_numpy(interaction_graph).long()
    
    
    g.labels = torch.from_numpy(labels).long()
    g.interaction_graph = interaction_graph
    g.num_labels = num_labels