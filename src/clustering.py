import numpy as np
from scipy.sparse.csgraph import laplacian, connected_components
from scipy.spatial import distance_matrix



def epsdel_clustering(data, epsv=1, delv=1):
    assert epsv <= delv
    dist_mat = distance_matrix(data, data)
    adj_mat = np.array(dist_mat)
    adj_mat[adj_mat <= epsv] = 0.
    adj_mat[adj_mat > delv] = 1.
    lap_mat = laplacian(adj_mat)
    n_clusters, clust_labels, _ = connected_components(adj_mat)
    return clust_labels, n_clusters, adj_mat, lap_mat



