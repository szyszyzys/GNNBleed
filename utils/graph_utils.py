import math
import os
import random

import numpy as np
import scipy as sp
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_scipy_sparse_matrix, from_scipy_sparse_matrix
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm


def save_graph(x, edge_index, edge_attr, y, graph_name=0, test_mask=None, train_mask=None, valid_mask=None):
    if not os.path.isdir(f'./reconstructed_graph'):
        os.makedirs(f'./reconstructed_graph')

    edge_index = edge_index.transpose(0, 1)
    np.savez(f'./reconstructed_graph/reconstructed_graph-{graph_name}', edge_index=edge_index, x=x, edge_type=edge_attr,
             y=y, test_mask=test_mask, train_mask=train_mask, valid_mask=valid_mask)


def add_new_edge_to_graph(dataset, node_1, node_2):
    new_edge = torch.tensor([[node_1, node_2], [node_2, node_1]])
    dataset.edge_index = torch.cat((dataset.edge_index, new_edge), 1)


def remove_column_with_specific_value(vector, target_value):
    index_to_keep = []
    for i in range(vector.shape[1]):
        if target_value not in vector[:, i]:
            index_to_keep.append(i)
    vector = vector[:, index_to_keep]
    return vector


def remove_target_node(dataset, target_node):
    # remove_node_feature
    dataset.x = torch.cat((dataset.x[:target_node], dataset.x[target_node + 1:]))
    dataset.y = torch.cat((dataset.y[:target_node], dataset.y[target_node + 1:]))

    # remove from edge_index
    dataset.edge_index = remove_column_with_specific_value(dataset.edge_index, target_node)
    return dataset


# def add_new_node_to_graph(dataset, new_node_attr, target_node, edge_attr=-1, y=-1):
#     # add new edge to edge_index
#     new_edge = torch.tensor([[target_node, dataset.x.shape[0]], [dataset.x.shape[0], target_node]]).cuda()
#     dataset.edge_index = torch.cat((dataset.edge_index, new_edge), 1)
#
#     # add new node's attribute
#     new_node_attr = new_node_attr.unsqueeze(0)
#     dataset.x = torch.cat((dataset.x, new_node_attr), 0)
#
#     dataset.y = torch.cat((dataset.y, torch.tensor([y]).cuda()), 0)
# def add_new_node_to_graph(dataset, new_node_attr, target_node, edge_attr=-1, y=-1):
#     """
#     Adds a new node to a PyG dataset and connects it to a target node.
#
#     This function bypasses the read-only property of `edge_index` by modifying
#     the dataset's internal dictionary.
#
#     Args:
#         dataset (Data): A PyG data object with attributes such as x, y, edge_index,
#                         and optionally edge_attr.
#         new_node_attr (Tensor): A tensor containing the new node's features (shape: (d,)).
#         target_node (int): The index of the node to which the new node will be connected.
#         edge_attr (any, optional): The attribute for the new edge(s). Default is -1.
#         y (any, optional): The label for the new node. Default is -1.
#     """
#     device = dataset.x.device
#     new_node_idx = dataset.x.size(0)  # index for the new node
#
#     # Create new edges connecting the target node and the new node (both directions).
#     new_edge = torch.tensor([[target_node, new_node_idx],
#                              [new_node_idx, target_node]], device=device)
#
#     # Update edge_index by modifying the underlying __dict__.
#     dataset.__dict__['edge_index'] = torch.cat((dataset.edge_index, new_edge), dim=1)
#
#     # Add new node features.
#     new_node_attr = new_node_attr.unsqueeze(0).to(device)
#     dataset.__dict__['x'] = torch.cat((dataset.x, new_node_attr), dim=0)
#
#     # Add new node label.
#     new_label = torch.tensor([y], device=device, dtype=dataset.y.dtype)
#     dataset.__dict__['y'] = torch.cat((dataset.y, new_label), dim=0)

def add_new_node_to_graph(data, new_node_attr, target_node, edge_attr=-1, y=-1):
    """
    Adds a new node to a PyG Data object and connects it to a target node.

    This function updates the Data object's attributes via direct assignment.

    Args:
        data (Data): A PyG Data object with attributes such as x, y, edge_index,
                     and optionally edge_attr.
        new_node_attr (Tensor): A tensor containing the new node's features (shape: (d,)).
        target_node (int): The index of the node to which the new node will be connected.
        edge_attr (any, optional): The attribute for the new edge(s). Default is -1.
        y (any, optional): The label for the new node. Default is -1.
    """
    device = data.x.device
    new_node_idx = data.x.size(0)  # new node index

    # Create new edges connecting the target node and the new node (both directions)
    new_edge = torch.tensor([[target_node, new_node_idx],
                             [new_node_idx, target_node]], device=device)

    # Update edge_index directly
    data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)

    # If the Data object has edge_attr, update it as well
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        new_edge_attr = torch.tensor([[edge_attr], [edge_attr]], device=device,
                                     dtype=data.edge_attr.dtype)
        data.edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)

    # Append new node features to data.x
    new_node_attr = new_node_attr.unsqueeze(0).to(device)  # shape becomes (1, d)
    data.x = torch.cat([data.x, new_node_attr], dim=0)

    # Append new node label to data.y
    new_label = torch.tensor([y], device=device, dtype=data.y.dtype)
    data.y = torch.cat([data.y, new_label], dim=0)

    # Optionally update num_nodes if the attribute exists
    if hasattr(data, 'num_nodes'):
        data.num_nodes = data.x.size(0)


def matrix_to_edge_index(matrix):
    edge_index = matrix.nonzero().t().contiguous()

    return edge_index


def sparse_matrix_to_egde_index(adj):
    row, col, attr = adj.t().coo()
    return torch.stack([row, col], dim=0)


def to_undirected_graph(adj_t):
    adj_t = adj_t.to_symmetric()
    return adj_t


def get_subgraph(targetNodes=[], khop=5, edge_index=None):
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(targetNodes, khop, edge_index)
    return subset, edge_index, mapping, edge_mask


def find_neighborhoods(target_node, edge_index):
    neighbor_list = set()
    for i in range(edge_index.shape[1]):
        if edge_index[0][i] == target_node or edge_index[1][i] == target_node:
            neighbor_list.add(edge_index[0][i].item())
            neighbor_list.add(edge_index[1][i].item())

    return list(neighbor_list)


def construct_k_hop_subgraph(data, target_nodes, k_hop=3):
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(target_nodes, k_hop, data.edge_index, relabel_nodes=True)

    return Data(x=data.x[subset], edge_index=edge_index, y=data.y[subset]), mapping


def get_degree(data):
    deg = degree(data.edge_index[0], data.num_nodes)
    return deg


def get_noise(noise_type, size, seed, eps=10, delta=1e-5, sensitivity=2):
    np.random.seed(seed)

    if noise_type == 'laplace':
        noise = np.random.laplace(0, sensitivity / eps, size)
    elif noise_type == 'gaussian':
        c = np.sqrt(2 * np.log(1.25 / delta))
        stddev = c * sensitivity / eps
        noise = np.random.normal(0, stddev, size)
    else:
        raise NotImplementedError('noise {} not implemented!'.format(noise_type))

    return noise


def edge_index_to_spmatrix(edge_index):
    # Determine the number of nodes in the graph
    num_nodes = max(max(edge[0], edge[1]) for edge in edge_index) + 1

    # Initialize lists for the CSR matrix
    data = [1] * edge_index.shape[1]  # Assuming an unweighted graph
    row_indices = [edge[0] for edge in edge_index]
    column_indices = [edge[1] for edge in edge_index]

    # Create the CSR matrix
    adj_matrix = sp.csr_matrix((data, (row_indices, column_indices)), shape=(num_nodes, num_nodes))

    return adj_matrix


def perturb_node_feature(dataset, factor_range=0.01, static_nodes=[]):
    dataset.x = dataset.x * generate_reweight_factors(dataset.x.shape[0], factor_range=factor_range,
                                                      static_nodes=static_nodes)


def generate_reweight_factors(num_rows, factor_range=0.01, static_nodes=[]):
    # Create an alternating list of factors
    lower = 1 - factor_range
    factors = torch.rand(num_rows) * (2 * factor_range) + lower  # Scale and shift to get values between 0.9 and 1.1
    factors[static_nodes] = 1
    return factors.clone().detach().view(-1, 1).cuda()


def evolve_graph(graph, new_node_rate=0.01, new_edge_rate=0.01, feature_change_rate=0.01, static_nodes=[],
                 target_node=0, n_neighbor_node_rate=0.2, evolving_mode="all", defense=False, perturb_type="lapgraph",
                 dp_epsilon=1):
    n_nodes = graph.x.shape[0]
    n_new_nodes = int(new_node_rate * n_nodes)
    n_new_edges = int(new_edge_rate * graph.edge_index.shape[1])

    if evolving_mode == "all" or evolving_mode == "feature":
        # Change node features
        perturb_node_feature(graph, feature_change_rate, static_nodes)

    if evolving_mode == "all" or evolving_mode == "local_structure":
        k_hops = get_k_hop_neighbor(target_node, 2, graph.edge_index)

        n_neighbor_node = math.ceil(n_neighbor_node_rate * len(k_hops))
        n_neighbor_node = random.randint(n_neighbor_node // 2, n_neighbor_node)
        for i in range(n_neighbor_node):
            tmp_node = random.randint(0, len(k_hops) - 1)
            add_new_node_to_graph(graph, graph.x[k_hops[tmp_node]], k_hops[tmp_node], edge_attr=-1, y=-1)

    # Add new nodes
    if evolving_mode == "all" or evolving_mode == "structure":
        for _ in range(n_new_nodes):
            new_node_i = random.randint(0, n_nodes - 1)
            add_new_node_to_graph(graph, graph.x[new_node_i], new_node_i, edge_attr=-1, y=-1)

    if defense:
        graph.edge_index = add_noise_to_graph(graph.edge_index, perturb_type, dp_epsilon).cuda()
    #
    # for _ in range(n_new_edges):
    #     # Randomly select two different nodes
    #     node_a, node_b = random.sample(node_list, 2)
    #     graph.add_edge(node_a, node_b)


def perturb_adj_discrete(adj, noise_type, dp_epsilon=0.1, noise_seed=42, dp_delta=1e-5):
    s = 2 / (np.exp(dp_epsilon) + 1)
    print(f's = {s:.4f}')
    N = adj.shape[0]

    np.random.seed(noise_seed)
    bernoulli = np.random.binomial(1, s, (N, N))

    entry = np.asarray(list(zip(*np.where(bernoulli))))

    dig_1 = np.random.binomial(1, 1 / 2, len(entry))
    indice_1 = entry[np.where(dig_1 == 1)[0]]
    indice_0 = entry[np.where(dig_1 == 0)[0]]

    add_mat = construct_sparse_mat(indice_1, N)
    minus_mat = construct_sparse_mat(indice_0, N)

    adj_noisy = adj + add_mat - minus_mat

    adj_noisy.data[np.where(adj_noisy.data == -1)[0]] = 0
    adj_noisy.data[np.where(adj_noisy.data == 2)[0]] = 1

    return adj_noisy


def construct_sparse_mat(indice, N):
    cur_row = -1
    new_indices = []
    new_indptr = []

    for i, j in tqdm(indice):
        if i >= j:
            continue

        while i > cur_row:
            new_indptr.append(len(new_indices))
            cur_row += 1

        new_indices.append(j)

    while N > cur_row:
        new_indptr.append(len(new_indices))
        cur_row += 1

    data = np.ones(len(new_indices), dtype=np.int64)
    indices = np.asarray(new_indices, dtype=np.int64)
    indptr = np.asarray(new_indptr, dtype=np.int64)

    mat = sp.csr_matrix((data, indices, indptr), (N, N))

    return mat + mat.T


def perturb_adj_continuous(adj, noise_type="laplace", dp_epsilon=0.1, noise_seed=42, dp_delta=1e-5):
    n_edges = len(adj.data) // 2

    N = adj.shape[0]
    A = sp.tril(adj, k=-1)

    eps_1 = dp_epsilon * 0.01
    eps_2 = dp_epsilon - eps_1
    noise = get_noise(noise_type=noise_type, size=(N, N), seed=noise_seed,
                      eps=eps_2, delta=dp_delta, sensitivity=1)
    noise *= np.tri(*noise.shape, k=-1, dtype=np.bool_)
    A += noise
    n_edges_keep = n_edges + int(
        get_noise(noise_type=noise_type, size=1, seed=noise_seed,
                  eps=eps_1, delta=dp_delta, sensitivity=1)[0])
    a_r = A.A.ravel()

    n_splits = 50
    len_h = len(a_r) // n_splits
    ind_list = []
    for i in range(n_splits - 1):
        ind = np.argpartition(a_r[len_h * i:len_h * (i + 1)], -n_edges_keep)[-n_edges_keep:]
        ind_list.append(ind + len_h * i)

    ind = np.argpartition(a_r[len_h * (n_splits - 1):], -n_edges_keep)[-n_edges_keep:]
    ind_list.append(ind + len_h * (n_splits - 1))

    ind_subset = np.hstack(ind_list)
    a_subset = a_r[ind_subset]
    ind = np.argpartition(a_subset, -n_edges_keep)[-n_edges_keep:]

    row_idx = []
    col_idx = []
    for idx in ind:
        idx = ind_subset[idx]
        row_idx.append(idx // N)
        col_idx.append(idx % N)
        assert (col_idx < row_idx)
    data_idx = np.ones(n_edges_keep, dtype=np.int32)

    mat = sp.csr_matrix((data_idx, (row_idx, col_idx)), shape=(N, N))
    return mat + mat.T


def add_noise_to_graph(edge_index, perturb_type, dp_epsilon):
    adj = to_scipy_sparse_matrix(edge_index)
    res = None
    if perturb_type == 'randedge':
        res = perturb_adj_discrete(adj, dp_epsilon=dp_epsilon)
    elif perturb_type == 'lapgraph':
        res = perturb_adj_continuous(adj, dp_epsilon=dp_epsilon)
    res, _ = from_scipy_sparse_matrix(res)
    return res


def create_sparse_dissimilar_vector(input_vector):
    """
    Create a sparse vector dissimilar to the input vector.
    The sparsity of the generated vector will be similar to the input vector.
    """
    input_vector = input_vector.cpu()
    input_vector = np.array(input_vector)
    sparse_vector = np.zeros_like(input_vector)

    # Identify zero and non-zero positions
    zero_positions = np.where(input_vector == 0)[0]
    non_zero_positions = np.where(input_vector != 0)[0]

    # If no zero positions are found, return the original input_vector as is
    if len(zero_positions) == 0:
        original_sum = np.sum(input_vector)
        random_vector = np.random.rand(len(input_vector))
        scale_factor = original_sum / np.sum(random_vector)
        dissimilar_vector = random_vector * scale_factor
        dissimilar_vector = dissimilar_vector.astype(np.float32)
        res = torch.tensor(dissimilar_vector).cuda()
        return res
    # Assign random values to zero positions
    sparse_vector[zero_positions] = np.random.rand(len(zero_positions))

    # Assign small values close to zero to a few non-zero positions to maintain similar sparsity
    num_small_values = min(len(non_zero_positions), len(zero_positions))
    small_value_positions = np.random.choice(non_zero_positions, num_small_values, replace=False)
    sparse_vector[small_value_positions] = np.random.uniform(low=0.0, high=0.01, size=num_small_values)

    # Normalize the sparse vector
    sparse_vector = sparse_vector / sparse_vector.sum() * input_vector.sum()
    return torch.tensor(sparse_vector, dtype=torch.double).cuda()


def get_k_hop_neighbor(target_node, k_hop, edge_index):
    k_hops, _, _, _ = k_hop_subgraph(target_node, k_hop, edge_index, relabel_nodes=False)
    k_hops = k_hops.tolist()
    return k_hops
