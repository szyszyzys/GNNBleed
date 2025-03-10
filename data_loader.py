import itertools
import json
import os.path as osp
import random
import time
from typing import Optional, Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from pytorch_lightning.core.datamodule import LightningDataModule
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import Planetoid, Flickr, Twitch, DGraphFin, LastFMAsia
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph, remove_self_loops, to_scipy_sparse_matrix, from_scipy_sparse_matrix
from tqdm import tqdm

from utils.graph_utils import get_noise

DEFAULT_DATA_DIR = './'
DEFAULT_NUM_WORKERS = 4


class GraphDataModule(LightningDataModule):
    def __init__(
            self,
            dataset_name: str = 'cora',
            is_inductive=False,
            batch_size: int = 32,
            data_dir: str = DEFAULT_DATA_DIR,
            num_workers: int = DEFAULT_NUM_WORKERS,
            seed: int = None,
            remove_self_loop=False,
            implement_dp=False,
            dp_epsilon=0.1,
            noise_seed=42,
            dp_delta=1e-5,
            noise_type='laplace',
            perturb_type='lapgraph',

    ):
        super().__init__()
        self.name = dataset_name.lower()
        dataset_name = dataset_name.lower()
        if dataset_name in ['cora']:
            self.dataset = Planetoid(root=data_dir, name=dataset_name,
                                     )
        elif dataset_name.startswith('twitch'):
            self.dataset = Twitch(root='./', name='ES')
        elif dataset_name.startswith('flickr'):
            self.dataset = Flickr(root='./data/flickr/', )
        elif dataset_name.startswith('dgraph'):
            self.dataset = DGraphFin(root='./data/dgraph/', )
        elif dataset_name.startswith('lastfm'):
            self.dataset = LastFMAsia(root='./data/lastfm/', )
        # load the graph from the dataset
        if remove_self_loop:
            self.dataset[0].edge_index = remove_self_loops(self.dataset[0].edge_index)
        self.data = self.dataset[0]
        # parameters for DP
        self.dp_epsilon = dp_epsilon
        self.noise_seed = noise_seed
        self.dp_delta = dp_delta
        self.noise_type = noise_type
        # Print information about the dataset
        print(f'Dataset: {self.dataset}')
        print('-------------------')
        print(f'Number of graphs: {len(self.dataset)}')
        print(f'Number of nodes: {self.data.x.shape[0]}')
        print(f'Number of features: {self.dataset.num_features}')
        print(f'Number of classes: {self.dataset.num_classes}')
        self.n_nodes = self.data.x.shape[0]
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # perturb the graph if we implement DP
        if dataset_name.startswith('twitch'):
            twitch_set_name = dataset_name[dataset_name.find('/') + 1:]
            self.train_data = self.dataset
            self.test_set = Twitch(root='./twitch', name=twitch_set_name.upper())
            if implement_dp:
                self.train_data[0].edge_index = self.add_noise_to_graph(self.train_data[0].edge_index, perturb_type)
                self.test_set[0].edge_index = self.add_noise_to_graph(self.test_set[0].edge_index, perturb_type)

            if remove_self_loop:
                self.test_set[0].edge_index = remove_self_loops(self.test_set[0].edge_index)

            self.train_loader = DataLoader(self.train_data, batch_size=64, shuffle=True, num_workers=self.num_workers)
            self.val_loader = DataLoader(self.test_set, batch_size=64, shuffle=False, num_workers=self.num_workers)
            self.test_loader = DataLoader(self.test_set, batch_size=64, shuffle=False, num_workers=self.num_workers)

            self.attack_graph = self.test_set[0].clone()
        elif dataset_name.startswith('dgraph'):
            self.train_loader = DataLoader([self.data], batch_size=64, shuffle=False, num_workers=self.num_workers)
            self.val_loader = DataLoader([self.data], batch_size=64, shuffle=False, num_workers=self.num_workers)
            self.test_loader = DataLoader([self.data], batch_size=64, shuffle=False, num_workers=self.num_workers)
        elif dataset_name.lower() in ('flickr', 'yelp', 'amazon'):
            self.train_data = construct_subgraph(self.data, self.data.train_mask)

            if implement_dp:
                self.train_data.edge_index = self.add_noise_to_graph(self.train_data.edge_index, perturb_type)
                self.data.edge_index = self.add_noise_to_graph(self.data.edge_index, perturb_type)

            self.train_loader = DataLoader([self.train_data], batch_size=64, shuffle=True, num_workers=self.num_workers)

            self.val_loader = DataLoader([self.data], batch_size=64, shuffle=False, num_workers=self.num_workers)
            self.test_loader = DataLoader([self.data], batch_size=64, shuffle=False, num_workers=self.num_workers)
            self.attack_graph = self.data.clone()

        elif dataset_name.lower() in ('lastfm'):
            # split the data
            split_dataset(self.data, train_ratio=0.7)
            self.train_data = construct_subgraph(self.data, self.data.train_mask)
            if implement_dp:
                self.data.edge_index = self.add_noise_to_graph(self.data.edge_index, perturb_type)
                self.train_data = construct_subgraph(self.data, self.data.train_mask)

            self.train_loader = DataLoader([self.train_data], batch_size=64, shuffle=True, num_workers=self.num_workers)

            self.val_loader = DataLoader([self.data], batch_size=64, shuffle=False, num_workers=self.num_workers)
            self.test_loader = DataLoader([self.data], batch_size=64, shuffle=False, num_workers=self.num_workers)
            self.attack_graph = self.data.clone()
            print("_+____________________________________________")

    def setup(self, stage: Optional[str] = None):
        print("setup")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def predict_dataloader(self):
        return self.test_loader

    @property
    def num_classes(self):
        return self.dataset.num_classes

    @property
    def num_features(self):
        return self.dataset.num_features

    @property
    def full_graph(self):
        return self.data

    @property
    def full_attack_graph(self):
        return self.attack_graph

    @property
    def dataset_name(self):
        return self.name

    def add_noise_to_graph(self, edge_index, perturb_type):
        adj = to_scipy_sparse_matrix(edge_index)
        res = None
        if perturb_type == 'randedge':
            res = self.perturb_adj_discrete(adj)
        elif perturb_type == 'lapgraph':
            res = self.perturb_adj_continuous(adj)
        res, _ = from_scipy_sparse_matrix(res)
        return res

    def construct_sparse_mat(self, indice, N):
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

    def perturb_adj_discrete(self, adj):
        s = 2 / (np.exp(self.dp_epsilon) + 1)
        print(f's = {s:.4f}')
        N = adj.shape[0]

        np.random.seed(self.noise_seed)
        bernoulli = np.random.binomial(1, s, (N, N))

        entry = np.asarray(list(zip(*np.where(bernoulli))))

        dig_1 = np.random.binomial(1, 1 / 2, len(entry))
        indice_1 = entry[np.where(dig_1 == 1)[0]]
        indice_0 = entry[np.where(dig_1 == 0)[0]]

        add_mat = self.construct_sparse_mat(indice_1, N)
        minus_mat = self.construct_sparse_mat(indice_0, N)

        adj_noisy = adj + add_mat - minus_mat

        adj_noisy.data[np.where(adj_noisy.data == -1)[0]] = 0
        adj_noisy.data[np.where(adj_noisy.data == 2)[0]] = 1

        return adj_noisy

    def perturb_adj_continuous(self, adj):
        n_edges = len(adj.data) // 2

        N = adj.shape[0]
        t = time.time()

        A = sp.tril(adj, k=-1)
        print('getting the lower triangle of adj matrix done!')

        eps_1 = self.dp_epsilon * 0.01
        eps_2 = self.dp_epsilon - eps_1
        noise = get_noise(noise_type=self.noise_type, size=(N, N), seed=self.noise_seed,
                          eps=eps_2, delta=self.dp_delta, sensitivity=1)
        noise *= np.tri(*noise.shape, k=-1, dtype=np.bool_)
        print(f'generating noise done using {time.time() - t} secs!')

        A += noise
        print(f'adding noise to the adj matrix done!')

        t = time.time()
        n_edges_keep = n_edges + int(
            get_noise(noise_type=self.noise_type, size=1, seed=self.noise_seed,
                      eps=eps_1, delta=self.dp_delta, sensitivity=1)[0])
        print(f'edge number from {n_edges} to {n_edges_keep}')
        a_r = A.A.ravel()

        n_splits = 50
        len_h = len(a_r) // n_splits
        ind_list = []
        for i in tqdm(range(n_splits - 1)):
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
        print(f'data preparation done using {time.time() - t} secs!')

        mat = sp.csr_matrix((data_idx, (row_idx, col_idx)), shape=(N, N))
        return mat + mat.T


class GraphDataset(InMemoryDataset):

    def __init__(self, root: str = './', name: str = 'twitch/ES',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):

        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def load_data(self):
        print(f'loading dataset: {self.name}!')
        if self.name.startswith('twitch'):
            x, edge_index, y = twitch_loader(self.dataset_target)
            scaler = StandardScaler()
            scaler.fit(x)
            x = torch.FloatTensor(scaler.transform(x))

            data = Data(x=x, edge_index=edge_index, y=y)

            return data

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        self.load_data()

        data = self.load_data()

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


def twitch_loader(dataset_name='twitch/ES', feature_size=-1):
    edge_index, n_nodes = read_edge(dataset_name)
    x = read_feature(dataset_name, feature_size)
    y = read_label(dataset_name, n_nodes)
    # unique, count = torch.unique(labels, return_counts=True)
    return x, edge_index, y


def read_edge(dataset, n_nodes=-1):
    if dataset.startswith('twitch'):
        identifier = dataset[dataset.find('/') + 1:]
        data = pd.read_csv(f'./twitch/{identifier}/musae_{identifier}_edges.csv')
        edges = data.values
        n = np.append(edges[:, 0], edges[:, 1])
        n_nodes = len(np.unique(n))
        # adj = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        #                     shape=(n_nodes, n_nodes),
        #                     dtype=np.float32)
        # return adj + adj.T, n_nodes
        return torch.tensor(edges.transpose(), dtype=torch.int64).contiguous(), n_nodes


def read_label(dataset, n_nodes=-1):
    if dataset.startswith('twitch'):
        identifier = dataset[dataset.find('/') + 1:]
        data = pd.read_csv('./data/{}/musae_{}_target.csv'.format(dataset, identifier))
        mature = list(map(int, data['mature'].values))
        new_id = list(map(int, data['new_id'].values))
        idx_map = {elem: i for i, elem in enumerate(new_id)}
        labels = [mature[idx_map[idx]] for idx in range(n_nodes)]

        labels = torch.LongTensor(labels)
        return labels


def read_feature(dataset, feature_size=-1):
    if dataset.startswith('twitch/'):
        if dataset.startswith('twitch'):
            identifier = dataset[dataset.find('/') + 1:]
            filename = './data/{}/musae_{}_features.json'.format(dataset, identifier)

        # read from json feature
        with open(filename) as f:
            data = json.load(f)
            n_nodes = len(data)

            items = sorted(set(itertools.chain.from_iterable(data.values())))
            n_features = 3170 if dataset.startswith('twitch') else max(items) + 1

            features = np.zeros((n_nodes, n_features))
            for idx, elem in data.items():
                features[int(idx), elem] = 1
        return features


def construct_subgraph(dataset, mask):
    sub_edge_index, _ = subgraph(mask, dataset.edge_index, relabel_nodes=True)
    sub_x = dataset.x[mask]
    sub_y = dataset.y[mask]
    return Data(x=sub_x, edge_index=sub_edge_index, y=sub_y)


def split_dataset(data, train_ratio=0.7, seed=42):
    n_samples = data.y.shape[0]
    test_ratio = val_ratio = (1 - train_ratio) / 2
    val_num = int(n_samples * val_ratio)
    test_num = int(n_samples * test_ratio)
    train_num = int(train_ratio * n_samples)
    idx_all = list(range(n_samples))

    random.seed(seed)
    random.shuffle(idx_all)
    test_idx = idx_all[:test_num]
    val_idx = idx_all[test_num:test_num + val_num]
    train_idx = idx_all[test_num + val_num:]

    train_mask = torch.zeros(n_samples, dtype=torch.bool)
    train_mask[torch.tensor(train_idx)] = True

    val_mask = torch.zeros(n_samples, dtype=torch.bool)
    val_mask[torch.tensor(val_idx)] = True

    test_mask = torch.zeros(n_samples, dtype=torch.bool)
    test_mask[torch.tensor(test_idx)] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
