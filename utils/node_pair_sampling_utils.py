import os
import random

import torch
from networkx import shortest_path
from torch_geometric.utils import k_hop_subgraph, to_networkx

SAVE_PATH = './revision/outputs/'


class NodePairsSampler:
    def __init__(self, graph_dataset, dataset_name, root_path='./outputs', sub_path='/'):
        self.dataset_name = dataset_name
        self.root_path = root_path
        self.sub_path = sub_path
        self.num_nodes = graph_dataset.x.shape[0]
        self.picked_nodes = set()

    # def get_node_centroid_pairs(self, target_node, set_size=50, k_hop=3, maximum_connected=20, balanced=False):
    #     # get all the nodes connected to the target node
    #     neighbors, _, _, _ = k_hop_subgraph(target_node, 1, self.graph_dataset.edge_index, relabel_nodes=False)
    #     # remove the target node itself
    #     neighbors = neighbors.tolist()
    #     neighbors.remove(target_node)
    #     k_hops, _, _, _ = k_hop_subgraph(target_node, k_hop, self.graph_dataset.edge_index, relabel_nodes=False)
    #     k_hops = k_hops[1:].tolist()
    #
    #     num_connected = len(neighbors)
    #     node_sets = neighbors[:maximum_connected]
    #     if balanced:
    #         for i in k_hops:
    #             if len(node_sets) >= 2 * maximum_connected:
    #                 break
    #             if i in neighbors:
    #                 continue
    #             node_sets.append(i)
    #     else:
    #         for i in k_hops:
    #             if len(node_sets) >= set_size:
    #                 break
    #             if i in neighbors:
    #                 continue
    #             node_sets.append(i)
    #     print(
    #         f'find node pairs contain: {target_node}, find {num_connected} connected nodes and'
    #         f' {set_size - min(num_connected, maximum_connected)} unconnected node pairs')
    #
    #     # node_pairs = []
    #     # for i in node_sets[:set_size]:
    #     #     node_pairs.append((target_node, i))
    #     return node_sets, min(num_connected, maximum_connected)

    # def construct_centroid_pairs_set_process(self, target_node, set_size=50, k_hop=3, maximum_connected=20,
    #                                          balanced=False):
    #     for node_id in target_node:
    #         neighbor, num_connected = self.get_node_centroid_pairs(node_id, set_size=set_size, k_hop=k_hop,
    #                                                                maximum_connected=maximum_connected,
    #                                                                balanced=balanced)
    #         self.save_node_pairs(
    #             {"target_node": node_id, "neighbor": neighbor, "num_direct_connection": num_connected},
    #             self.dataset_name, str(node_id), balanced)
    #
    # def construct_centroid_pairs_set(self, num_target, set_size=50, k_hop=3, maximum_connected=20,
    #                                  degree_range=None, balanced=False):
    #     if degree_range is None:
    #         degree_range = [0, 10000]
    #     for _ in range(num_target):
    #         target_node = random.randint(0, self.num_nodes)
    #         neighbors, _, _, _ = k_hop_subgraph(target_node, 1, self.graph_dataset.edge_index, relabel_nodes=False)
    #         # remove the target node itself
    #         neighbors = neighbors[1:].tolist()
    #         target_node_degree = len(neighbors)
    #         # filtering, find target node
    #         while target_node in self.picked_nodes or target_node_degree < degree_range[0] or target_node_degree > \
    #                 degree_range[1]:
    #             target_node = random.randint(0, self.num_nodes)
    #             neighbors, _, _, _ = k_hop_subgraph(target_node, 1, self.graph_dataset.edge_index, relabel_nodes=False)
    #             # remove the target node itself
    #             neighbors = neighbors[1:].tolist()
    #             target_node_degree = len(neighbors)
    #
    #         self.picked_nodes.add(target_node)
    #
    #         self.construct_centroid_pairs_set_process([target_node], set_size, k_hop, maximum_connected, balanced)
    #     print(f'done sampling, find {len(self.picked_nodes)} target nodes, with degree between {degree_range}')

    # def get_node_centroid_pairs(self, node_id, set_size=50, k_hop=3, maximum_connected=20, twohop=False,
    #                             balanced=False):
    #     """
    #     For a given target node, construct the pair set.
    #
    #     Positive samples: All 1-hop neighbors (up to maximum_connected).
    #     Negative samples:
    #       - If balanced is True: use 2-hop neighbors (excluding the target and its 1-hop neighbors).
    #       - Otherwise, randomly sample negatives from the rest of the nodes (same number as positives).
    #
    #     Args:
    #         node_id (int): The target node.
    #         set_size (int): (Optional) Maximum number of pairs to consider (not used directly here).
    #         k_hop (int): (Optional) Parameter for future expansion (default 3).
    #         maximum_connected (int): Maximum number of 1-hop neighbors to consider.
    #         balanced (bool): Whether to sample negatives based on the 2-hop neighborhood.
    #
    #     Returns:
    #         neighbor (dict): A dictionary with keys 'positive' and 'negative' containing lists of node indices.
    #         num_direct_connection (int): Number of positive neighbors (1-hop neighbors).
    #     """
    #     device = self.graph_dataset.edge_index.device  # assume edge_index is on the desired device
    #
    #     # Get 1-hop subgraph: positive neighbors (using PyG utility)
    #     pos_nodes, _, _, _ = k_hop_subgraph(node_id, 1, self.graph_dataset.edge_index, relabel_nodes=False)
    #     pos_nodes = pos_nodes.tolist()
    #     # Remove the target node itself, if present
    #     if node_id in pos_nodes:
    #         pos_nodes.remove(node_id)
    #     # Optionally limit the number of positives to maximum_connected
    #     positive_neighbors = pos_nodes[:maximum_connected]
    #     num_direct_connection = len(positive_neighbors)
    #
    #     # Negative sampling:
    #     if twohop:
    #         # Get 2-hop subgraph (neighbors within 2 hops)
    #         two_hop_nodes, _, _, _ = k_hop_subgraph(node_id, 2, self.graph_dataset.edge_index, relabel_nodes=False)
    #         two_hop_nodes = two_hop_nodes.tolist()
    #         # Exclude the target node and its 1-hop neighbors
    #         negative_candidates = list(set(two_hop_nodes) - {node_id} - set(positive_neighbors))
    #         # Sample negatives: if there are enough candidates, sample the same number as positives.
    #         if balanced and len(negative_candidates) >= len(positive_neighbors) and len(positive_neighbors) > 0:
    #             negatives = random.sample(negative_candidates, len(positive_neighbors))
    #         else:
    #             negatives = negative_candidates
    #     else:
    #         # Random negatives: sample from all nodes excluding the target and its positives.
    #         all_nodes = set(range(self.num_nodes))
    #         candidate_nodes = list(all_nodes - {node_id} - set(positive_neighbors))
    #         if balanced and len(candidate_nodes) >= len(positive_neighbors) and len(positive_neighbors) > 0:
    #             negatives = random.sample(candidate_nodes, len(positive_neighbors))
    #         else:
    #             negatives = candidate_nodes
    #     print(len(negatives))
    #     # Construct a dictionary containing positive and negative samples.
    #     neighbor = {"positive": positive_neighbors, "negative": negatives}
    #     return neighbor, num_direct_connection

    def get_node_centroid_pairs(self, node_id, graph_dataset=None, set_size=50, k_hop=3, maximum_connected=20,
                                twohop=False,
                                balanced=False):
        """
        For a given target node, construct the pair set.

        Positive samples: All 1-hop neighbors (up to maximum_connected).
        Negative samples:
          - If balanced is True: use 2-hop neighbors (excluding the target and its 1-hop neighbors).
          - Otherwise, randomly sample negatives from the rest of the nodes (same number as positives).

        Args:
            node_id (int): The target node.
            set_size (int): (Optional) Maximum number of pairs to consider (not used directly here).
            k_hop (int): (Optional) Parameter for future expansion (default 3).
            maximum_connected (int): Maximum number of 1-hop neighbors to consider.
            balanced (bool): Whether to sample negatives based on the 2-hop neighborhood.

        Returns:
            neighbor (dict): A dictionary with keys 'positive' and 'negative' containing lists of node indices.
            num_direct_connection (int): Number of positive neighbors (1-hop neighbors).
        """

        # Get 1-hop subgraph: positive neighbors (using PyG utility)
        pos_nodes, _, _, _ = k_hop_subgraph(node_id, 1, graph_dataset.edge_index, relabel_nodes=False)
        pos_nodes = pos_nodes.tolist()
        # Remove the target node itself, if present
        if node_id in pos_nodes:
            pos_nodes.remove(node_id)
        # Optionally limit the number of positives to maximum_connected
        positive_neighbors = pos_nodes[:maximum_connected]
        num_direct_connection = len(positive_neighbors)

        # Negative sampling:
        if twohop:
            print("doing two hop sampling")
            # Get 2-hop subgraph (neighbors within 2 hops)
            two_hop_nodes, _, _, _ = k_hop_subgraph(node_id, 2, graph_dataset.edge_index, relabel_nodes=False)
            two_hop_nodes = two_hop_nodes.tolist()
            # Exclude the target node and its 1-hop neighbors
            negative_candidates = list(set(two_hop_nodes) - {node_id} - set(positive_neighbors))
            # Sample negatives: if there are enough candidates, sample the same number as positives.
            if balanced and len(negative_candidates) >= len(positive_neighbors) and len(positive_neighbors) > 0:
                negatives = random.sample(negative_candidates, len(positive_neighbors))
            else:
                negatives = negative_candidates
        else:
            # Random negatives: sample from all nodes excluding the target and its positives.
            all_nodes = set(range(graph_dataset.num_nodes))
            candidate_nodes = list(all_nodes - {node_id} - set(positive_neighbors))
            if balanced and len(candidate_nodes) >= len(positive_neighbors) and len(positive_neighbors) > 0:
                negatives = random.sample(candidate_nodes, len(positive_neighbors))
            else:
                negatives = candidate_nodes
        print(len(negatives))
        # Construct a dictionary containing positive and negative samples.
        neighbor = {"positive": positive_neighbors, "negative": negatives}
        return neighbor, num_direct_connection

    def construct_centroid_pairs_set_process(self, target_nodes, graph_dataset, set_size=50, k_hop=3,
                                             maximum_connected=20,
                                             balanced=False, twohop=False):
        """
        For each target node in target_nodes, construct the centroid pairs and save them.

        Args:
            target_nodes (list of int): List of target node indices.
            set_size (int): Maximum set size (for potential future use).
            k_hop (int): Parameter (not actively used here, but reserved for future expansion).
            maximum_connected (int): Maximum number of 1-hop neighbors to consider.
            balanced (bool): Determines which negative sampling method to use.
        """
        if k_hop == 2:
            twohop = True
        else:
            twohop = False
        for node_id in target_nodes:
            neighbor, num_connected = self.get_node_centroid_pairs(
                node_id,
                graph_dataset=graph_dataset,
                twohop=twohop,
                maximum_connected=maximum_connected,
                balanced=balanced,
            )
            print(f"target node: {node_id}, pos: {len(neighbor['positive'])}, neg: {len(neighbor['negative'])}")
            self.save_node_pairs(
                {"target_node": node_id, "neighbor": neighbor, "num_direct_connection": num_connected},
                self.dataset_name,
                str(node_id),
                balanced
            )

    def construct_centroid_pairs_set(self, num_target, graph_dataset, set_size=50, k_hop=3, maximum_connected=1000,
                                     degree_range=None,
                                     balanced=False):
        """
        Constructs centroid pairs for a specified number of target nodes.

        Each target node is randomly sampled from the graph subject to a degree filter.
        For each valid target node, its 1-hop neighbors become the positive samples, and
        negative samples are selected according to the 'balanced' flag.

        Args:
            num_target (int): Number of target nodes to sample.
            set_size (int): (Optional) Size parameter for the set (not actively used here).
            k_hop (int): (Optional) Hop count parameter (default 3; currently not used in the sampling).
            maximum_connected (int): Maximum number of direct neighbors (positives) to consider.
            degree_range (list or tuple): [min_degree, max_degree] for a target node to be valid.
            balanced (bool): If True, negative samples are taken as nodes 2 hops away; otherwise, they are random.
        """
        if degree_range is None:
            degree_range = [0, 10000]
        for _ in range(num_target):
            target_node = random.randint(0, self.num_nodes - 1)
            # Get 1-hop neighbors to compute the degree of the target node.
            pos_nodes, _, _, _ = k_hop_subgraph(target_node, 1, graph_dataset.edge_index, relabel_nodes=False)
            pos_nodes = pos_nodes.tolist()
            if target_node in pos_nodes:
                pos_nodes.remove(target_node)
            target_node_degree = len(pos_nodes)
            # Filter: ensure the target node hasn't been picked and meets the degree criteria.
            attempts = 0
            while (target_node in self.picked_nodes or
                   target_node_degree < degree_range[0] or
                   target_node_degree > degree_range[1]) and attempts < 100:
                target_node = random.randint(0, self.num_nodes - 1)
                pos_nodes, _, _, _ = k_hop_subgraph(target_node, 1, graph_dataset.edge_index, relabel_nodes=False)
                pos_nodes = pos_nodes.tolist()
                if target_node in pos_nodes:
                    pos_nodes.remove(target_node)
                target_node_degree = len(pos_nodes)
                attempts += 1
            print(f"current_target: {target_node}")
            if attempts >= 100:
                print("Warning: Could not find a valid target node after 100 attempts.")
                continue

            self.picked_nodes.add(target_node)
            # Process this target node.
            self.construct_centroid_pairs_set_process([target_node], graph_dataset, set_size, k_hop, maximum_connected,
                                                      balanced)
        print(f'Done sampling, found {len(self.picked_nodes)} target nodes, with degree between {degree_range}')

    def save_node_pairs(self, node_pairs, dataset_name='', file_name='', balanced=False):
        if balanced:
            save_path = f'{self.root_path}/node_pairs/{dataset_name}/balanced/{self.sub_path}'
        else:
            save_path = f'{self.root_path}/node_pairs/{dataset_name}/unbalanced/{self.sub_path}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(node_pairs,
                   f'{save_path}/{file_name}')
        print(f'save node pairs successfully to {save_path}')

    def load_node_pairs(self, dataset_name=''):
        file_path = f'{self.root_path}/{dataset_name}'
        file_list = os.listdir(file_path)
        node_pairs = []
        for i in file_list:
            tmp = torch.load(f'{file_path}/{i}')
            node_pairs.append(tmp)
        return node_pairs


def sample_node_pair_with_distance_density(dataset, distance=1, nums=50, degree_min=1, degree_max=5):
    node_pairs = []
    G = to_networkx(dataset, to_undirected=True)
    for i in range(nums):
        node1 = random.randint(0, dataset.x.shape[0] - 1)
        node2 = 0
        dist = None
        try:
            dist = len(shortest_path(G, node1, node2))
        except Exception:
            pass

        while dist == None or dist != (distance + 1) or tuple(sorted([node1, node2])) in node_pairs or not G.degree[
                                                                                                               node1] in range(
            degree_min, degree_max + 1) or not G.degree[node2] in range(degree_min, degree_max + 1):
            node2 += 1
            if node2 == dataset.x.shape[0]:
                node1 = random.randint(0, dataset.x.shape[0] - 1)
                node2 = 0
            try:
                dist = len(shortest_path(G, node1, node2))
            except Exception:
                pass

        new_pair = tuple(sorted([node1, node2]))
        node_pairs.append(new_pair)
    return node_pairs


def random_sample_unconnected_node_pairs(dataset, nums):
    node_pairs = []
    G = to_networkx(dataset, to_undirected=True)
    for i in range(nums):
        node1 = random.randint(0, dataset.x.shape[0] - 1)
        node2 = random.randint(0, dataset.x.shape[0] - 1)
        dist = None
        try:
            dist = len(shortest_path(G, node1, node2))
        except Exception:
            pass
        while dist == 1 or dist == 2 or tuple(sorted([node1, node2])) in node_pairs:
            node2 += 1
            if node2 == dataset.x.shape[0]:
                node1 = random.randint(0, dataset.x.shape[0] - 1)
                node2 = 0
            try:
                dist = len(shortest_path(G, node1, node2))
            except Exception:
                pass

        new_pair = tuple(sorted([node1, node2]))
        node_pairs.append(new_pair)
    return node_pairs


def load_node_pairs(path):
    file_path = path
    file_list = os.listdir(file_path)
    node_pairs = []
    for i in file_list:
        tmp = torch.load(f'{file_path}/{i}')
        node_pairs.append(tmp)
    return node_pairs


def read_node_pairs(dataset_name, mode, twohop=True, is_balanced=False, total_number_limit=50):
    if is_balanced:
        node_pairs = load_node_pairs(f'{SAVE_PATH}/twohop_{twohop}/node_pairs/{dataset_name}/balanced/{mode}')
    else:
        node_pairs = load_node_pairs(f'{SAVE_PATH}/twohop_{twohop}/node_pairs/{dataset_name}/unbalanced/{mode}')
    centroid_node_list = []
    formatted_node_pairs = []
    num_connected_nodes = []
    for i in node_pairs:
        centroid_node = i["target_node"]
        neighbour = i["neighbor"]
        num_direct_connect = i["num_direct_connection"]

        centroid_node_list.append(centroid_node)
        formatted_node_pair = [(centroid_node, node_index) for node_index in neighbour["positive"]] + \
                              [(centroid_node, node_index) for node_index in neighbour["negative"]]
        if num_direct_connect > total_number_limit // 2:
            formatted_node_pair = formatted_node_pair[:2 * num_direct_connect]
        elif len(formatted_node_pair) > total_number_limit:
            formatted_node_pair = formatted_node_pair[:total_number_limit]
        formatted_node_pairs.append(formatted_node_pair)
        num_connected_nodes.append(num_direct_connect)
    return centroid_node_list, formatted_node_pairs, num_connected_nodes
