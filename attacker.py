import copy
import logging
import time

import numpy
import torch
from scipy.spatial.distance import cosine, euclidean, correlation, chebyshev, braycurtis, canberra, cityblock, \
    sqeuclidean
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from utils.attack_utils import evaluate_attack_auc_ap, find_centroid_node, find_mean_node_feature, \
    find_median_node_feature
from utils.graph_utils import remove_target_node, add_new_node_to_graph, add_new_edge_to_graph, \
    construct_k_hop_subgraph, evolve_graph, create_sparse_dissimilar_vector
from utils.node_pair_sampling_utils import read_node_pairs


class Attacker:
    def __init__(self, model, dataset, dataset_name, subgraph_hop=3, is_balanced=False, insert_node_mode="random",
                 target_degree='uncons', defense_type='', clipping_param=3, dynamic=False, influence_rate=1,
                 dynamic_rate=0.01, pipeline=None, insert_node_strategy="same", dynamic_insert_neighbor=False,
                 n_neighborhood_new_node=0.3, dp2_insert_node='candidate', evolving_mode="all", dp_epsilon=1,
                 twohop=True):
        self.model = model.cuda()
        self.pipeline = pipeline
        self.dataset = dataset
        self.dataset_name = dataset_name
        # load target node pairs
        self.target_degree = target_degree
        self.insert_node_strategy = insert_node_strategy
        self.target_nodes, self.node_pairs, self.num_direct_connections = self.load_node_pairs(is_balanced, twohop)
        self.num_attack_nodes = len(self.target_nodes)
        # self.distance_metric_list = [cosine, euclidean, correlation]
        # self.distance_metric_list_name = ['cosine', 'euclidean', 'correlation']
        self.distance_metric_list = [cosine, euclidean, correlation, chebyshev, braycurtis, canberra, cityblock,
                                     sqeuclidean]
        self.distance_metric_list_name = ['cosine', 'euclidean', 'correlation', 'chebyshev', 'braycurtis', 'canberra',
                                          'cityblock', 'sqeuclidean']
        self.num_distance_metric = len(self.distance_metric_list)
        self.subgraph_hop = subgraph_hop
        self.precomputed_subgraph = {}
        self.precomputed_subgraph_mapping = {}
        self.insert_node_feature = self.get_insert_node_feature(insert_node_mode)
        self.insert_node_feature = self.insert_node_feature.cuda()
        self.defense_type = defense_type
        self.clipping_param = clipping_param
        self.dynamic = dynamic
        self.dynamic_rate = dynamic_rate
        self.influence_rate = influence_rate
        self.dynamic_insert_neighbor = dynamic_insert_neighbor
        self.n_neighborhood_new_node = n_neighborhood_new_node
        self.dp2_insert_node = dp2_insert_node
        self.dp_epsilon = dp_epsilon
        self.evolving_mode = evolving_mode

    def get_insert_node_feature(self, insert_node_mode):
        if insert_node_mode == "random":
            # num_nodes = self.dataset.x.shape[0]
            # # Generate a random index using torch.randint.
            # rand_index = torch.randint(0, num_nodes, (1,)).item()
            # Select the corresponding node feature.
            insert_node_feature = self.dataset.x[0]

        elif insert_node_mode == "typical":
            insert_node_feature = self.dataset.x[find_centroid_node(self.dataset.x.cpu().numpy())]
        elif insert_node_mode == "mean":
            insert_node_feature = torch.from_numpy(find_mean_node_feature(self.dataset.x.cpu().numpy()))
        elif insert_node_mode == "median":
            insert_node_feature = torch.from_numpy(find_median_node_feature(self.dataset.x.cpu().numpy()))
        elif insert_node_mode == "target":
            insert_node_feature = -1
        elif insert_node_mode == "zero":
            insert_node_feature = torch.zeros(self.dataset.x.shape[1])
        else:
            insert_node_feature = self.dataset.x[0]
            print("didn't specify node feature sample strategy, take the feature of the node 0")
        return insert_node_feature

    def attack(self, attack_type, attack_node_id):
        self.model.eval()
        t = time.time()
        if attack_type == 'simi_1':
            return self.similarity_attack_1(self.node_pairs[attack_node_id],
                                            self.num_direct_connections[attack_node_id])
        elif attack_type == 'simi_2':
            return self.similarity_attack_2(self.node_pairs[attack_node_id],
                                            self.num_direct_connections[attack_node_id])
        elif attack_type == 'simi_3':
            return self.similarity_attack_3(self.node_pairs[attack_node_id],
                                            self.num_direct_connections[attack_node_id])
        elif attack_type == 'simi_4':
            return self.similarity_attack_4(self.node_pairs[attack_node_id],
                                            self.num_direct_connections[attack_node_id])
        elif attack_type == 'inf_1':
            return self.influence_attack_1(self.node_pairs[attack_node_id],
                                           self.num_direct_connections[attack_node_id])

        elif attack_type == 'inf_2':
            return self.influence_attack_2(self.node_pairs[attack_node_id],
                                           self.num_direct_connections[attack_node_id])

        elif attack_type == 'inf_3':
            return self.influence_attack_3(self.node_pairs[attack_node_id],
                                           self.num_direct_connections[attack_node_id])
        elif attack_type == 'inf_4':
            return self.influence_attack_4(self.node_pairs[attack_node_id],
                                           self.num_direct_connections[attack_node_id])
        elif attack_type == 'lsa_post':
            return self.link_stealing_post(self.node_pairs[attack_node_id],
                                           self.num_direct_connections[attack_node_id])
        elif attack_type == 'lsa_attr':
            return self.link_stealing_attr(self.node_pairs[attack_node_id],
                                           self.num_direct_connections[attack_node_id])
        elif attack_type == 'lta':
            return self.link_teller(self.node_pairs[attack_node_id],
                                    self.num_direct_connections[attack_node_id])
        elif attack_type == 'infiltration':
            return self.infiltration(self.node_pairs[attack_node_id],
                                     self.num_direct_connections[attack_node_id])
        elif attack_type == 'dp1':
            return self.dp1(self.node_pairs[attack_node_id],
                            self.num_direct_connections[attack_node_id])
        elif attack_type == 'dp2':
            return self.dp2(self.node_pairs[attack_node_id],
                            self.num_direct_connections[attack_node_id])
        elif attack_type == 'ltao':
            return self.link_teller_origin(self.node_pairs[attack_node_id],
                                           self.num_direct_connections[attack_node_id])
        elif attack_type == 'test':
            return self.test(self.node_pairs[attack_node_id],
                             self.num_direct_connections[attack_node_id])
        else:
            print("cannot find target attack")

        logging.info(f'done attack {attack_type}, takes {time.time() - t}')

    def load_node_pairs(self, is_balanced, twohop):
        return read_node_pairs(self.dataset_name, mode=self.target_degree, twohop=twohop, is_balanced=is_balanced)

    def get_node_pairs(self):
        return self.target_nodes, self.node_pairs, self.num_direct_connections

    def similarity_attack_1(self, node_pairs, num_connected):

        # get all the model output for the 2 new added nodes.
        all_model_output = []
        statistics = []
        for i in range(len(node_pairs)):
            insert_node_attribute = self.insert_node_feature
            if not torch.is_tensor(self.insert_node_feature):
                insert_node_attribute = self.dataset.x[node_pairs[i][0]]
            output = self.similarity_attack_1_process(node_pairs[i][0], node_pairs[i][1],
                                                      insert_node_attribute=insert_node_attribute)
            all_model_output.append(output[-2:])

        # get the similarity between 2 inserted nodes
        simis = {}
        for output in all_model_output:
            for metrics_name, distanceMetrics in zip(self.distance_metric_list_name, self.distance_metric_list):
                node_1_emb = output[-1].detach().numpy()
                node_2_emb = output[-2].detach().numpy()
                tmp = simis.get(metrics_name, [])
                tmp.append(distanceMetrics(node_1_emb, node_2_emb))
                simis[metrics_name] = tmp
        auc_list = []
        ap_list = []
        for i in range(len(self.distance_metric_list)):
            auc, ap = evaluate_attack_auc_ap(simis[self.distance_metric_list_name[i]][:num_connected],
                                             simis[self.distance_metric_list_name[i]][num_connected:])
            auc_list.append(auc)
            ap_list.append(ap)
        print(f'done evaluating auc: {auc_list}, ap: {ap_list}')
        return [], simis, auc_list

    def similarity_attack_1_process(self, target_node_1, target_node_2, insert_node_attribute):
        # insert 1 node connected to node 1
        subgraph, target_node_1, target_node_2 = self.get_k_hop_subgraph([target_node_1, target_node_2])

        add_new_node_to_graph(subgraph, insert_node_attribute, target_node_1,
                              y=subgraph.y[0])

        # insert 1 node connected to node 2
        add_new_node_to_graph(subgraph, insert_node_attribute, target_node_2,
                              y=subgraph.y[0])

        # output after 2 nodes inserted
        output = self.model.predict_step(subgraph)

        return output

    def similarity_attack_2(self, node_pairs, num_connected):
        statistic_list = {}
        ori_output_list = []
        post_output_list = []
        aucs = []
        n_pairs = len(node_pairs)

        for i in range(n_pairs):
            insert_node_attribute = self.insert_node_feature
            if not torch.is_tensor(self.insert_node_feature):
                insert_node_attribute = self.dataset.x[node_pairs[i][0]]
            ori_output, post_output = self.similarity_attack_2_process(node_pairs[i][0], node_pairs[i][1],
                                                                       insert_node_attribute)
            ori_output_list.append(ori_output)
            post_output_list.append(post_output)
        ori_simi_list = []
        post_simi_list = []

        for i in range(n_pairs):
            ori_simi_list.append(self.calculate_similarity(ori_output_list[i][0], ori_output_list[i][1]))
            post_simi_list.append(self.calculate_similarity(post_output_list[i][0], post_output_list[i][1]))

        model_output_list = [ori_simi_list, post_simi_list]
        for i in range(n_pairs):
            for j in range(self.num_distance_metric):
                current_dmeteic = self.distance_metric_list_name[j]
                so = ori_simi_list[i][current_dmeteic][0]
                sp = post_simi_list[i][current_dmeteic][0]

                tmp = statistic_list.get(current_dmeteic, [])

                tmp.append(sp - so)
                statistic_list[current_dmeteic] = tmp

        for i in self.distance_metric_list_name:
            auc, ap = evaluate_attack_auc_ap(statistic_list[i][:num_connected],
                                             statistic_list[i][num_connected:])
            aucs.append(auc)

        return model_output_list, statistic_list, aucs

    def similarity_attack_2_process(self, target_node_1, target_node_2, insert_node_attribute):
        # insert 2 node connected to node 1
        attack_dataset = self.dataset.clone()

        add_new_node_to_graph(attack_dataset, insert_node_attribute, target_node_1,
                              y=attack_dataset.y[0])
        add_new_node_to_graph(attack_dataset, insert_node_attribute, target_node_2,
                              y=attack_dataset.y[0])

        insert_node_1_ind = attack_dataset.x.shape[0] - 2
        insert_node_2_ind = attack_dataset.x.shape[0] - 1

        ori_output = self.model.predict_step(attack_dataset)

        # perturbation added to the node feature
        # perturb the node feature
        add_new_edge_to_graph(attack_dataset, target_node_1, insert_node_2_ind)
        post_change_output = self.model.predict_step(attack_dataset)

        return ori_output[-2:], post_change_output[-2:]

    # def similarity_attack_3(self, node_pairs, num_connected):
    #     statistic_list = {}
    #     ori_output_list = []
    #     post_output_list = []
    #     aucs = []
    #     n_pairs = len(node_pairs)
    #
    #     for i in range(n_pairs):
    #         insert_node_attribute = self.insert_node_feature
    #         if not torch.is_tensor(self.insert_node_feature):
    #             insert_node_attribute = self.dataset.x[node_pairs[i][0]]
    #         ori_output, post_output = self.similarity_attack_3_process(node_pairs[i][0], node_pairs[i][1],
    #                                                                    insert_node_attribute)
    #         ori_output_list.append(ori_output)
    #         post_output_list.append(post_output)
    #     ori_simi_list = []
    #     post_simi_list = []
    #
    #     for i in range(n_pairs):
    #         ori_simi_list.append(self.calculate_similarity(ori_output_list[i][0], ori_output_list[i][1]))
    #         post_simi_list.append(self.calculate_similarity(post_output_list[i][0], post_output_list[i][1]))
    #
    #     model_output_list = [ori_simi_list, post_simi_list]
    #     for i in range(n_pairs):
    #         for j in range(self.num_distance_metric):
    #             current_dmeteic = self.distance_metric_list_name[j]
    #             so = ori_simi_list[i][current_dmeteic][0]
    #             sp = post_simi_list[i][current_dmeteic][0]
    #
    #             tmp = statistic_list.get(current_dmeteic, [])
    #
    #             tmp.append(sp - so)
    #             statistic_list[current_dmeteic] = tmp
    #
    #     for i in self.distance_metric_list_name:
    #         auc, ap = evaluate_attack_auc_ap(statistic_list[i][:num_connected],
    #                                          statistic_list[i][num_connected:])
    #         aucs.append(auc)
    #
    #     return model_output_list, statistic_list, aucs

    def similarity_attack_3(self, node_pairs, num_connected):
        statistic_list = {}
        ori_output_list = []
        post_output_list = []
        aucs = []
        n_pairs = len(node_pairs)

        for i in range(len(node_pairs)):
            ori_output, post_output = self.similarity_attack_3_process(node_pairs[i][0], node_pairs[i][1], 0)
            ori_output_list.append(ori_output)
            post_output_list.append(post_output)

        ori_simi_list = []
        post_simi_list = []

        for i in range(n_pairs):
            ori_simi_list.append(self.calculate_similarity(ori_output_list[i][0], ori_output_list[i][1]))
            post_simi_list.append(self.calculate_similarity(post_output_list[i][0], post_output_list[i][1]))

        model_output_list = [ori_simi_list, post_simi_list]
        for i in range(n_pairs):
            for metrics_name in self.distance_metric_list_name:
                so = ori_simi_list[i][metrics_name]
                sp = post_simi_list[i][metrics_name]
                if so > sp:
                    tmp = statistic_list.get(metrics_name, [])
                    tmp.append(sp - so)
                    statistic_list[metrics_name] = tmp
                else:
                    tmp = statistic_list.get(metrics_name, [])
                    tmp.append(-100)
                    statistic_list[metrics_name] = tmp

        for i in range(self.num_distance_metric):
            auc, _ = evaluate_attack_auc_ap(statistic_list[self.distance_metric_list_name[i]][:num_connected],
                                            statistic_list[self.distance_metric_list_name[i]][num_connected:])
            aucs.append(auc)
        return model_output_list, statistic_list, aucs

    def similarity_attack_3_process(self, target_node_1, target_node_2, insert_node_attribute=0):
        attack_dataset = self.dataset.clone()

        add_new_node_to_graph(attack_dataset, attack_dataset.x[insert_node_attribute], target_node_1,
                              y=attack_dataset.y[insert_node_attribute])
        add_new_node_to_graph(attack_dataset, attack_dataset.x[insert_node_attribute], target_node_2,
                              y=attack_dataset.y[insert_node_attribute])

        add_new_node_to_graph(attack_dataset, attack_dataset.x[insert_node_attribute], target_node_1,
                              y=attack_dataset.y[insert_node_attribute])

        insert_node_1_ind = attack_dataset.x.shape[0] - 3
        insert_node_2_ind = attack_dataset.x.shape[0] - 2
        insert_node_extra_ind = attack_dataset.x.shape[0] - 1

        ori_output = self.model.predict_step(attack_dataset)

        add_new_edge_to_graph(attack_dataset, insert_node_1_ind, insert_node_2_ind)

        post_output = self.model.predict_step(attack_dataset)

        return ori_output[-3:], post_output[-3:]

    def similarity_attack_4(self, node_pairs, num_connected):
        ori_output_list = []
        post_output_list = []
        for i in range(len(node_pairs)):
            ori_output, post_output = self.similarity_attack_4_process(node_pairs[i][0], node_pairs[i][1])
            ori_output_list.append(ori_output[-2:])
            post_output_list.append(post_output[-2:])

        print('done getting model outputs')

        return ori_output_list, post_output_list

    def similarity_attack_4_process(self, target_node_1, target_node_2, insert_node_attribute=0):
        # insert 2 node connected to node 1
        add_new_node_to_graph(self.dataset, self.dataset.x[insert_node_attribute], target_node_1,
                              y=self.dataset.y[insert_node_attribute])

        # insert 1 node connected to node 2
        add_new_node_to_graph(self.dataset, self.dataset.x[insert_node_attribute], target_node_2,
                              y=self.dataset.y[insert_node_attribute])

        # perturbation to the node feature
        insert_node_1_ind = self.dataset.x.shape[0] - 2
        insert_node_2_ind = self.dataset.x.shape[0] - 1

        ori_output = self.model.predict_step(self.dataset)

        # perturbation added to the node feature
        # perturb the node feature
        add_new_edge_to_graph(self.dataset, insert_node_1_ind, target_node_2)
        post_output = self.model.predict_step(self.dataset)

        # delete inserted nodes
        remove_target_node(self.dataset, self.dataset.x.shape[0] - 1)
        remove_target_node(self.dataset, self.dataset.x.shape[0] - 1)

        return ori_output, post_output

    def influence_attack_1(self, node_pairs, num_connected):
        influence_score_list = []
        model_output_list = []
        for i in range(len(node_pairs)):
            insert_node_attribute = self.insert_node_feature
            if not torch.is_tensor(self.insert_node_feature):
                insert_node_attribute = self.dataset.x[node_pairs[i][0]]

            inf_score, origin_output, pert_output = self.influence_attack_1_process(node_pairs[i][0], node_pairs[i][1],
                                                                                    insert_node_attribute=insert_node_attribute)
            influence_score_list.append(inf_score)
            # model_output_list.append((origin_output[-2:], pert_output[-2:]))
        auc, ap = evaluate_attack_auc_ap(influence_score_list[:num_connected], influence_score_list[num_connected:])
        return model_output_list, influence_score_list, auc

    def influence_attack_1_process(self, target_node_1, target_node_2, insert_node_attribute, influence_rate=0.0001):
        # insert 2 node connected to node 1
        attack_dataset = self.dataset.clone()
        add_new_node_to_graph(attack_dataset, insert_node_attribute, target_node_1,
                              y=attack_dataset.y[0])
        add_new_node_to_graph(attack_dataset, insert_node_attribute, target_node_2,
                              y=attack_dataset.y[0])

        # perturbation to the node feature
        insert_node_1_ind = attack_dataset.x.shape[0] - 2
        insert_node_2_ind = attack_dataset.x.shape[0] - 1

        influence_matrix, ori_output, pert_output = self.get_influence_score(attack_dataset, insert_node_1_ind,
                                                                             influence_rate,
                                                                             [insert_node_1_ind, insert_node_2_ind])

        # get the influence of target node on all other nodes
        influence_score = influence_matrix[insert_node_2_ind].norm().item()
        return influence_score, ori_output, pert_output

    def get_influence_score(self, attack_dataset, target_node, influence=0.1, static_nodes=[]):
        ori_output = self.model.predict_step(attack_dataset)
        # perturb the node feature
        pert = attack_dataset.x[target_node] * influence
        attack_dataset.x[target_node] = attack_dataset.x[target_node] + pert
        if self.dynamic:
            if self.dynamic_insert_neighbor:
                evolve_graph(attack_dataset, self.dynamic_rate, self.dynamic_rate, self.dynamic_rate, static_nodes,
                             target_node, self.n_neighborhood_new_node, evolving_mode=self.evolving_mode)
            else:
                evolve_graph(attack_dataset, self.dynamic_rate, self.dynamic_rate, self.dynamic_rate, static_nodes,
                             evolving_mode=self.evolving_mode)
        pert_output = self.model.predict_step(attack_dataset)
        grad = (pert_output.detach()[:ori_output.shape[0]] - ori_output.detach()) / influence
        return grad, ori_output, pert_output

    def influence_attack_2(self, node_pairs, num_connected):
        influence_score_list_1 = []
        influence_score_list_2 = []
        influence_score_list_3 = []
        for i in range(len(node_pairs)):
            insert_node_attribute = self.insert_node_feature
            if not torch.is_tensor(self.insert_node_feature):
                insert_node_attribute = self.dataset.x[node_pairs[i][0]]

            influence_score_2, influence_score_3 = self.influence_attack_2_process(node_pairs[i][0], node_pairs[i][1],
                                                                                   insert_node_attribute=insert_node_attribute)
            influence_score_list_1.append(
                euclidean(influence_score_2.detach().numpy(), influence_score_3.detach().numpy()))
            influence_score_list_2.append(
                cosine(influence_score_2.detach().numpy(), influence_score_3.detach().numpy()))
            influence_score_list_3.append(
                correlation(influence_score_2.detach().numpy(), influence_score_3.detach().numpy()))
        evaluate_attack_auc_ap(influence_score_list_1[:num_connected], influence_score_list_1[num_connected:])
        evaluate_attack_auc_ap(influence_score_list_2[:num_connected], influence_score_list_2[num_connected:])
        evaluate_attack_auc_ap(influence_score_list_3[:num_connected], influence_score_list_3[num_connected:])

        return influence_score_list_1, influence_score_list_2, influence_score_list_3

    def influence_attack_2_process(self, target_node_1, target_node_2, insert_node_attribute=0, influence_rate=0.1):
        # insert 2 node connected to node 1
        attack_dataset = self.dataset.clone()
        evolve_graph(attack_dataset, self.dynamic_rate, self.dynamic_rate, self.dynamic_rate,
                     evolving_mode=self.evolving_mode)
        add_new_node_to_graph(attack_dataset, attack_dataset.x[insert_node_attribute], target_node_1,
                              y=attack_dataset.y[insert_node_attribute])
        add_new_node_to_graph(attack_dataset, attack_dataset.x[insert_node_attribute], target_node_2,
                              y=attack_dataset.y[insert_node_attribute])

        add_new_node_to_graph(attack_dataset, attack_dataset.x[insert_node_attribute], target_node_1,
                              y=attack_dataset.y[insert_node_attribute])
        # perturbation to the node feature

        insert_node_1_ind = attack_dataset.x.shape[0] - 3
        insert_node_2_ind = attack_dataset.x.shape[0] - 2
        insert_node_3_ind = attack_dataset.x.shape[0] - 1

        influence_matrix, ori_output, pert_output = self.get_influence_score(attack_dataset, insert_node_1_ind,
                                                                             influence_rate,
                                                                             [insert_node_1_ind, insert_node_2_ind,
                                                                              insert_node_3_ind])

        # get the influence of target node on all other nodes
        influence_score_2 = influence_matrix[insert_node_2_ind]
        influence_score_3 = influence_matrix[insert_node_3_ind]

        return influence_score_2, influence_score_3

    def influence_attack_3(self, node_pairs, num_connected):
        influence_score_list_processed = []
        influence_score_list_raw = []
        model_outputs_list = []
        num_nodes = self.dataset.x.shape[0]
        # Generate a random index using torch.randint.
        rand_index = torch.randint(0, num_nodes, (1,)).item()
        # Select the corresponding node feature.

        for i in range(len(node_pairs)):
            insert_node_attribute = self.insert_node_feature
            # if not torch.is_tensor(self.insert_node_feature):
            #     insert_node_attribute = self.dataset.x[node_pairs[i][0]]
            influence_score_1, influence_score_3, ori_output, pert_output = self.influence_attack_3_process(
                node_pairs[i][0], node_pairs[i][1], insert_node_attribute=insert_node_attribute,
                influence_rate=self.influence_rate, rand_index=rand_index)
            if influence_score_3 != 0:
                influence_score_list_processed.append(influence_score_1 / influence_score_3)
            else:
                influence_score_list_processed.append(influence_score_1)
            influence_score_list_raw.append((influence_score_1, influence_score_3))
        auc, ap = evaluate_attack_auc_ap(influence_score_list_processed[:num_connected],
                                         influence_score_list_processed[num_connected:])
        return model_outputs_list, influence_score_list_raw, auc

    def influence_attack_3_process(self, target_node_1, target_node_2, insert_node_attribute, influence_rate=0.1,
                                   rand_index=0):
        # insert 2 node connected to node 1
        # self.insert_node_strategy = "inf3"
        # insert_perturb_node_feature, insert_anchor_node_feature, insert_target_node_feature = self.get_insert_node_strategy(
        #     rand_index, 3)
        attack_dataset = self.dataset.clone()
        add_new_node_to_graph(attack_dataset, insert_node_attribute, target_node_1,
                              y=attack_dataset.y[0])
        add_new_node_to_graph(attack_dataset, insert_node_attribute, target_node_2,
                              y=attack_dataset.y[0])

        add_new_node_to_graph(attack_dataset, insert_node_attribute, target_node_2,
                              y=attack_dataset.y[0])

        insert_node_1_ind = attack_dataset.x.shape[0] - 3
        insert_node_2_ind = attack_dataset.x.shape[0] - 2
        insert_node_3_ind = attack_dataset.x.shape[0] - 1

        influence_matrix, ori_output, pert_output = self.get_influence_score(attack_dataset, insert_node_2_ind,
                                                                             influence_rate,
                                                                             [insert_node_1_ind, insert_node_2_ind,
                                                                              insert_node_3_ind])

        # get the influence of target node on all other nodes
        if self.defense_type == 'output_clipping':
            values1, _ = torch.topk(influence_matrix[insert_node_1_ind], self.clipping_param)
            values2, _ = torch.topk(influence_matrix[insert_node_3_ind], self.clipping_param)
            influence_score_1 = values1.norm().item()
            influence_score_3 = values2.norm().item()
        else:
            influence_score_1 = influence_matrix[insert_node_1_ind].norm().item()
            influence_score_3 = influence_matrix[insert_node_3_ind].norm().item()

        return influence_score_1, influence_score_3, ori_output, pert_output

    def infiltration(self, node_pairs, num_connected):
        influence_score_list = []
        model_output_list = []
        insert_node_feature = torch.zeros(self.dataset.x.shape[1])
        for i in range(len(node_pairs)):
            inf_score = self.infiltration_process(node_pairs[i][0], node_pairs[i][1],
                                                  insert_node_attribute=insert_node_feature)
            influence_score_list.append(inf_score)
        auc, ap = evaluate_attack_auc_ap(influence_score_list[:num_connected], influence_score_list[num_connected:])
        return model_output_list, influence_score_list, auc

    def infiltration_process(self, target_node_1, target_node_2, insert_node_attribute):
        # insert 2 node connected to node 1
        attack_dataset = self.dataset.clone()

        add_new_node_to_graph(attack_dataset, insert_node_attribute, target_node_1,
                              y=attack_dataset.y[0])
        ori_output = self.model.predict_step(attack_dataset)
        anchor = attack_dataset.x.shape[0] - 1

        if self.dynamic:
            if self.dynamic_insert_neighbor:
                evolve_graph(attack_dataset, self.dynamic_rate, self.dynamic_rate, self.dynamic_rate, [anchor],
                             target_node_1, self.n_neighborhood_new_node, evolving_mode=self.evolving_mode)
            else:
                evolve_graph(attack_dataset, self.dynamic_rate, self.dynamic_rate, self.dynamic_rate, [anchor],
                             evolving_mode=self.evolving_mode)

        add_new_node_to_graph(attack_dataset, insert_node_attribute, target_node_2,
                              y=attack_dataset.y[0])
        pert_output = self.model.predict_step(attack_dataset)

        influence_matrix = pert_output.detach()[anchor] - ori_output.detach()[anchor]
        influence_score = influence_matrix.norm().item()
        return influence_score

    def get_k_hop_subgraph(self, target_nodes):
        subgraph, mapping = construct_k_hop_subgraph(self.dataset, target_nodes,
                                                     k_hop=self.subgraph_hop)
        return subgraph, mapping[0], mapping[1]

    def influence_attack_4(self, node_pairs, num_connected):
        statistic_list_processed = {}
        statistic_list_raw = []
        for i in range(len(node_pairs)):
            insert_node_attribute = self.insert_node_feature
            if not torch.is_tensor(self.insert_node_feature):
                insert_node_attribute = self.dataset.x[node_pairs[i][0]]

            influence_similarity, ori_output, pert_output = self.influence_attack_4_process(
                node_pairs[i][0], node_pairs[i][1], insert_node_attribute, self.influence_rate)
            statistic_list_raw.append(influence_similarity)
        for inf in statistic_list_raw:
            for metric_name in self.distance_metric_list_name:
                tmp = statistic_list_processed.get(metric_name, [])
                # greater the distance greater less likely the nodes are close to each other.
                tmp.append(-inf[metric_name][0])
                statistic_list_processed[metric_name] = tmp
        aucs = []
        return statistic_list_raw, statistic_list_processed, aucs

    def influence_attack_4_process(self, target_node_1, target_node_2, insert_node_attribute, influence_rate=0.1):
        # insert 2 node connected to node 1
        attack_dataset = self.dataset.clone()

        add_new_node_to_graph(attack_dataset, insert_node_attribute, target_node_1,
                              y=attack_dataset.y[0])
        add_new_node_to_graph(attack_dataset, insert_node_attribute, target_node_2,
                              y=attack_dataset.y[0])

        add_new_node_to_graph(attack_dataset, insert_node_attribute, target_node_2,
                              y=attack_dataset.y[0])

        insert_node_1_ind = attack_dataset.x.shape[0] - 3
        insert_node_2_ind = attack_dataset.x.shape[0] - 2
        insert_node_3_ind = attack_dataset.x.shape[0] - 1

        influence_matrix, ori_output, pert_output = self.get_influence_score(attack_dataset, insert_node_2_ind,
                                                                             influence_rate,
                                                                             [insert_node_1_ind, insert_node_2_ind,
                                                                              insert_node_3_ind])

        # get the influence of target node on all other nodes
        influence_score_1 = influence_matrix[insert_node_1_ind]
        influence_score_3 = influence_matrix[insert_node_3_ind]

        influence_similarity = self.calculate_similarity(influence_score_1, influence_score_3)

        return influence_similarity, ori_output, pert_output

    def dp1(self, node_pairs, num_connected, observe_len=15):
        statistic_list_processed = {}
        statistic_list_raw = {}
        target_change_all_list = []
        candidate_change_all_list = []
        for i in range(len(node_pairs)):
            insert_node_attribute = self.insert_node_feature
            if not torch.is_tensor(self.insert_node_feature):
                insert_node_attribute = self.dataset.x[node_pairs[i][0]]
            target_node = node_pairs[i][0]
            candidate_node = node_pairs[i][1]
            attack_dataset = self.dataset.clone()
            add_new_node_to_graph(attack_dataset, insert_node_attribute, target_node,
                                  y=attack_dataset.y[0])
            add_new_node_to_graph(attack_dataset, insert_node_attribute, candidate_node,
                                  y=attack_dataset.y[0])

            a_1_index = attack_dataset.x.shape[0] - 2
            a_2_index = attack_dataset.x.shape[0] - 1
            target_change_all = []
            candidate_change_all = []
            original_output = self.model.predict_step(attack_dataset)
            for round in range(observe_len):
                evolve_graph(attack_dataset, self.dynamic_rate, self.dynamic_rate, self.dynamic_rate,
                             evolving_mode=self.evolving_mode)
                pert_output = self.model.predict_step(attack_dataset)

                target_change_all.append((pert_output[a_1_index] - original_output[a_1_index]).detach().cpu().numpy())
                candidate_change_all.append(
                    (pert_output[a_2_index] - original_output[a_2_index]).detach().cpu().numpy())
            target_change_all_list.append(target_change_all)
            candidate_change_all_list.append(candidate_change_all)
        statistic_list_raw["target"] = target_change_all_list
        statistic_list_raw["candidate"] = candidate_change_all_list
        aucs = []
        return statistic_list_raw, statistic_list_processed, aucs

    def dp2(self, node_pairs, num_connected, observe_len=15):
        statistic_list_processed = {}
        statistic_list_raw = {}
        target_change_all_list = []
        candidate_change_all_list = []
        anchor_change_all_list = []
        for i in range(len(node_pairs)):
            target_node = node_pairs[i][0]
            candidate_node = node_pairs[i][1]
            if self.dp2_insert_node == 'candidate':
                in_node = candidate_node
            elif self.dp2_insert_node == 'target':
                in_node = target_node
            else:
                in_node = 0
            insert_perturb_node_feature, insert_anchor_node_feature, insert_target_node_feature = self.get_insert_node_strategy(
                in_node, 3)
            attack_dataset = self.dataset.clone()
            add_new_node_to_graph(attack_dataset, insert_target_node_feature, target_node,
                                  y=attack_dataset.y[0])
            add_new_node_to_graph(attack_dataset, insert_perturb_node_feature, candidate_node,
                                  y=attack_dataset.y[0])
            add_new_node_to_graph(attack_dataset, insert_anchor_node_feature, candidate_node,
                                  y=attack_dataset.y[0])

            a_t_index = attack_dataset.x.shape[0] - 3
            a_c_perturb_index = attack_dataset.x.shape[0] - 2
            a_c_anchor_index = attack_dataset.x.shape[0] - 1
            target_change_all = []
            candidate_change_all = []
            anchor_change_all = []
            original_output = self.model.predict_step(attack_dataset)
            for round in range(observe_len):
                if self.dynamic:
                    if self.dynamic_insert_neighbor:
                        evolve_graph(attack_dataset, self.dynamic_rate, self.dynamic_rate, self.dynamic_rate,
                                     [a_t_index, a_c_perturb_index, a_c_anchor_index],
                                     target_node, self.n_neighborhood_new_node, evolving_mode=self.evolving_mode,
                                     dp_epsilon=self.dp_epsilon)
                    else:
                        evolve_graph(attack_dataset, self.dynamic_rate, self.dynamic_rate, self.dynamic_rate,
                                     [a_t_index, a_c_perturb_index, a_c_anchor_index], evolving_mode=self.evolving_mode,
                                     dp_epsilon=self.dp_epsilon)
                pert = attack_dataset.x[target_node] * self.influence_rate
                # if round % 2 == 0:
                #     attack_dataset.x[target_node] = attack_dataset.x[target_node] - pert
                # else:
                attack_dataset.x[target_node] = attack_dataset.x[target_node] + pert
                pert_output = self.model.predict_step(attack_dataset)
                target_change_all.append((pert_output[a_t_index] - original_output[a_t_index]).detach().cpu().numpy())
                candidate_change_all.append(
                    (pert_output[a_c_perturb_index] - original_output[a_c_perturb_index]).detach().cpu().numpy())
                anchor_change_all.append(
                    (pert_output[a_c_anchor_index] - original_output[a_c_anchor_index]).detach().cpu().numpy())
            target_change_all_list.append(target_change_all)
            candidate_change_all_list.append(candidate_change_all)
            anchor_change_all_list.append(anchor_change_all)
        statistic_list_raw["target"] = target_change_all_list
        statistic_list_raw["perturbed"] = candidate_change_all_list
        statistic_list_raw["anchor"] = anchor_change_all_list
        aucs = []
        return statistic_list_raw, statistic_list_processed, aucs

    def calculate_similarity(self, a, b):
        # get the similarity between 2 inserted nodes
        simis = {}
        for metrics_name, distanceMetrics in zip(self.distance_metric_list_name, self.distance_metric_list):
            node_1_emb = a.cpu().detach().numpy()
            node_2_emb = b.cpu().detach().numpy()
            if numpy.sum(node_1_emb) == 0 or numpy.sum(node_2_emb) == 0:
                dist = 100
            else:
                dist = distanceMetrics(node_1_emb, node_2_emb)
            tmp = simis.get(metrics_name, [])
            tmp.append(dist)
            simis[metrics_name] = tmp
        return simis

    # process the saved similarity info
    def process_similarity(self, raw_similarity, num_connected):
        processed_similarity = [[] for _ in range(self.num_distance_metric)]
        for simi in raw_similarity:
            for i in range(self.num_distance_metric):
                processed_similarity[i].append(simi[i])

        aucs = []
        for i in range(self.num_distance_metric):
            auc, ap = evaluate_attack_auc_ap(processed_similarity[i][:num_connected],
                                             processed_similarity[i][num_connected:])
            aucs.append(auc)
        return aucs

    def link_teller(self, node_pairs, num_connected, influence_rate=0.0001):
        attack_dataset = self.dataset.clone()
        original_data = []
        influence_list = []
        target_node = node_pairs[0][0]
        influence_matrix, ori_output, pert_output = self.get_influence_score(attack_dataset, target_node,
                                                                             influence_rate,
                                                                             [node_pairs[0][0]])

        for i in node_pairs:
            influence_list.append(influence_matrix[i[1]].norm().item())
        auc, _ = evaluate_attack_auc_ap(influence_list[:num_connected], influence_list[num_connected:])
        return original_data, influence_list, auc

    def test(self, node_pairs, num_connected, influence_rate=0.0001):
        attack_dataset = self.dataset.clone()
        original_data = []
        influence_list = []
        target_node = node_pairs[0][0]
        influence_matrix, ori_output, pert_output = self.get_influence_score_t(attack_dataset, target_node,
                                                                               influence_rate,
                                                                               [node_pairs[0][0]])

        for i in node_pairs:
            influence_list.append(influence_matrix[i[1]].norm().item())
        auc, _ = evaluate_attack_auc_ap(influence_list[:num_connected], influence_list[num_connected:])
        return original_data, influence_list, auc

    def get_influence_score_t(self, attack_dataset, target_node, influence=0.1, static_nodes=[]):
        self.model.eval()  # Ensure evaluation mode

        attack_dataset_copy = copy.deepcopy(attack_dataset)  # Prevent dataset modifications

        with torch.no_grad():  # Disable gradient tracking

            ori_output = self.model.predict_step(attack_dataset_copy)
            hh = hash(str(attack_dataset_copy))  # The hash should remain the same

            pert_output = self.model.predict_step(attack_dataset_copy)
            hhp = hash(str(attack_dataset_copy))  # The hash should remain the same
        diff = (ori_output - pert_output).abs().max()
        grad = pert_output.detach()[:ori_output.shape[0]] - ori_output.detach()
        return grad, ori_output, pert_output

    def link_stealing_post(self, node_pairs, num_connected):
        attack_dataset = self.dataset.clone()

        ori_output = self.model.predict_step(attack_dataset).detach().numpy()

        target_posterior_list = []
        for v, u in node_pairs:
            target_posterior_list.append((ori_output[v], ori_output[u]))

        simis, auc_list = self.get_similarity(target_posterior_list, num_connected)
        return [], simis, auc_list

    def get_similarity(self, target_posterior_list, num_connected):
        simis = {}
        for output in target_posterior_list:
            for metrics_name, distanceMetrics in zip(self.distance_metric_list_name, self.distance_metric_list):
                node_1_emb = output[-1]
                node_2_emb = output[-2]
                tmp = simis.get(metrics_name, [])
                tmp.append(distanceMetrics(node_1_emb, node_2_emb))
                simis[metrics_name] = tmp

        auc_list = []
        ap_list = []
        for i in range(len(self.distance_metric_list)):
            auc, ap = evaluate_attack_auc_ap(simis[self.distance_metric_list_name[i]][:num_connected],
                                             simis[self.distance_metric_list_name[i]][num_connected:])
            auc_list.append(auc)
            ap_list.append(ap)
        return simis, auc_list

    def link_stealing_attr(self, node_pairs, num_connected):
        attack_dataset = self.dataset.clone()
        feature_pairs = []
        for v, u in node_pairs:
            feature_pairs.append((attack_dataset.x[v].detach().numpy(), attack_dataset.x[u].detach().numpy()))

        simis, auc_list = self.get_similarity(feature_pairs, num_connected)

        return [], simis, auc_list

    def get_insert_node_strategy(self, node_index, num_insert_nodes):
        if self.insert_node_strategy == "same":
            return [self.dataset.x[node_index]] * num_insert_nodes
        elif self.insert_node_strategy == "diff1":
            diff_feature = create_sparse_dissimilar_vector(self.dataset.x[node_index])
            return [self.dataset.x[node_index], diff_feature, diff_feature]
        elif self.insert_node_strategy == "diff2":
            diff_feature = create_sparse_dissimilar_vector(self.dataset.x[node_index])
            return [self.dataset.x[node_index], diff_feature, self.dataset.x[node_index]]
        elif self.insert_node_strategy == "randsame":
            diff_feature = create_sparse_dissimilar_vector(self.dataset.x[node_index])
            return [self.dataset.x[0]] * num_insert_nodes
        elif self.insert_node_strategy == "inf3":
            return [self.dataset.x[node_index], self.dataset.x[node_index - 1 if node_index > 0 else node_index],
                    self.dataset.x[node_index]]
        return [self.dataset.x[0]] * num_insert_nodes

    # def link_teller(self, node_pairs, num_connected, influence_rate=0.0001):
    #     attack_dataset = self.dataset.clone()
    #     original_data = []
    #     influence_list = []
    #     target_node = node_pairs[0][0]
    #     influence_matrix, ori_output, pert_output = self.get_influence_score(attack_dataset, target_node,
    #                                                                          influence_rate,
    #                                                                          [node_pairs[0][0]])
    #
    #     for i in node_pairs:
    #         influence_list.append(influence_matrix[i[1]].norm().item())
    #     auc, _ = evaluate_attack_auc_ap(influence_list[:num_connected], influence_list[num_connected:])
    #     return original_data, influence_list, auc

    def link_teller_origin(self, node_pairs, influence_rate=0.1):
        attack_dataset = self.dataset.clone()

        sub_data, updated_pairs = get_subgraph_from_node_pairs(attack_dataset, node_pairs)
        # Evaluate influence-based edge prediction.
        print(f"current subgraph size: {sub_data.num_nodes}")
        auc, f1 = evaluate_influence_edge_prediction(
            self.model, sub_data, updated_pairs, perturb_factor=influence_rate
        )
        return auc, f1


def get_subgraph_from_node_pairs(data, node_pairs):
    """
    Given a list of node pairs, extract the subgraph induced by all nodes in those pairs,
    and return both the subgraph and the updated node pairs in subgraph indexing.

    Args:
        data (Data): A PyG Data object (expects data.edge_index, data.x, optionally data.edge_attr).
        node_pairs (list[tuple]): A list of (u, v) node pairs (original/global node indices).

    Returns:
        sub_data (Data): The induced subgraph as a PyG Data object.
            - sub_data.n_id holds the original/global node indices.
        updated_pairs (list[tuple]): The same node pairs, but now referenced by new (local) subgraph indices.
    """
    # 1) Collect unique node IDs that appear in the node pairs
    nodes = list({n for (u, v) in node_pairs for n in (u, v)})
    nodes_sorted = sorted(nodes)  # optional: not strictly needed, but can be cleaner

    # 2) Extract subgraph using PyG’s `subgraph` utility
    #    This will relabel the nodes in 'nodes_sorted' to [0..len(nodes_sorted)-1].
    edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
    sub_edge_index, sub_edge_attr = subgraph(
        nodes_sorted,
        data.edge_index,
        edge_attr=edge_attr,
        relabel_nodes=True,
        num_nodes=data.num_nodes
    )

    # 3) Extract node features (if any)
    sub_x = data.x[nodes_sorted] if hasattr(data, 'x') and data.x is not None else None

    # 4) Create a new Data object
    sub_data = Data(
        x=sub_x,
        edge_index=sub_edge_index,
        edge_attr=sub_edge_attr
    )
    # Store the original/global node IDs in sub_data.n_id for reference
    sub_data.n_id = torch.tensor(nodes_sorted, dtype=torch.long)
    sub_data.num_nodes = len(nodes_sorted)

    # 5) Update node_pairs to local subgraph indices
    #    The 'mapping' tensor tells us where each original ID appears in the subgraph.
    updated_pairs = []
    # We create a dict to map original node -> local index for efficiency
    # (mapping[i] gives local index of nodes_sorted[i], so we do the inverse).
    old_to_sub = {}
    for local_idx, old_idx in enumerate(nodes_sorted):
        old_to_sub[old_idx] = local_idx

    for (u, v) in node_pairs:
        # Convert the original indices (u, v) to subgraph indices
        u_sub = old_to_sub[u]
        v_sub = old_to_sub[v]
        updated_pairs.append((u_sub, v_sub))

    return sub_data, updated_pairs


def evaluate_influence_edge_prediction(
        model,
        sub_data,
        node_pairs_interest,
        perturb_factor=0.1,
        device='cpu'
):
    """
    Evaluate an influence-based edge prediction method on a subgraph, then
    compute evaluation metrics only for a provided list of node pairs.

    Steps:
      1. Move model and sub_data to the same device.
      2. Compute baseline model output on sub_data.x.
      3. For each node in sub_data, perturb its features (e.g. scale by (1+perturb_factor))
         and compute the sum of L2-norm differences over all nodes as its "influence" score.
      4. For each provided node pair (u,v), define its continuous score as influence[u] + influence[v].
      5. Compute the ground truth edge set for the subgraph (using sub_data.edge_index, as undirected edges).
      6. Let k = (number of ground truth edges in the subgraph).
      7. Label the provided node pairs by sorting them by score (descending) and assigning label 1 to
         the top k pairs and 0 to the rest.
      8. Independently, compute the ground truth label for each provided node pair (1 if the edge exists, 0 otherwise).
      9. Evaluate the performance (ROC AUC and Average Precision) on the provided node pairs.

    Args:
        model: A callable PyTorch Geometric model: forward(x, edge_index, ...) -> node-level outputs.
        sub_data (Data): A PyG Data object for the subgraph (with sub_data.x, sub_data.edge_index, etc.).
        node_pairs_interest (list of tuple): List of node pairs (u, v) to evaluate in the subgraph.
        perturb_factor (float): Factor used to perturb node features.
        device (str): "cpu" or "cuda" (e.g. "cuda:0").

    Returns:
        influence (Tensor): Influence scores for nodes in sub_data.
        pair_scores (dict): { (u, v): continuous score } for each node pair in node_pairs_interest.
        predicted_labels (dict): Binary labels for each node pair based on top-k scoring.
        auc (float): ROC AUC over node_pairs_interest.
        ap (float): Average precision over node_pairs_interest.
    """
    # 1) Move the data and model to the same device
    sub_data = sub_data.to(device)
    model = model.to(device)
    model.eval()

    # 2) Compute baseline model outputs
    with torch.no_grad():
        # If your model expects forward(x, edge_index), do:
        baseline = model.predict_step_xy(sub_data.x, sub_data.edge_index)  # shape: (N, out_dim)
        # Or if you truly need `predict_step`, ensure it uses edge_index inside:
        # baseline = model.predict_step(sub_data)

    N = sub_data.num_nodes
    influence = torch.zeros(N, device=device)

    # 3) Compute influence scores by perturbing each node
    for i in range(N):
        # Copy node features and perturb node i
        x_pert = sub_data.x.clone()
        x_pert[i] = x_pert[i] * (1 + perturb_factor)

        # Build a new Data object for inference OR clone + override x
        d_pert = Data(
            x=x_pert,
            edge_index=sub_data.edge_index,
            edge_attr=getattr(sub_data, 'edge_attr', None)
        ).to(device)

        with torch.no_grad():
            new_output = model.predict_step_xy(d_pert.x, d_pert.edge_index)  # shape: (N, out_dim)

        # Sum of L2 norms across all nodes
        diff = new_output - baseline
        influence[i] = diff.norm(p=2, dim=1).sum()

    # 4) Compute a continuous score for each node pair
    pair_scores = {}
    for (u, v) in node_pairs_interest:
        pair_scores[(u, v)] = influence[u].item() + influence[v].item()

    # 5) Get the ground-truth edge set in the subgraph (undirected)
    edge_index = sub_data.edge_index.cpu().numpy()
    gt_edge_set = set()
    for j in range(edge_index.shape[1]):
        a = int(edge_index[0, j])
        b = int(edge_index[1, j])
        gt_edge_set.add(tuple(sorted((a, b))))

    # 6) Let k = number of actual edges in this subgraph
    k = len(gt_edge_set)

    # 7) Label the node pairs by descending score (top k = predicted edges)
    sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
    predicted_labels = {}
    for idx, (pair, score) in enumerate(sorted_pairs):
        predicted_labels[pair] = 1 if idx < k else 0

    # 8) Compute ground-truth labels for each node pair
    ground_truth = {}
    for (u, v) in node_pairs_interest:
        ground_truth[(u, v)] = 1 if tuple(sorted((u, v))) in gt_edge_set else 0
    # 9) Evaluate performance on the node pairs of interest
    y_true = [ground_truth[pair] for pair in node_pairs_interest]
    y_score = [pair_scores[pair] for pair in node_pairs_interest]
    y_pred = [predicted_labels[pair] for pair in node_pairs_interest]

    f1 = f1_score(y_true, y_pred)
    # Try to compute metrics (handle edge cases if y_true has only 0s or 1s)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = None
    print(f"Current AUC: {auc}, f1: {f1}")

    return auc, f1
# def evaluate_influence_edge_prediction(model, sub_data, node_pairs, perturb_factor=0.1):
#     """
#     Evaluate edge inference on a subgraph based on influence scores.
#
#     Procedure:
#       1. Compute baseline outputs: baseline = model(sub_data.x).
#       2. For each node i in the subgraph, perturb its features (e.g. scale by 1+perturb_factor)
#          and compute new outputs; then define the node's influence as the sum over all nodes
#          of the norm of the difference (new_output - baseline).
#       3. For each node pair (u,v) in node_pairs, assign a continuous pair score = influence[u] + influence[v].
#       4. Compute ground truth: among the nodes in node_pairs, find the number of true edges from sub_data.edge_index.
#       5. Label the top-k node pairs (by score) as connected, where k equals the ground-truth number.
#       6. Also compute ground-truth labels for each node pair (1 if the edge exists, 0 otherwise) and
#          evaluate performance using ROC AUC and Average Precision.
#
#     Args:
#         model: A callable model that takes in a tensor of node features and returns output predictions.
#         sub_data (Data): A PyG Data object representing the subgraph (should contain sub_data.x and sub_data.edge_index).
#         node_pairs (list of tuple): List of node pairs (u, v) for which to evaluate edge existence.
#         perturb_factor (float): Factor by which to perturb each node’s features.
#
#     Returns:
#         influence (Tensor): A tensor of influence scores, one per node in sub_data.
#         pair_scores (dict): Mapping from each node pair to its continuous score.
#         predicted_labels (dict): Predicted binary labels for each node pair (1 for predicted edge, 0 otherwise).
#         auc (float): ROC AUC score.
#         ap (float): Average Precision score.
#     """
#     device = sub_data.x.device
#     # Compute baseline outputs.
#     model.eval()
#     with torch.no_grad():
#         baseline = model(sub_data.x)  # shape: (N, out_dim)
#
#     N = sub_data.x.size(0)
#     influence = torch.zeros(N, device=device)
#
#     # For each node, perturb its feature and measure influence.
#     for i in range(N):
#         x_pert = sub_data.x.clone()
#         # Perturb node i by scaling its feature vector.
#         x_pert[i] = x_pert[i] * (1 + perturb_factor)
#         with torch.no_grad():
#             new_output = model(x_pert)
#         diff = new_output - baseline  # shape: (N, out_dim)
#         # Influence score: sum over all nodes of the L2 norm difference.
#         influence_i = diff.norm(p=2, dim=1).sum()
#         influence[i] = influence_i
#
#     # Compute pair scores for each provided node pair.
#     pair_scores = {}
#     for pair in node_pairs:
#         u, v = pair
#         pair_scores[pair] = influence[u].item() + influence[v].item()
#
#     # Determine ground truth edges among the nodes in node_pairs.
#     # Get all nodes from the node pairs.
#     nodes_in_pairs = set([u for (u, v) in node_pairs] + [v for (u, v) in node_pairs])
#
#     # Build a set of ground truth edges (undirected) among these nodes.
#     edge_index = sub_data.edge_index.cpu().numpy()
#     gt_edge_set = set()
#     for i in range(edge_index.shape[1]):
#         u = int(edge_index[0, i])
#         v = int(edge_index[1, i])
#         if u in nodes_in_pairs and v in nodes_in_pairs:
#             # For undirected, store the sorted tuple.
#             gt_edge_set.add(tuple(sorted((u, v))))
#     gt_num_edges = len(gt_edge_set)
#
#     # Label node pairs using continuous scores: assign predicted label 1 for top gt_num_edges pairs.
#     sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
#     predicted_labels = {}
#     for idx, (pair, score) in enumerate(sorted_pairs):
#         predicted_labels[pair] = 1 if idx < gt_num_edges else 0
#
#     # Also compute ground truth labels for the provided node pairs.
#     # A node pair is positive if it exists (in either order) in gt_edge_set.
#     ground_truth = {}
#     for pair in node_pairs:
#         u, v = pair
#         ground_truth[pair] = 1 if tuple(sorted((u, v))) in gt_edge_set else 0
#
#     # For evaluation, construct lists of continuous scores and ground truth labels.
#     y_true = [ground_truth[pair] for pair in node_pairs]
#     y_score = [pair_scores[pair] for pair in node_pairs]
#
#     try:
#         auc = roc_auc_score(y_true, y_score)
#     except ValueError:
#         auc = None
#     ap = average_precision_score(y_true, y_score)
#
#     return influence, pair_scores, predicted_labels, auc, ap
