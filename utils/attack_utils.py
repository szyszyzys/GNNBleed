import logging
import math
import os
import sys

import numpy as np
from scipy.spatial.distance import cosine, euclidean, correlation, chebyshev, braycurtis, canberra, cityblock, \
    sqeuclidean
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import recall_score, precision_score, f1_score

DISTANCE_METRICS_LIST = [cosine, euclidean, correlation, chebyshev, braycurtis, canberra, cityblock, sqeuclidean]
DISTANCE_METRICS_LIST_NAME = ['cosine', 'euclidean', 'correlation', 'chebyshev', 'braycurtis', 'canberra', 'cityblock',
                              'sqeuclidean']


# DISTANCE_METRICS_LIST = [cosine, euclidean, correlation]
# DISTANCE_METRICS_LIST_NAME = ['cosine', 'euclidean', 'correlation']

def evaluate_attack_auc_ap(connected, unconnected):
    """
    Evaluates the attack performance using ROC AUC and average precision.

    Args:
        connected (list or array): Predicted scores for connected samples (positive class, label 0).
        unconnected (list or array): Predicted scores for unconnected samples (negative class, label 1).

    Returns:
        auc (float): Area Under the ROC Curve (ensured to be >= 0.5).
        ap (float): Average precision score.
    """
    y = [0] * len(connected) + [1] * len(unconnected)
    pred = connected + unconnected
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=0)
    auc = metrics.auc(fpr, tpr)
    auc = max(auc, 1 - auc)
    ap = metrics.average_precision_score(y, pred)
    return auc, ap


def get_similarity(inp1, inp2):
    simi_list = []
    for metrics_name, distance_metrics in zip(DISTANCE_METRICS_LIST_NAME, DISTANCE_METRICS_LIST):
        simi_list.append(distance_metrics(inp1, inp2))

    return simi_list


def get_similarity_dict(inp1, inp2):
    # metrics_combinations = [("cosine", "euclidean"), ("chebyshev", "euclidean"), ("chebyshev", "braycurtis"),
    #                         ("canberra", "braycurtis"), ("canberra", "correlation"), ("canberra", "correlation"),
    #                         ("cosine", "braycurtis")]
    simi_list = {}
    for metrics_name, distance_metrics in zip(DISTANCE_METRICS_LIST_NAME, DISTANCE_METRICS_LIST):
        simi_list[metrics_name] = 1 - distance_metrics(inp1, inp2)
    # for metrics_combination in metrics_combinations:
    #     d1 = metrics_combination[0]
    #     d2 = metrics_combination[1]
    #     simi_list[d1 + "+" + d2] = simi_list[d1] + simi_list[d2]
    return simi_list


def get_similarity_dict_list(inp1, inp2):
    simi_list = {}
    for metrics_name, distance_metrics in zip(DISTANCE_METRICS_LIST_NAME, DISTANCE_METRICS_LIST):
        for i in range(len(inp1)):
            simi_list[metrics_name] = 1 - distance_metrics(inp1[i], inp2[i]) + simi_list.get(metrics_name, 0)
    return simi_list


def get_similarity_dict_seperate(inp1, inp2):
    metrics_combinations = [("cosine", "euclidean"), ("chebyshev", "euclidean"), ("chebyshev", "braycurtis"),
                            ("canberra", "braycurtis"), ("canberra", "correlation"), ("canberra", "correlation"),
                            ("cosine", "braycurtis")]
    simi_list = {}
    for metrics_name, distance_metrics in zip(DISTANCE_METRICS_LIST_NAME, DISTANCE_METRICS_LIST):
        simi_list[metrics_name] = 1 - distance_metrics(inp1, inp2)
    for metrics_combination in metrics_combinations:
        d1 = metrics_combination[0]
        d2 = metrics_combination[1]
        simi_list[d1 + "+" + d2] = simi_list[d1] + simi_list[d2]
    return simi_list


def config_attack_logger(log_path='./log', log_name=''):
    os.makedirs(log_path, exist_ok=True)

    fileHandler = logging.FileHandler(f'{log_path}/{log_name}.log')

    handlers = [fileHandler]
    consoleHandler = logging.StreamHandler(sys.stdout)
    handlers.append(consoleHandler)

    logging.basicConfig(
        level=logging.INFO,
        format=
        "%(asctime)s [%(process)d] [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=handlers)


def get_average(data_list):
    return sum(data_list) / len(data_list)


def evaluate_similarity_attack(result, args):
    statistics = result['statistic']
    average_recall_list, average_precision_list, average_f1_list, average_ap, average_auc = {}, {}, {}, {}, {}
    for metrics_name in DISTANCE_METRICS_LIST_NAME:
        current_metric_outputs = []
        for output in statistics:
            if result["attack_type"] == 'simi_1' or result["attack_type"] == 'lsa_post':
                current_metric_outputs.append([(1 - i) for i in output[metrics_name]])
            else:
                current_metric_outputs.append(output[metrics_name])
        average_recall_list[metrics_name], average_precision_list[metrics_name], average_f1_list[
            metrics_name], average_ap[metrics_name], average_auc[metrics_name] = get_average_metrics(
            current_metric_outputs, result["num_direct_connections"],
            threshold_ratio=1)
    return average_auc, average_recall_list, average_precision_list, average_f1_list, average_ap


def concatenate_first_k_arrays(array_list, k, axis=0):
    if k > len(array_list):
        raise ValueError("k is larger than the list size.")

    # Select the first k arrays from the list
    selected_arrays = array_list[:k]

    # Concatenate the selected arrays
    concatenated_array = np.concatenate(selected_arrays, axis=axis)

    return concatenated_array


def evaluate_dynamic_attack(result, args):
    return evaluate_dp2(result, args)


def evaluate_dynamic_attack_norm(result, args):
    statistics = result['origin_output']
    norm_list = []
    for output in statistics:
        target_change_r = output["target"]
        candidate_change_r = output["candidate"]
        cur_norm_list = []
        for i in range(len(target_change_r)):
            target_change = concatenate_first_k_arrays(target_change_r[i], 5)
            candidate_change = concatenate_first_k_arrays(candidate_change_r[i], 5)
            norm = np.linalg.norm(target_change - candidate_change)
            cur_norm_list.append(norm)
        norm_list.append(cur_norm_list)
    average_recall, average_precision, average_f1, average_ap, average_auc = get_average_metrics(
        norm_list,
        result["num_direct_connections"],
        threshold_ratio=args.threshold_ratio)

    return average_auc, average_recall, average_precision, average_f1, average_ap


def evaluate_dp2(result, args):
    statistics = result['origin_output']
    average_recall_list, average_precision_list, average_f1_list, average_ap, average_auc = {}, {}, {}, {}, {}
    agg_dict_list = []
    for output in statistics:
        target_change_all_list = output["target"]
        perturbed_change_all_list = output["perturbed"]
        anchor_change_all_list = output["anchor"]
        all_simi = []
        for i in range(len(target_change_all_list)):
            target_change = concatenate_first_k_arrays(target_change_all_list[i], 5)
            perturbed_change = concatenate_first_k_arrays(perturbed_change_all_list[i], 5)
            anchor_change = concatenate_first_k_arrays(perturbed_change_all_list[i], 5)
            inf_change = perturbed_change - anchor_change
            simi_dict = get_similarity_dict(target_change, inf_change)
            all_simi.append(simi_dict)
        agg_dict = append_dict_contents(all_simi)
        agg_dict_list.append(agg_dict)

    for metrics_name in DISTANCE_METRICS_LIST_NAME:
        current_metric_outputs = []
        for output in agg_dict_list:
            current_metric_outputs.append(output[metrics_name])
        average_recall_list[metrics_name], average_precision_list[metrics_name], average_f1_list[
            metrics_name], average_ap[metrics_name], average_auc[metrics_name] = get_average_metrics(
            current_metric_outputs, result["num_direct_connections"],
            threshold_ratio=1)

    return average_auc, average_recall_list, average_precision_list, average_f1_list


def append_dict_contents(dict_list):
    aggregated_dict = {}
    for d in dict_list:
        for key, value in d.items():
            if key in aggregated_dict:
                tmp = aggregated_dict[key]
                tmp.append(value)
                aggregated_dict[key] = tmp
            else:
                aggregated_dict[key] = [value]
    return aggregated_dict


def evaluate_dynamic_attack_simi(result, args):
    statistics = result['origin_output']
    average_recall_list, average_precision_list, average_f1_list, average_ap, average_auc = {}, {}, {}, {}, {}
    agg_dict_list = []
    for output in statistics:
        target_change_r = output["target"]
        candidate_change_r = output["candidate"]
        all_simi = []
        for i in range(len(target_change_r)):
            target_change = concatenate_first_k_arrays(target_change_r[i], 5)
            candidate_change = concatenate_first_k_arrays(candidate_change_r[i], 5)
            simi_dict = get_similarity_dict(target_change, candidate_change)
            all_simi.append(simi_dict)
        agg_dict = append_dict_contents(all_simi)
        agg_dict_list.append(agg_dict)

    for metrics_name in DISTANCE_METRICS_LIST_NAME:
        current_metric_outputs = []
        for output in agg_dict_list:
            current_metric_outputs.append(output[metrics_name])
        average_recall_list[metrics_name], average_precision_list[metrics_name], average_f1_list[
            metrics_name], average_ap[metrics_name], average_auc[metrics_name] = get_average_metrics(
            current_metric_outputs, result["num_direct_connections"],
            threshold_ratio=1)
    return average_auc, average_recall_list, average_precision_list, average_f1_list


def eval_influence_attack(result, args):
    print('start evaluating influence.................................')
    if args.by_degree:
        if result["attack_type"] in ("inf_1", "inf_2", "inf_3", "lta"):
            return eval_influence_attack_with_degree(result, args)
        else:
            return 0, 0, 0, 0, 0
    processed_statistic = []
    inf3 = []
    inf1 = []
    if result["attack_type"] == 'inf_3':
        for t in result["statistic"]:
            influence_score_list_processed = []
            for influence_score_1, influence_score_3 in t:
                if influence_score_3 != 0:
                    influence_score_list_processed.append(influence_score_1 / influence_score_3)
                else:
                    influence_score_list_processed.append(influence_score_1)
                inf1.append(influence_score_1)
                inf3.append(influence_score_3)
            processed_statistic.append(influence_score_list_processed)

    elif result["attack_type"] == 'inf_4':
        return evaluate_similarity_attack(result, args)
    else:
        processed_statistic = result["statistic"]

    average_recall, average_precision, average_f1, average_ap, average_auc = get_average_metrics(
        processed_statistic,
        result[
            "num_direct_connections"],
        threshold_ratio=args.threshold_ratio)
    if result["attack_type"] in ['inf_3', 'inf_4']:
        a1 = sum(inf1) / len(inf1)
        a3 = sum(inf3) / len(inf3)
        print(f"average inf 1 == {a1}")
        print(f"average inf 3 == {a3}")
        print(f"average inf 1:3 rate == {a1 / a3}")
    else:
        s = 0
        cnt = 0
        for t in result["statistic"]:
            for score in t:
                s += score
                cnt += 1
        print(f"average inf == {s / cnt}")

    return average_auc, average_recall, average_precision, average_f1, average_ap


def evaluate_with_threshold(data_list, num_connected, threshold):
    y_true = [1] * num_connected + [0] * (len(data_list) - num_connected)
    sorted_res = sorted(data_list, reverse=True)
    # threshold - 1 == the threshold-th largest number in the sorted list
    k_th_largest = sorted_res[threshold]
    data_list = np.array(data_list)
    y_true = np.array(y_true)
    permutation = np.random.permutation(len(data_list))
    shuffled_probs = data_list[permutation]
    shuffled_labels = y_true[permutation]
    y_pred = []
    cnt = 0
    for cur in shuffled_probs:
        if k_th_largest == 0:
            if cur > 0:
                y_pred.append(1)
                cnt += 1
            else:
                y_pred.append(0)
        elif cur > k_th_largest and cnt < threshold:
            y_pred.append(1)
            cnt += 1
        else:
            y_pred.append(0)
    y_pred = np.array(y_pred)
    recall = recall_score(shuffled_labels, y_pred)
    precision = precision_score(shuffled_labels, y_pred)
    # if recall != precision:
    #     print(recall)
    #     print(precision)
    #     print(shuffled_labels)
    #     print(y_pred)
    f1 = f1_score(shuffled_labels, y_pred)
    return recall, precision, f1


def get_average_metrics(data_list, num_connected_list, threshold_ratio=1):
    recalls = []
    precisions = []
    f1s = []
    aps = []
    aucs = []
    num_samples = len(data_list)

    for i in range(num_samples):
        n_degree = num_connected_list[i]
        if n_degree < 3:
            continue
        cur_data = data_list[i]
        if threshold_ratio > 1:
            threshold = math.ceil(threshold_ratio * n_degree)
        else:
            threshold = math.floor(threshold_ratio * n_degree)
        recall, precision, f1 = evaluate_with_threshold(data_list[i], n_degree, threshold)
        for num in range(len(cur_data)):
            if math.isnan(cur_data[num]):
                cur_data[num] = -100
        auc, ap = evaluate_attack_auc_ap(cur_data[:n_degree], cur_data[n_degree:])
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        aps.append(ap)
        aucs.append(auc)

    return get_average(recalls), get_average(precisions), get_average(f1s), get_average(
        aps), get_average(aucs)


def eval_influence_attack_with_degree(result, args):
    low_degree_indices = []
    high_degree_indices = []
    medium_degree_indices = []
    statistic_list = result["statistic"]
    if result["attack_type"] == 'inf_3':
        statistic_list = []
        for t in result["statistic"]:
            influence_score_list_processed = []
            for influence_score_1, influence_score_3 in t:
                if influence_score_3 != 0:
                    influence_score_list_processed.append(influence_score_1 / influence_score_3)
                else:
                    influence_score_list_processed.append(influence_score_1)
            statistic_list.append(influence_score_list_processed)

    degree_list = result["num_direct_connections"]
    for i in range(len(degree_list)):
        if degree_list[i] <= args.degree_low:
            low_degree_indices.append(i)
        elif degree_list[i] >= args.degree_high:
            high_degree_indices.append(i)
        else:
            medium_degree_indices.append(i)

    # low
    print(f'total number of samples: {len(statistic_list)}')
    print(f'low degree target node (degree <= {args.degree_low})')
    average_recall_l, average_precision_l, average_f1_l, average_ap_l, average_auc_l = eval_attack_full(
        extract_sub_list(statistic_list, low_degree_indices),
        extract_sub_list(degree_list, low_degree_indices),
        args)

    print(
        f'low degree: number of samples: {len(low_degree_indices)}, auc: {average_auc_l}, recall: {average_recall_l}, '
        f'precision: {average_precision_l}, ap: {average_ap_l}, f1: {average_f1_l}')

    # medium
    print(f'medium degree target node ({args.degree_low} < degree < {args.degree_high})')
    average_recall_m, average_precision_m, average_f1_m, average_ap_m, average_auc_m = eval_attack_full(
        extract_sub_list(statistic_list, high_degree_indices),
        extract_sub_list(degree_list, high_degree_indices),
        args)
    print(
        f'medium degree: number of samples: {len(medium_degree_indices)}, auc: {average_auc_m},'
        f'recall: {average_recall_m}, precision: {average_precision_m}, ap: {average_ap_m}, f1: {average_f1_m}')

    # high
    print(f'high degree target node ({args.degree_high}  <= degree )')
    average_recall_h, average_precision_h, average_f1_h, average_ap_h, average_auc_h = eval_attack_full(
        extract_sub_list(statistic_list, high_degree_indices),
        extract_sub_list(degree_list, high_degree_indices),
        args)
    print(
        f'high degree: number of samples: {len(high_degree_indices)}, auc: {average_auc_h},'
        f'recall: {average_recall_h}, precision: {average_precision_h}, ap: {average_ap_h}, f1: {average_f1_h}')

    print(
        f'recall: {average_recall_h * len(high_degree_indices) + average_recall_l * len(low_degree_indices) + len(medium_degree_indices) * average_recall_m}'
        f'precision: {average_precision_h * len(high_degree_indices) + average_precision_l * len(low_degree_indices) + len(medium_degree_indices) * average_precision_m}')

    return 0, 0, 0, 0, 0


def extract_sub_list(origin_list, indices):
    return [origin_list[index] for index in indices]


def eval_attack_full(statistic_list, degree_list, args):
    average_recall, average_precision, average_f1, average_ap, average_auc = get_average_metrics(statistic_list,
                                                                                                 degree_list,
                                                                                                 threshold_ratio=args.threshold_ratio)
    return average_recall, average_precision, average_f1, average_ap, average_auc


def cluster_fun(inp, k, clustering_method='Agglomerative'):
    if clustering_method == 'Agglomerative':
        clustering = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='complete').fit(inp)
    elif clustering_method == 'KMeans':
        clustering = KMeans(n_clusters=k).fit(inp)
    elif clustering_method == 'Spectral':
        clustering = SpectralClustering(n_clusters=k, affinity='precomputed').fit(inp)
    else:
        print(f'unknown cluster method: {clustering_method}')
        return
    return clustering


def find_centroid_node(vectors):
    # Compute the centroid (mean along the rows)
    centroid = np.mean(vectors, axis=0)

    distances = np.linalg.norm(vectors - centroid, axis=1)

    # Find the index of the user with the minimum distance to the centroid
    closest_user_index = np.argmin(distances)
    return closest_user_index


def find_mean_node_feature(vectors):
    return np.mean(vectors, axis=0)


def find_median_node_feature(vectors):
    return np.median(vectors, axis=0)


def pca(data):
    # Initialize PCA to reduce data to 2 dimensions
    pca = PCA(n_components=2)
    # Fit PCA on the dataset
    pca.fit(data)
    # Transform the data according to the PCA
    transformed_data = pca.transform(data)
    return transformed_data


def mds(data):
    mds = MDS(n_components=2, random_state=42)
    # Fit MDS on the dataset
    transformed_data = mds.fit_transform(data)
    return transformed_data
