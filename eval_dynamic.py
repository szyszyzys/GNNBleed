import argparse

from sklearn.metrics import recall_score, precision_score, f1_score

from utils.attack_utils import evaluate_with_threshold, evaluate_attack_auc_ap, get_average, \
    get_similarity_dict_list
import math

import torch
import numpy as np
from utils.attack_utils import concatenate_first_k_arrays, get_similarity_dict, append_dict_contents, \
    DISTANCE_METRICS_LIST_NAME, mds, pca


def evaluate_dp1(result):
    statistics = result['origin_output']
    average_recall_list, average_precision_list, average_f1_list, average_ap, average_auc = {}, {}, {}, {}, {}
    agg_dict_list = []
    mode = "con"
    for output in statistics:
        target_change_all_list = output["target"]
        perturbed_change_all_list = output["candidate"]
        all_simi = []
        for i in range(len(perturbed_change_all_list)):
            target_change = data_processor(target_change_all_list[i], mode)
            perturbed_change = data_processor(perturbed_change_all_list[i], mode)

            simi_dict = get_similarity_dict(target_change, perturbed_change)
            simi_dict = {key: 1 - value for key, value in simi_dict.items()}
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

    return average_auc, average_recall_list, average_precision_list, average_f1_list, average_ap


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
            if math.isinf(cur_data[num]):
                cur_data[num] = -100
        auc, ap = evaluate_attack_auc_ap(cur_data[:n_degree], cur_data[n_degree:])
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        aps.append(ap)
        aucs.append(auc)
    return get_average(recalls), get_average(precisions), get_average(f1s), get_average(
        aps), get_average(aucs)


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
    recall = recall_score(shuffled_labels, y_pred, zero_division=0)
    precision = precision_score(shuffled_labels, y_pred, zero_division=0)
    f1 = f1_score(shuffled_labels, y_pred)
    return recall, precision, f1


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


def data_processor(data, mode="con", len=15):
    if mode == "con":
        return concatenate_first_k_arrays(data, len)
    elif mode == "mean":
        return np.mean(data, axis=0)
    elif mode == "median":
        return np.median(data, axis=0)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default='attack')
    parser.add_argument('--simi_type', type=str, default='attack')
    parser.add_argument('--eval_type', type=str, default='d')
    parser.add_argument('--n_query', type=int, default=15)
    parser.add_argument('--type', type=str, default="inf")

    return parser.parse_args()


def append_dict_contents(dict_list):
    aggregated_dict = {}
    for d in dict_list:
        for key, value in d.items():
            if key in aggregated_dict.keys():
                tmp = aggregated_dict[key]
                tmp.append(value)
                aggregated_dict[key] = tmp
            else:
                aggregated_dict[key] = [value]
    return aggregated_dict


def evaluate_dp2(result, l=15, mode="con", simi_type="perturb"):
    statistics = result['origin_output']
    average_recall_list, average_precision_list, average_f1_list, average_ap, average_auc = {}, {}, {}, {}, {}
    agg_dict_list = []
    for output in statistics:
        target_change_all_list = output["target"]
        perturbed_change_all_list = output["perturbed"]
        anchor_change_all_list = output["anchor"]

        all_simi = []
        for i in range(len(perturbed_change_all_list)):

            target_change = data_processor(target_change_all_list[i], mode, l)
            perturbed_change = data_processor(perturbed_change_all_list[i], mode, l)
            anchor_change = data_processor(anchor_change_all_list[i], mode, l)
            if simi_type == "perturb":
                simi_dict = get_similarity_dict(target_change, perturbed_change)
                simi_dict = {key: value for key, value in simi_dict.items()}
                all_simi.append(simi_dict)
            elif simi_type == "anchor":
                simi_dict = get_similarity_dict(target_change, anchor_change)
                simi_dict = {key: value for key, value in simi_dict.items()}
                all_simi.append(simi_dict)
            else:
                diff_change = perturbed_change - anchor_change
                simi_dict = get_similarity_dict(target_change, diff_change)
                simi_dict = {key: value for key, value in simi_dict.items()}
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
    return average_f1_list, average_auc


def evaluate_dp2_sep(result, l=15, mode="con", simi_type="perturb"):
    statistics = result['origin_output']
    average_recall_list, average_precision_list, average_f1_list, average_ap, average_auc = {}, {}, {}, {}, {}
    agg_dict_list = []
    for output in statistics:
        target_change_all_list = output["target"]
        perturbed_change_all_list = output["perturbed"]
        anchor_change_all_list = output["anchor"]
        all_simi = []
        for i in range(len(perturbed_change_all_list)):
            target_change = target_change_all_list[i]
            perturbed_change = perturbed_change_all_list[i]
            anchor_change = anchor_change_all_list[i]
            if simi_type == "perturb":
                simi_dict = get_similarity_dict_list(target_change, perturbed_change)
                simi_dict = {key: value for key, value in simi_dict.items()}
                all_simi.append(simi_dict)
            elif simi_type == "anchor":
                simi_dict = get_similarity_dict_list(target_change, anchor_change)
                simi_dict = {key: value for key, value in simi_dict.items()}
                all_simi.append(simi_dict)
            else:
                diff_change = perturbed_change - anchor_change
                simi_dict = get_similarity_dict_list(target_change, diff_change)
                simi_dict = {key: value for key, value in simi_dict.items()}
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
    return average_f1_list, average_auc


def evaluate_dp2_pca(result, l=15, mode="con", simi_type="perturb"):
    statistics = result['origin_output']
    average_recall_list, average_precision_list, average_f1_list, average_ap, average_auc = {}, {}, {}, {}, {}
    agg_dict_list = []
    for output in statistics:
        target_change_all_list = output["target"]
        perturbed_change_all_list = output["perturbed"]
        anchor_change_all_list = output["anchor"]
        all_simi = []
        target_changes, perturbed_changes, anchor_changes = [], [], []
        for i in range(len(perturbed_change_all_list)):
            target_changes.append(data_processor(target_change_all_list[i], mode, l))
            perturbed_changes.append(data_processor(perturbed_change_all_list[i], mode, l))
            anchor_changes.append(data_processor(anchor_change_all_list[i], mode, l))
        target_change_list = pca(target_changes)
        anchor_change_list = pca(anchor_changes)
        perturbed_change_list = pca(perturbed_changes)

        for i in range(len(perturbed_change_all_list)):
            target_change = target_change_list[i]
            perturbed_change = perturbed_change_list[i]
            anchor_change = anchor_change_list[i]
            if simi_type == "perturb":
                simi_dict = get_similarity_dict(target_change, perturbed_change)
                simi_dict = {key: value for key, value in simi_dict.items()}
                all_simi.append(simi_dict)
            elif simi_type == "anchor":
                simi_dict = get_similarity_dict(target_change, anchor_change)
                simi_dict = {key: value for key, value in simi_dict.items()}
                all_simi.append(simi_dict)
            else:
                diff_change = perturbed_change - anchor_change
                simi_dict = get_similarity_dict(target_change, diff_change)
                simi_dict = {key: value for key, value in simi_dict.items()}
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
    return average_f1_list, average_auc


def evaluate_dp2_pca_sum(result, l=15, mode="con", simi_type="perturb"):
    statistics = result['origin_output']
    agg_dict_list = []
    for output in statistics:
        target_change_all_list = output["target"]
        perturbed_change_all_list = output["perturbed"]
        anchor_change_all_list = output["anchor"]
        target_changes, perturbed_changes, anchor_changes = [], [], []
        for i in range(len(perturbed_change_all_list)):
            target_changes.append(data_processor(target_change_all_list[i], mode, l))
            perturbed_changes.append(data_processor(perturbed_change_all_list[i], mode, l))
            anchor_changes.append(data_processor(anchor_change_all_list[i], mode, l))
        target_change_list = pca(target_changes)

        for i in range(len(perturbed_change_all_list)):
            target_change = target_change_list[i]
            norm = target_change.norm(dim=1)
        agg_dict_list.append(norm)
    average_recall_list, average_precision_list, average_f1_list, average_ap, average_auc = get_average_metrics(
        agg_dict_list, result["num_direct_connections"],
        threshold_ratio=1)
    return average_f1_list, average_auc


#
# def evaluate_dp2_test(result, l=15, mode="con"):
#     statistics = result['origin_output']
#     average_recall_list, average_precision_list, average_f1_list, average_ap, average_auc = {}, {}, {}, {}, {}
#     agg_dict_list = []
#     for output in statistics:
#         target_change_all_list = output["target"]
#         perturbed_change_all_list = output["perturbed"]
#         anchor_change_all_list = output["anchor"]
#         all_simi = []
#         for i in range(len(perturbed_change_all_list)):
#             target_change = data_processor(target_change_all_list[i], mode, l)
#             perturbed_change = data_processor(perturbed_change_all_list[i], mode, l)
#             anchor_change = data_processor(anchor_change_all_list[i], mode, l)
#             inf_change = perturbed_change - anchor_change
#             simi_dict = get_similarity_dict(target_change, perturbed_change)
#             simi_dict = {key: value for key, value in simi_dict.items()}
#             all_simi.append(simi_dict)
#         agg_dict = append_dict_contents(all_simi)
#         agg_dict_list.append(agg_dict)
#     for metrics_name in DISTANCE_METRICS_LIST_NAME:
#         current_metric_outputs = []
#         for output in agg_dict_list:
#             current_metric_outputs.append(output[metrics_name])
#         average_recall_list[metrics_name], average_precision_list[metrics_name], average_f1_list[
#             metrics_name], average_ap[metrics_name], average_auc[metrics_name] = get_average_metrics(
#             current_metric_outputs, result["num_direct_connections"],
#             threshold_ratio=1)
#     return average_f1_list

def evaluate_inf(res):
    processed_statistic = res["statistic"]
    average_recall, average_precision, average_f1, average_ap, average_auc = get_average_metrics(
        processed_statistic,
        res[
            "num_direct_connections"],
        threshold_ratio=1)

    return average_f1, average_auc


def evaluate_inf_3(result):
    processed_statistic = []
    if result["attack_type"] == 'inf_3':
        for t in result["statistic"]:
            influence_score_list_processed = []
            for influence_score_1, influence_score_3 in t:
                if influence_score_3 != 0:
                    influence_score_list_processed.append(influence_score_1 / influence_score_3)
                else:
                    influence_score_list_processed.append(influence_score_1)

            processed_statistic.append(influence_score_list_processed)

    average_recall, average_precision, average_f1, average_ap, average_auc = get_average_metrics(
        processed_statistic,
        result[
            "num_direct_connections"],
        threshold_ratio=1)

    return average_f1, average_auc


def get_max_metrics(sample_dict):
    max_key = max(sample_dict, key=sample_dict.get)

    # Print the key-value pair
    max_value = sample_dict[max_key]
    print(f"Key with the greatest value: {max_key}, Value: {max_value}")


def main():
    args = get_arguments()
    print(f"start evaluate result, current file: {args.result_path}")
    res = torch.load(args.result_path)
    if "infiltration" in args.result_path:
        pre, auc = evaluate_inf(res)
        print("f1---------------")
        print(pre)
        print("auc---------------")
        print(auc)
        return
    elif args.type == "inf":
        pre, auc = evaluate_inf_3(res)
        print("f1---------------")
        print(pre)
        print("auc---------------")
        print(auc)
        return
    elif args.eval_type == 'sep':
        pre, auc = evaluate_dp2_sep(res, 15, "con", args.simi_type)
    elif args.eval_type == 'pca':
        pre, auc = evaluate_dp2_pca(res, 15, "con", args.simi_type)
    else:
        pre, auc = evaluate_dp2(res, args.n_query, args.eval_type, args.simi_type)
    print("f1---------------")
    print(pre)
    get_max_metrics(pre)
    print("auc-------------")
    print(auc)
    get_max_metrics(auc)


if __name__ == '__main__':
    main()
