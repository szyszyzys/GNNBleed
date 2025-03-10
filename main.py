import argparse
import csv
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from attacker import Attacker
from data_loader import GraphDataModule
from lighting_modules import TrainingPipeline
from utils.attack_utils import evaluate_similarity_attack, eval_influence_attack, \
    evaluate_dynamic_attack
from utils.node_pair_sampling_utils import NodePairsSampler
from utils.train_utils import build_model


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='attack',
                        help='running mode: [attack | train | sample]')

    # common params
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--n_epoch', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--dataset', type=str, default='flickr',
                        help='Dataset')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--model', type=str.lower, default='gcn',
                        help='[ gcn, gat, sage ] ')

    # attack params
    parser.add_argument('--attack_type', type=str, default='simi_1',
                        help='attack types: [simi_1, simi_2, inf_1, inf_2].')
    parser.add_argument('--attack_node_num', type=int, default=100000,
                        help='maximum number of node pairs to attack')
    parser.add_argument('--attack_low_degree', type=int, default=6,
                        help='maximum number of node pairs to attack')
    parser.add_argument('--attack_high_degree', type=int, default=10,
                        help='maximum number of node pairs to attack')
    parser.add_argument('--insert_node_feature', type=str, default='random',
                        help='insert model feature: [random, typical, target, mean, median].')
    parser.add_argument('--attack_node_degree', type=str, default='uncons',
                        help='attack model degree: [uncons, low, high].')
    parser.add_argument('--dynamic', action='store_true', default=False,
                        help='dynamic graph')

    parser.add_argument('--insert_node_strategy', type=str, default='same',
                        help='attack types: [same, diff1, diff2].')
    parser.add_argument('--perturb_rate', type=float, default=1.0,
                        help='perturb rate of node feature.')
    parser.add_argument('--dynamic_insert_neighbor', action='store_true', default=False,
                        help='dynamic graph')
    parser.add_argument('--n_neighborhood_new_node', type=float, default=0.2,
                        help='perturb rate of node feature.')
    parser.add_argument('--dp2_insert_node', type=str, default="target",
                        help='.')
    parser.add_argument('--root_path', type=str, default="/revision",
                        help='.')
    # sample node pairs params
    parser.add_argument('--num_node_pairs', type=int, default=10, help='number of node pairs to sample.')
    parser.add_argument('--sample_node_subpath', type=str, default='/', help='sub path of sampled nodes')
    parser.add_argument('--dynamic_rate', type=float, default=0.01,
                        help='Initial learning rate.')
    # graph evolving params
    parser.add_argument('--evolving_mode', type=str, default="all",
                        help='graph evolving mode: [all, structure, feature, local_structure].')
    # train model params
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate.')
    parser.add_argument('--num_epoch', type=int, default=300,
                        help='training epoch.')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of model layers.')
    parser.add_argument('--h_dim', type=int, default=256,
                        help='model hidden layer dimension')
    parser.add_argument('--remove_self_loop', action='store_true', default=False,
                        help='if remove self loop.')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience of early stop.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='training batch size.')
    parser.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau',
                        help='lr_scheduler.')
    parser.add_argument('--gpuid',
                        type=list, action='store', default=[0])
    parser.add_argument('--num_of_worker', type=int, default=32,
                        help='dataloader number of workers.')

    # evaluation params
    parser.add_argument('--result_path', type=str,
                        help='path to attack result.')
    parser.add_argument('--threshold_ratio', type=float, default=1,
                        help='threshold_ratio for evaluating the attack.')
    parser.add_argument('--by_degree', action='store_false', default=False,
                        help='if evaluate attack by degree.')
    parser.add_argument('--degree_low', type=int, default=5,
                        help='dataloader number of workers.')
    parser.add_argument('--degree_high', type=int, default=10,
                        help='dataloader number of workers.')
    parser.add_argument('--is_balanced', action='store_true', default=False,
                        help='Set this flag if the nodes are balanced.')

    # defense params
    parser.add_argument('--against_defense', action='store_true')

    parser.add_argument('--noise_seed', type=int, default=42)
    parser.add_argument('--defense_type', type=str, default='',
                        help='[randedge, lapgraph, output_noise, output_clipping].')
    parser.add_argument('--dp_epsilon', type=float, default=0.1,
                        help='privacy budget.')
    parser.add_argument('--dp_delta', type=float, default=1e-5)
    parser.add_argument('--clipping_param', type=int, default=3)
    parser.add_argument('--twohop', action='store_true', default=False,
                        help='Set this flag if the nodes are balanced.')

    parser.set_defaults(assign_seed=42)

    return parser.parse_args()


def sample_node_pairs(args, root_path):
    # load dataset
    datamodule = GraphDataModule(
        dataset_name=args.dataset,
        remove_self_loop=args.remove_self_loop,
        num_workers=args.num_of_worker
    )
    graph_dataset = datamodule.full_attack_graph
    node_sampler = NodePairsSampler(graph_dataset, args.dataset,
                                    root_path=f'./{root_path}/outputs/twohop_{args.twohop}',
                                    sub_path=args.sample_node_subpath)
    if args.twohop:
        k_hop = 2
    else:
        k_hop = 3
    node_sampler.construct_centroid_pairs_set(args.num_node_pairs, graph_dataset, k_hop=k_hop,
                                              degree_range=[args.degree_low, args.degree_high],
                                              balanced=args.is_balanced)


def append_to_filename(original_path, new_items):
    """
    Append new items to the file name, before the file extension.

    :param original_path: The original file path.
    :param new_items: A list of items to append to the file name.
    :return: The modified file path with new items appended.
    """
    import os

    # Splitting the original path into a name and extension
    file_name, file_extension = os.path.splitext(original_path)

    # Adding new items to the file name
    for item in new_items:
        file_name += f"_{item}"

    # Reconstructing the full path with the new file name and original extension
    new_path = f"{file_name}{file_extension}"

    return new_path


def append_to_csv(root_path, model_name, num_layers, hidden_dim, learning_rate, test_accuracy, attack_name="",
                  attack_auc=0):
    Path(f'{root_path}').mkdir(parents=True, exist_ok=True)
    f_path = f"{root_path}/model_performance.csv"
    file_exists = os.path.isfile(f_path)
    with open(f_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write the header only if the file doesn't exist
            writer.writerow(
                ['Model', 'Num_Layers', 'Hidden_Dim', 'Learning_Rate', 'Test_Accuracy', 'Attack', 'Attack AUC'])
        writer.writerow([model_name, num_layers, hidden_dim, learning_rate, test_accuracy, attack_name, attack_auc])


def attack(args, root_path):
    # load target model
    current = datetime.now()
    current.strftime("%Y-%m-%d")

    # _lapgraph_seed_42_eps_1.0
    if args.against_defense:
        if args.defense_type == 'randedge' or args.defense_type == 'lapgraph':
            defense = f'{args.defense_type}_seed_{args.noise_seed}_eps_{args.dp_epsilon}'
            model, datamodule, trainer = build_model(args.model, args.dataset,
                                                     ckpt=args.model_path[:-3] + f'_{defense}.pt',
                                                     lr=args.lr,
                                                     h_dim=args.h_dim, num_layers=args.num_layers, n_epoch=args.n_epoch,
                                                     dropout=args.dropout, remove_self_loop=args.remove_self_loop,
                                                     gpuid=args.gpuid, arg=args)

        elif args.defense_type == 'output_noise' or args.defense_type == 'output_clipping':
            defense = f'{args.defense_type}_{args.clipping_param}'
            model, datamodule, trainer = build_model(args.model, args.dataset, ckpt=args.model_path, lr=args.lr,
                                                     h_dim=args.h_dim, num_layers=args.num_layers, n_epoch=args.n_epoch,
                                                     dropout=args.dropout, remove_self_loop=args.remove_self_loop,
                                                     gpuid=args.gpuid, arg=args)
    else:
        defense = 'None'

        model, datamodule, trainer = build_model(args.model, args.dataset, ckpt=args.model_path, lr=args.lr,
                                                 h_dim=args.h_dim, num_layers=args.num_layers, n_epoch=args.n_epoch,
                                                 dropout=args.dropout, remove_self_loop=args.remove_self_loop,
                                                 gpuid=args.gpuid, arg=args)
    # if args.dataset.startswith("twitch"):
    #     config_attack_logger(f'./log/{args.dataset[:7]}', log_name=f'{args.model}_{args.dataset[7:]}_{current}')
    # else:
    #     config_attack_logger('./log', log_name=f'{args.model}_{args.dataset}_{current}')

    pipeline = TrainingPipeline(model, datamodule, trainer)
    model_performance = pipeline.test()[0]
    test_accuracy = model_performance.get("run/test_accuracy", 0)

    # Append results to CSV
    # get the full graph
    graph_dataset = datamodule.full_attack_graph.clone().cuda()

    attacker = Attacker(model, graph_dataset, args.dataset, is_balanced=args.is_balanced,
                        insert_node_mode=args.insert_node_feature, target_degree=args.attack_node_degree,
                        defense_type=args.defense_type, clipping_param=args.clipping_param, dynamic=args.dynamic,
                        dynamic_rate=args.dynamic_rate, pipeline=pipeline,
                        insert_node_strategy=args.insert_node_strategy, influence_rate=args.perturb_rate,
                        dynamic_insert_neighbor=args.dynamic_insert_neighbor,
                        n_neighborhood_new_node=args.n_neighborhood_new_node, dp2_insert_node=args.dp2_insert_node,
                        evolving_mode=args.evolving_mode, dp_epsilon=args.dp_epsilon, twohop=args.twohop)
    target_nodes, node_pairs, num_direct_connections = attacker.get_node_pairs()
    logging.info('start attack')
    logging.info(f'attack dataset: {args.dataset}')
    logging.info(f'attack model: {args.model}')
    logging.info(f'number of target nodes: {len(target_nodes)}')
    start_time = time.time()
    all_res = []
    all_output = []
    all_statistics = []
    lta_f1 = []
    lta_auc = []
    base_path = f'./{root_path}/twohop_{args.twohop}/outputs/model_outputs/n_target_{min(len(target_nodes), args.attack_node_num)}'
    for i in tqdm(range(min(len(target_nodes), args.attack_node_num))):
        if args.attack_type == "ltao":
            auc, f1 = attacker.attack(args.attack_type, i)
            lta_f1.append(f1)
            lta_auc.append(auc)
        else:
            model_output, statistics, aucs = attacker.attack(args.attack_type, i)
            all_output.append(model_output)
            all_statistics.append(statistics)
            all_res.append(aucs)

    if args.attack_type == "ltao":
        if lta_f1:  # check that the list is not empty
            average_value_f1 = sum(lta_f1) / len(lta_f1)
            print("Average f1:", average_value_f1)
        if lta_auc:  # check that the list is not empty
            average_value_auc = sum(lta_auc) / len(lta_auc)
            print("Average auc:", average_value_auc)

        return

    # save the file path
    if args.dynamic:
        result_path = f'{base_path}/dynamic/{args.dataset}/{args.model}/{args.num_layers}/{args.attack_node_degree}/dim_{args.h_dim}_lr_{args.lr}/prate_{args.perturb_rate}_drate{args.dynamic_rate}/is_balanced_{args.is_balanced}'
    else:
        result_path = f'{base_path}/{args.dataset}/{args.model}/{args.num_layers}/{args.attack_node_degree}/dim_{args.h_dim}_lr_{args.lr}/prate_{args.perturb_rate}/is_balanced_{args.is_balanced}'
    if args.against_defense:
        result_path = f'{result_path}/{args.defense_type}/{args.attack_type}'
    if "softmax" in args.model_path:
        result_path = f'{result_path}/softmax'

    os.makedirs(result_path, exist_ok=True)
    resultpath = f'{result_path}/{args.attack_type}_{args.insert_node_feature}_defense_{defense}_result.pt'

    if args.dynamic:
        if args.attack_type == "dp2":
            resultpath = f'{result_path}/{args.attack_type}_{args.insert_node_feature}_dynamic_{args.dynamic_rate}_insertNode_{args.insert_node_strategy}_neighborinsert_{args.dynamic_insert_neighbor}_{args.n_neighborhood_new_node}.pt'
        else:
            resultpath = f'{result_path}/{args.attack_type}_{args.insert_node_feature}_dynamic_{args.dynamic_rate}_insertNode_{args.insert_node_strategy}_neighborinsert_{args.dynamic_insert_neighbor}_{args.n_neighborhood_new_node}.pt'
        append_to_filename(resultpath, [args.evolving_mode])

    torch.save({
        "statistic": all_statistics,
        "target_nodes": target_nodes,
        "node_pairs": node_pairs,
        "num_direct_connections": num_direct_connections,
        "attack_time": time.time() - start_time,
        "attack_params": args,
        "attack_type": args.attack_type,
        "node_pair_num": min(len(target_nodes), args.attack_node_num),
        "origin_output": all_output,
        "auc": all_res,
    },
        resultpath)
    logging.info(f'done attack, takes {time.time() - start_time}')
    print(f"result saved to {result_path}")

    if args.attack_type != "inf_4":
        print("++++++++++++++++++++++++++++++++++++++++++++")
        print(f"avegrate auc = {sum(all_res) / len(all_res)}")
        append_to_csv(
            f"{base_path}/{args.dataset}/{args.model}/{args.num_layers}/dim_{args.h_dim}_lr_{args.lr}/prate_{args.perturb_rate}/is_balanced_{args.is_balanced}",
            args.model, args.num_layers, args.h_dim, args.lr,
            test_accuracy, args.attack_type,
            sum(all_res) / len(all_res))

def train_model(args, root_path):
    model, datamodule, trainer = build_model(args.model, args.dataset, lr=args.lr,
                                             h_dim=args.h_dim, num_layers=args.num_layers, n_epoch=args.n_epoch,
                                             dropout=args.dropout, remove_self_loop=args.remove_self_loop,
                                             lr_scheduler=args.lr_scheduler,
                                             batch_size=args.batch_size, gpuid=args.gpuid,
                                             weight_decay=args.weight_decay,
                                             momentum=args.momentum, patience=args.patience,
                                             num_workers=args.num_of_worker,
                                             arg=args
                                             )

    pipeline = TrainingPipeline(model, datamodule, trainer)
    if args.defense_type == 'randedge' or args.defense_type == 'lapgraph':
        defense = f'{args.defense_type}_seed_{args.noise_seed}_eps_{args.dp_epsilon}'
    elif args.defense_type == 'output_noise' or args.defense_type == 'output_clipping':
        defense = args.defense_type
    else:
        defense = 'None'
    root_path = f"./{root_path}/outputs/trained_model"
    save_path = f'{root_path}/{args.model}/{args.dataset}/nlayer_{args.num_layers}_hdim_{args.h_dim}_lr_{args.lr}_epoch_{args.n_epoch}_{defense}.pt'
    Path(f'{root_path}/{args.model}/{args.dataset}').mkdir(parents=True, exist_ok=True)
    pipeline.run()
    pipeline.test()
    torch.save(model, save_path)
    print(f'done train model, save to {save_path}')


def analyze_result(args, root_path):
    result_path = f"{root_path}/{args.result_path}"
    results = torch.load(args.result_path)
    print("====================================================================================")
    print(f'start evaluating, current file: {args.result_path}, nums samples: {len(results["statistic"])}')
    print(f'current attack: {results["attack_type"]}')
    if results["attack_type"].startswith('simi') or results["attack_type"].startswith('lsa'):
        average_auc, average_recall, average_precision, average_f1, average_ap = evaluate_similarity_attack(results,
                                                                                                            args)
    elif results["attack_type"].startswith('dp'):
        average_auc, average_recall, average_precision, average_f1 = evaluate_dynamic_attack(results, args)
    else:
        average_auc, average_recall, average_precision, average_f1, average_ap = eval_influence_attack(results, args)
    print(f'average auc: {average_auc}')
    print(f'average recall: {average_recall}')
    print(f'average precision: {average_precision}')
    print(f'average f1: {average_f1}')

    # print(f'average ap: {average_ap}')
    print("====================================================================================")


def analyze_result_balanced(args, root_path):
    result_path = f"{root_path}/{args.result_path}"
    results = torch.load(result_path)
    n_samples = len(results["statistic"])
    print(f'start evaluating, current file: {args.result_path}, nums samples: {len(results["statistic"])}')
    balanced_statistic = []
    for i in range(n_samples):
        balanced_statistic.append(results["statistic"][i][:results["num_direct_connections"][i] * 2])
    results["statistic"] = balanced_statistic
    if args.attack_type.startswith('simi'):
        # trim the result
        average_auc, average_recall, average_precision, average_f1, average_ap = evaluate_similarity_attack(results,
                                                                                                            args)
    else:
        average_auc, average_recall, average_precision, average_f1, average_ap = eval_influence_attack(results, args)
    print(
        f'average auc: {average_auc}, average recall: {average_recall}, average precision: {average_precision}, '
        f'average f1: {average_f1}, average ap: {average_ap}')
    print("====================================================================================")


def eval_model(args, root_path):
    if args.defense_type == 'randedge' or args.defense_type == 'lapgraph':
        defense = f'{args.defense_type}_seed_{args.noise_seed}_eps_{args.dp_epsilon}'
        model, datamodule, trainer = build_model(args.model, args.dataset,
                                                 ckpt=args.model_path[:-3] + f'_{defense}.pt',
                                                 lr=args.lr,
                                                 h_dim=args.h_dim, num_layers=args.num_layers, n_epoch=args.n_epoch,
                                                 dropout=args.dropout, remove_self_loop=args.remove_self_loop,
                                                 gpuid=args.gpuid, arg=args)

    else:
        defense = 'None'

        model, datamodule, trainer = build_model(args.model, args.dataset, ckpt=args.model_path, lr=args.lr,
                                                 h_dim=args.h_dim, num_layers=args.num_layers, n_epoch=args.n_epoch,
                                                 dropout=args.dropout, remove_self_loop=args.remove_self_loop,
                                                 gpuid=args.gpuid, arg=args)

    pipeline = TrainingPipeline(model, datamodule, trainer)
    model_performance = pipeline.test()
    print(model_performance)


def main():
    args = get_arguments()
    if args.mode != 'eval':
        print(str(args))
        print(f'start: {args.mode}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    root_path = args.root_path
    torch.set_float32_matmul_precision('medium')
    print(f"current arguments: {args}")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    if args.mode == 'attack':
        attack(args, root_path)
    elif args.mode == 'train':
        train_model(args, root_path)
    elif args.mode == 'sample':
        sample_node_pairs(args, root_path)
    elif args.mode == 'eval':
        print(f"start evaluate result, current file: {args.result_path}")
        analyze_result(args, root_path)
    elif args.mode == 'eval_b':
        analyze_result_balanced(args, root_path)
    elif args.mode == 'eval_model':
        eval_model(args, root_path)
    else:
        print(f'cannot find current mode: {args.mode}')


if __name__ == '__main__':
    main()
