from functools import partial
import torch

from sc2rl.config.nn_configs import VERY_LARGE_NUMBER
from sc2rl.config.graph_configs import NODE_ENEMY


def get_batched_index(batched_graph, index_list, return_num_targets=False):
    _num_nodes = 0
    return_indices = []
    if return_num_targets:
        num_targets = []

    for num_node, target_index in zip(batched_graph.batch_num_nodes, index_list):
        indices = [i + _num_nodes for i in target_index]
        return_indices.extend(indices)
        _num_nodes += num_node
        if return_num_targets:
            num_targets.append(len(target_index))

    if not return_num_targets:
        return return_indices
    else:
        return return_indices, num_targets


def pop_node_feature_dict(graph, node_feature_key='node_feature'):
    ret_dict = dict()
    for ntype in graph.ntypes:
        ret_dict[ntype] = graph.nodes[ntype].data.pop(node_feature_key)
    return ret_dict


def set_node_feature_dict(graph, feature_dict, node_feature_key='node_feature'):
    for key, val in feature_dict.items():
        graph.nodes[key].data[node_feature_key] = val


def filter_by_edge_type_idx(edges, etype_idx):
    return edges.data['edge_type'] == etype_idx


def get_filtered_edge_index_by_type(graph, etype_idx):
    filter_func = partial(filter_by_edge_type_idx, etype_idx=etype_idx)
    edge_idx = graph.filter_edges(filter_func)
    return edge_idx


def filter_by_node_type_idx(nodes, ntype_idx):
    return nodes.data['node_type'] == ntype_idx


def get_filtered_node_index_by_type(graph, ntype_idx):
    filter_func = partial(filter_by_node_type_idx, ntype_idx=ntype_idx)
    node_idx = graph.filter_nodes(filter_func)
    return node_idx


def get_largest_number_of_enemy_nodes(graphs):
    max_num_enemy = 0
    for graph in graphs:
        num_enemy = len(get_filtered_node_index_by_type(graph, NODE_ENEMY))
        if max_num_enemy <= num_enemy:
            max_num_enemy = num_enemy
    return max_num_enemy


def curie_initializer(shape, dtype, ctx, id_range):
    return torch.ones(shape, dtype=dtype, device=ctx) * - VERY_LARGE_NUMBER
