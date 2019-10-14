from functools import partial


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
