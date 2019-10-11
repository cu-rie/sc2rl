
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
