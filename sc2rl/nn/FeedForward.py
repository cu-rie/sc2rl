from functools import partial
import torch

from sc2rl.nn.MultiLayerPerceptron import MultiLayerPerceptron as MLP
from sc2rl.utils.graph_utils import get_filtered_node_index_by_type



class FeedForward(torch.nn.Module):

    def __init__(self, conf, num_node_types):
        super(FeedForward, self).__init__()
        mlp_conf = conf.mlp_conf

        node_updater_dict = {}
        for i in range(num_node_types):
            node_updater_dict['node_updater{}'.format(i)] = MLP(**mlp_conf)
        self.node_updater = torch.nn.ModuleDict(node_updater_dict)

    def forward(self, graph, node_feature, update_node_type_indices):
        graph.ndata['node_feature'] = node_feature
        for ntype_idx in update_node_type_indices:
            node_index = get_filtered_node_index_by_type(graph, ntype_idx)
            apply_func = partial(self.apply_node_function, ntype_idx=ntype_idx)
            graph.apply_nodes(func=apply_func, v=node_index)
        updated_node_feature = graph.ndata.pop('node_feature')
        return updated_node_feature

    def apply_node_function(self, nodes, ntype_idx):
        updater = self.node_updater['node_updater{}'.format(ntype_idx)]
        return {'node_feature': updater(nodes.data['node_feature'])}
