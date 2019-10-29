import torch
from functools import partial
from .MultiLayerPerceptron import MultiLayerPerceptron as MLP
from sc2rl.utils.graph_utils import get_filtered_node_index_by_type, get_filtered_edge_index_by_type


class FeedForwardNeighbor(torch.nn.Module):

    def __init__(self,
                 model_dim,
                 neighbor_degree: int = 0,
                 num_node_types: int = 1,
                 num_neurons: list = [128, 128]):
        super(FeedForwardNeighbor, self).__init__()

        self.model_dim = model_dim
        assert 0 <= neighbor_degree <= 1, "'neighbor_degree' should be 0 or 1"
        self.neighbor_degree = neighbor_degree

        input_dim = (neighbor_degree + 1) * model_dim
        node_updater_dict = {}
        for i in range(num_node_types):
            node_updater_dict['node_updater{}'.format(i)] = MLP(input_dim, model_dim, num_neurons=num_neurons)

        self.node_updater = torch.nn.ModuleDict(node_updater_dict)

    def forward(self, graph, node_feature, update_node_type_indices, update_edge_type_indices):
        """
        :param graph:
        :param node_feature:
        :param update_node_type_indices:
        :param update_edge_type_indices:
        :return:
        """
        graph.ndata['node_feature'] = node_feature

        if self.neighbor_degree == 0:  # Update features with only own features
            for ntype_idx in update_node_type_indices:
                node_index = get_filtered_node_index_by_type(graph, ntype_idx)
                apply_func = partial(self.apply_node_function_no_neighbor, ntype_idx=ntype_idx)
                graph.apply_nodes(func=apply_func, v=node_index)
        else:  # Update features with own features and 1 hop neighbor features
            for etype_idx in update_edge_type_indices:
                edge_index = get_filtered_edge_index_by_type(graph, etype_idx)
                graph.send_and_recv(edge_index, message_func=self.message_function, reduce_func=self.reduce_function)
            for ntype_idx in update_node_type_indices:
                node_index = get_filtered_node_index_by_type(graph, ntype_idx)
                apply_func = partial(self.apply_node_function_yes_neighbor, ntype_idx=ntype_idx)
                graph.apply_nodes(func=apply_func, v=node_index)

        updated_node_feature = graph.ndata.pop('node_feature')
        if self.neighbor_degree >= 1:
            graph.ndata.pop('aggregated_message')

        return updated_node_feature

    @staticmethod
    def message_function(edges):
        return {'neighbor_feature': edges.src['node_feature']}

    @staticmethod
    def reduce_function(nodes):
        return {'aggregated_message': nodes.mailbox['neighbor_feature'].sum(1)}

    def apply_node_function_yes_neighbor(self, nodes, ntype_idx):
        _inp = torch.cat((nodes.data['aggregated_message'], nodes.data['node_feature']), dim=-1)
        updater = self.node_updater['node_updater{}'.format(ntype_idx)]
        return {'node_feature': updater(_inp)}

    def apply_node_function_no_neighbor(self, nodes, ntype_idx):
        updater = self.node_updater['node_updater{}'.format(ntype_idx)]
        return {'node_feature': updater(nodes.data['node_feature'])}
