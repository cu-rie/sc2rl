import torch
from .MultiLayerPerceptron import MultiLayerPerceptron as MLP


class FeedForwardNeighbor(torch.nn.Module):

    def __init__(self,
                 model_dim,
                 neighbor_degree: int = 0,
                 num_neurons: list = [128, 128]):
        super(FeedForwardNeighbor, self).__init__()

        self.model_dim = model_dim
        assert 0 <= neighbor_degree <= 1, "'neighbor_degree' should be 0 or 1"
        self.neighbor_degree = neighbor_degree

        input_dim = (neighbor_degree + 1) * model_dim
        self.node_updater = MLP(input_dim, model_dim, num_neurons=num_neurons)

    def forward(self, graph, node_feature):
        """
        :param graph: Structure only graph. Input graph has no node features
        :param node_feature: Tensor. Node features
        :return: updated features
        """
        graph.ndata['node_feature'] = node_feature

        if self.neighbor_degree == 0:  # Update features with only own features
            graph.apply_nodes(func=self.apply_node_function_no_neighbor)
        else:  # Update features with own features and 1 hop neighbor features
            graph.update_all(message_func=self.message_function,
                             reduce_func=self.reduce_function,
                             apply_node_func=self.apply_node_function_yes_neighbor)

        # Delete intermediate feature to maintain structure only graph
        _ = graph.ndata.pop('node_feature')
        if self.neighbor_degree >= 1:
            _ = graph.ndata.pop('aggregated_message')

        return graph.ndata.pop('updated_node_feature')

    @staticmethod
    def message_function(edges):
        return {'neighbor_feature': edges.src['node_feature']}

    @staticmethod
    def reduce_function(nodes):
        return {'aggregated_message': nodes.mailbox['neighbor_feature'].sum(1)}

    def apply_node_function_yes_neighbor(self, nodes):
        _inp = torch.cat((nodes.data['aggregated_message'], nodes.data['node_feature']), dim=-1)
        return {'updated_node_feature': self.node_updater(_inp)}

    def apply_node_function_no_neighbor(self, nodes):
        return {'updated_node_feature': self.node_updater(nodes.data['node_feature'])}
