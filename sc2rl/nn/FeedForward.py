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

    def forward(self, graph, node_feature, update_node_types=['ally']):
        """
        :param graph: Structure only graph. Input graph has no node features
        :param node_feature:
        :return: updated features
        """
        graph.ndata['node_feature'] = node_feature

        if self.neighbor_degree == 0:  # Update features with only own features
            for ntype in update_node_types:
                graph.apply_nodes(func=self.apply_node_function_no_neighbor, ntype=ntype)
        else:  # Update features with own features and 1 hop neighbor features
            for etype in graph.etypes:
                graph.send_and_recv(graph[etype].edges(),
                                    message_func=self.message_function,
                                    reduce_func=self.reduce_function,
                                    etype=etype)

            for ntype in update_node_types:
                graph.apply_nodes(func=self.apply_node_function_yes_neighbor, ntype=ntype)

        ret_dict = dict()

        for ntype in graph.ntypes:
            ret_dict[ntype] = graph.nodes[ntype].data.pop('node_feature')
        if self.neighbor_degree >= 1:  # clear intermediate messages
            for ntype in update_node_types:
                graph.nodes[ntype].data.pop('aggregated_message')

        return ret_dict

    @staticmethod
    def message_function(edges):
        return {'neighbor_feature': edges.src['node_feature']}

    @staticmethod
    def reduce_function(nodes):
        return {'aggregated_message': nodes.mailbox['neighbor_feature'].sum(1)}

    def apply_node_function_yes_neighbor(self, nodes):
        _inp = torch.cat((nodes.data['aggregated_message'], nodes.data['node_feature']), dim=-1)
        return {'node_feature': self.node_updater(_inp)}

    def apply_node_function_no_neighbor(self, nodes):
        return {'node_feature': self.node_updater(nodes.data['node_feature'])}
