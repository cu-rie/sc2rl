from functools import partial
import torch

from sc2rl.nn.MultiLayerPerceptron import MultiLayerPerceptron as MLP
from sc2rl.utils.graph_utils import get_filtered_edge_index_by_type

from sc2rl.config.nn_configs import VERY_LARGE_NUMBER


class MoveModule(torch.nn.Module):

    def __init__(self,
                 node_dim: int,
                 move_dim: int = 4,
                 num_neurons: list = [128],
                 hidden_activation: str = 'leaky_relu',
                 out_activation: str = None):
        super(MoveModule, self).__init__()
        self.move_argument_calculator = MLP(node_dim, move_dim, num_neurons,
                                            hidden_activation=hidden_activation,
                                            out_activation=out_activation)

    def forward(self, graph, node_feature):
        graph.ndata['node_feature'] = node_feature
        graph.apply_nodes(func=self.apply_node_function)
        move_argument = graph.ndata.pop('move_argument')
        return move_argument

    def apply_node_function(self, nodes):
        input_node_feature = nodes.data['node_feature']
        move_argument = self.move_argument_calculator.forward(input_node_feature)
        return {'move_argument': move_argument}


class AttackModule(torch.nn.Module):

    def __init__(self,
                 node_dim: int,
                 num_neurons: list = [128],
                 hidden_activation: str = 'leaky_relu',
                 out_activation: str = None):
        super(AttackModule, self).__init__()
        input_dim = node_dim * 2
        self.attack_argument_calculator = MLP(input_dim, 1, num_neurons,
                                              hidden_activation=hidden_activation,
                                              out_activation=out_activation)

    # def forward(self, graph, node_feature, maximum_num_enemy: int, attack_edge_type_index: int):
    #     """
    #     :param graph: (dgl.Graph or dgl.BatchedGraph) Attack graph is a graph that only contains edges for denoting attack relationship
    #     :param node_feature: (pytorch Tensor) Node features of units.
    #     :param attack_edge_type_index: key value for describing 'attack' edges
    #     :return: attack_argument, enemy_index
    #     """
    #     graph.ndata['node_feature'] = node_feature
    #
    #     edge_index = get_filtered_edge_index_by_type(graph, attack_edge_type_index)
    #     graph.send_and_recv(edges=edge_index,
    #                         message_func=self.message_function,
    #                         reduce_func=self.reduce_function)
    #     if len(edge_index) != 0:
    #         attack_argument = graph.ndata.pop('attack_argument')
    #         print(graph.ndata['enemy_tag'])
    #     else:
    #         attack_argument = 0
    #     return attack_argument

    def message_function(self, edges):
        enemy_node_features = edges.src['node_feature']  # Enemy units' feature
        enemy_tag = edges.src['tag']
        ally_node_features = edges.dst['node_feature']  # Ally units' feature
        attack_argument_input = torch.cat((ally_node_features, enemy_node_features), dim=-1)
        attack_argument = self.attack_argument_calculator(attack_argument_input)
        return {'attack_argument': attack_argument, 'enemy_tag': enemy_tag}

    @staticmethod
    def get_action_reduce_function(nodes, num_enemy_units):
        mailbox_attack_argument = nodes.mailbox['attack_argument']
        attack_argument = torch.ones(size=(len(nodes), num_enemy_units)) * - VERY_LARGE_NUMBER
        attack_argument[:, :mailbox_attack_argument.shape[1]] = mailbox_attack_argument.squeeze(dim=-1)

        mailbox_enemy_tag = nodes.mailbox['enemy_tag']
        enemy_tag = torch.zeros(size=(len(nodes), num_enemy_units), dtype=torch.long)
        enemy_tag[:, :mailbox_enemy_tag.shape[1]] = mailbox_enemy_tag
        return {'attack_argument': attack_argument, 'enemy_tag': enemy_tag}

    def forward(self, graph, node_feature, maximum_num_enemy: int, attack_edge_type_index: int):
        num_total_nodes = graph.number_of_nodes()
        graph.ndata['node_feature'] = node_feature
        edge_index = get_filtered_edge_index_by_type(graph, attack_edge_type_index)
        reduce_func = partial(self.get_action_reduce_function, num_enemy_units=maximum_num_enemy)
        graph.send_and_recv(edges=edge_index,
                            message_func=self.message_function,
                            reduce_func=reduce_func)
        if len(edge_index) != 0:
            attack_argument = graph.ndata.pop('attack_argument')
        else:
            attack_argument = torch.ones(size=(num_total_nodes, maximum_num_enemy)) * - VERY_LARGE_NUMBER
        return attack_argument


class HoldModule(torch.nn.Module):
    def __init__(self,
                 node_dim: int,
                 num_neurons=[128],
                 hidden_activation: str = 'leaky_relu',
                 out_activation: str = None):
        super(HoldModule, self).__init__()
        self.hold_argument_calculator = MLP(node_dim, 1, num_neurons,
                                            hidden_activation=hidden_activation,
                                            out_activation=out_activation)

    def forward(self, graph, node_feature):
        graph.ndata['node_feature'] = node_feature
        graph.apply_nodes(func=self.apply_function)
        return graph.ndata.pop('hold_argument')

    def apply_function(self, nodes):
        node_features = nodes.data['node_feature']
        hold_argument = self.hold_argument_calculator.forward(node_features)
        return {'hold_argument': hold_argument}
