import dgl
import torch
from sc2rl.nn.MultiLayerPerceptron import MultiLayerPerceptron as MLP


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

    def forward(self, graph, node_feature, attack_edge_key='attack_in_range'):
        """
        :param graph: (dgl.Graph or dgl.BatchedGraph) Attack graph is a graph that only contains edges for denoting attack relationship
        :param node_feature: (pytorch Tensor) Node features of units.
        :param attack_edge_key: key value for describing 'attack' edges
        :return: attack_argument, enemy_index
        """
        graph.ndata['node_feature'] = node_feature

        for i in range(5):
            print(" ########### Did you check the 'attack_edge_key'? ###########")

        graph.pull(message_func=self.message_function,
                   reduce_func=self.reduce_function,
                   etype=attack_edge_key)

        attack_argument = graph.ndata.pop('attack_argument')
        return attack_argument

    def message_function(self, edges):
        enemy_node_features = edges.src['node_feature']  # Enemy units' feature
        ally_node_features = edges.dst['node_feature']  # Ally units' feature
        attack_argument_input = torch.cat((ally_node_features, enemy_node_features), dim=-1)
        attack_argument = self.attack_argument_calculator(attack_argument_input)
        return {'attack_argument': attack_argument}

    @staticmethod
    def reduce_function(nodes):
        attack_argument = nodes.mailbox['attack_argument']
        return {'attack_argument': attack_argument.squeeze(dim=-1)}


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
