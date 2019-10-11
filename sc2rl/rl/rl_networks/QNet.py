import dgl
import torch

from sc2rl.nn.RelationalNetwork import RelationalNetwork
from sc2rl.rl.rl_modules.ActionModules import MoveModule, AttackModule, HoldModule
from sc2rl.utils.graph_utils import get_batched_index


class QNet(torch.nn.Module):

    def __init__(self,
                 num_layers: int,
                 node_dim: int,
                 global_dim: int,
                 global_norm: int,
                 use_hypernet=True,
                 hypernet_input_dim=None,
                 num_relations=None,
                 num_head: int = 1,
                 use_norm=True,
                 neighbor_degree=1,
                 relational_enc_num_neurons=[128, 128],
                 move_dim=4,
                 module_num_neurons=[128],
                 pooling_op='relu'):
        super(QNet, self).__init__()
        self.move_dim = move_dim
        self.relational_enc = RelationalNetwork(num_layers=num_layers,
                                                model_dim=node_dim,
                                                global_dim=global_dim,
                                                global_norm=global_norm,
                                                use_hypernet=use_hypernet,
                                                hypernet_input_dim=hypernet_input_dim,
                                                num_relations=num_relations,
                                                num_head=num_head,
                                                use_norm=use_norm,
                                                neighbor_degree=neighbor_degree,
                                                num_neurons=relational_enc_num_neurons,
                                                pooling_op=pooling_op)

        module_input_dim = node_dim + global_dim

        self.move_module = MoveModule(node_dim=module_input_dim,
                                      move_dim=move_dim,
                                      num_neurons=module_num_neurons)
        self.hold_module = HoldModule(node_dim=module_input_dim, num_neurons=module_num_neurons)
        self.attack_module = AttackModule(node_dim=module_input_dim, num_neurons=module_num_neurons)

    def forward(self, graph, attack_graph, node_feature, global_feature, ally_indices, soft_pooling, device):
        """
        :param graph: (dgl.Graph or dgl.BatchedGraph)
        :param attack_graph: (dgl.Graph or dgl.BatchedGraph)
        :param node_feature: (pytorch Tensor) [ (Batched) # Nodes x node_feature dim]
        :param global_feature: (pytorch Tensor) [# Graphs x global_feature dim]
        :param ally_indices: (list of list-like) Each element contains index of allies in the graph
        :param soft_pooling: (Bool) Pooling layer returns probability or not
        :param device: (str) where the computation happen.
        :return:
        """

        node_updated, global_updated = self.relational_enc.forward(graph, node_feature, global_feature, device)

        if type(graph) == dgl.BatchedDGLGraph:
            num_nodes = graph.batch_num_nodes
            num_nodes = torch.tensor(num_nodes)
            global_updated = torch.repeat_interleave(global_updated, num_nodes, dim=0)
            batched_ally_indices = get_batched_index(graph, ally_indices)

        else:
            global_updated = global_updated.repeat((node_feature.shape[0], 1))
            batched_ally_indices = ally_indices

        move_node_feature = torch.cat((node_updated, global_updated), dim=-1)
        hold_node_feature = torch.cat((node_updated, global_updated), dim=-1)
        attack_node_feature = torch.cat((node_updated, global_updated), dim=-1)

        move_argument = self.move_module.forward(graph, move_node_feature, batched_ally_indices)
        hold_argument = self.hold_module.forward(graph, hold_node_feature, batched_ally_indices)
        attack_argument = self.attack_module.forward(attack_graph, attack_node_feature, ally_indices)

        q_move = move_argument
        q_hold = hold_argument
        q_attack, attack_indices = attack_argument[0], attack_argument[1]

        return q_move, q_hold, q_attack, attack_indices
