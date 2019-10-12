import dgl
import torch

from sc2rl.utils.graph_utils import get_batched_index
from sc2rl.nn.RelationalNetwork import RelationalNetwork
from sc2rl.rl.rl_modules.ActionModules import MoveModule, HoldModule, AttackModule

VERY_SMALL_NUMBER = 1e-10


class Actor(torch.nn.Module):

    def __init__(self,
                 num_layers: int,
                 node_dim: int,
                 global_dim: int,
                 use_hypernet=True,
                 hypernet_input_dim=None,
                 num_relations=None,
                 num_head: int = 1,
                 use_norm=True,
                 neighbor_degree=1,
                 relational_enc_num_neurons=[128, 128],
                 move_dim=4,
                 module_num_neurons=[128],
                 hidden_activation: str = 'tanh',
                 out_activation: str = 'tanh',
                 pooling_op: str = 'relu'):
        super(Actor, self).__init__()
        self.move_dim = move_dim
        self.relational_enc = RelationalNetwork(num_layers=num_layers,
                                                model_dim=node_dim,
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
                                      num_neurons=module_num_neurons,
                                      hidden_activation=hidden_activation,
                                      out_activation=out_activation)

        self.hold_module = HoldModule(node_dim=module_input_dim,
                                      num_neurons=module_num_neurons,
                                      hidden_activation=hidden_activation,
                                      out_activation=out_activation)

        self.attack_module = AttackModule(node_dim=module_input_dim,
                                          num_neurons=module_num_neurons,
                                          hidden_activation=hidden_activation,
                                          out_activation=out_activation)

    def forward(self, graph, node_feature, global_feature, attack_edge_key):
        """
        :param graph: (dgl.Graph or dgl.BatchedGraph)
        :param node_feature: (pytorch Tensor) [ (Batched) # Nodes x node_feature dim]
        :param global_feature: (pytorch Tensor) [# Graphs x global_feature dim]
        :param attack_edge_key: (str)
        """

        node_updated = self.relational_enc(graph, node_feature)

        if type(graph) == dgl.BatchedDGLGraph:
            num_nodes = graph.batch_num_nodes
            num_nodes = torch.tensor(num_nodes)
            global_updated = torch.repeat_interleave(global_feature, num_nodes, dim=0)
        else:
            global_updated = global_feature.repeat((node_feature.shape[0], 1))

        module_input = torch.cat((node_updated, global_updated), dim=-1)

        move_argument = self.move_module(graph, module_input)
        hold_argument = self.hold_module(graph, module_input)
        attack_argument = self.attack_module(graph, module_input, attack_edge_key)
        return move_argument, hold_argument, attack_argument

    def compute_probs(self, graph, node_feature, global_feature, attack_edge_key='attack_edge'):
        move_arg, hold_arg, attack_arg = self.forward(graph, node_feature, global_feature, attack_edge_key)
        # Prepare un-normalized probabilities of attacks
        max_num_enemy = 0
        total_num_units = 0

        num_units = []
        for attack_a in attack_arg[0]:
            num_unit, num_enemy = attack_a.shape[0], attack_a.shape[1]
            total_num_units += num_unit
            num_units.append(num_unit)
            if max_num_enemy <= num_enemy:
                max_num_enemy = num_enemy

        attack_arg_out = torch.zeros(total_num_units, max_num_enemy)
        cum_num_units = 0
        for attack_a, num_unit in zip(attack_arg[0], num_units):
            attack_arg_out[cum_num_units:cum_num_units + num_unit, :attack_a.shape[1]] = attack_a
            attack_arg_out[cum_num_units:cum_num_units + num_unit, attack_a.shape[1]:] = 0
            cum_num_units += num_unit

        unnormed_ps = torch.cat((move_arg, hold_arg, attack_arg_out), dim=-1)
        ps = torch.nn.functional.softmax(unnormed_ps, dim=-1)
        log_ps = torch.log(ps + VERY_SMALL_NUMBER)
        unit_entropy = - torch.sum(log_ps * ps, dim=-1)  # per unit entropy

        log_p_move, log_p_hold, log_p_attack = torch.split(log_ps, [self.move_dim, 1, max_num_enemy], dim=1)

        return_dict = dict()
        return_dict['probs'] = ps
        return_dict['log_p_move'] = log_p_move
        return_dict['log_p_hold'] = log_p_hold
        return_dict['log_p_attack'] = log_p_attack
        return_dict['unit_entropy'] = unit_entropy
        return return_dict

    def get_action(self, graph, node_feature, global_feature,
                   ally_node_key='ally', attack_edge_key='attack_in_range'):
        """
        :param graph: (dgl.Graph or dgl.BatchedGraph)
        :param node_feature: (pytorch Tensor) [ (Batched) # Nodes x node_feature dim]
        :param global_feature: (pytorch Tensor) [# Graphs x global_feature dim]
        :param index2units: (list of index2unit dictionaries)
        :param ally_node_key: (str)
        :param attack_edge_key: (str)
        """

        prob_dict = self.compute_probs(graph=graph, node_feature=node_feature, global_feature=global_feature,
                                       attack_edge_key=attack_edge_key)
        probs = prob_dict['probs']

        ally_indices = graph.get_ntype_id(ally_node_key)
        ally_probs = probs[ally_indices, :]

        #
        if self.training:  # Sample from categorical dist
            dist = torch.distributions.Categorical(probs=ally_probs)
            nn_action = dist.sample()

        else:
            nn_action = ally_probs.argmax(dim=1)

        return nn_action
