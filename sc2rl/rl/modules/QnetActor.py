import torch
from sc2rl.rl.modules.Actions import MoveModule, HoldModule, AttackModule
from sc2rl.config.graph_configs import EDGE_IN_ATTACK_RANGE, NODE_ALLY
from sc2rl.utils.graph_utils import get_filtered_node_index_by_type


class Qnet(torch.nn.Module):

    def __init__(self,
                 conf,
                 move_dim: int = 4):
        super(Qnet, self).__init__()
        self.conf = conf
        self.move_dim = move_dim
        node_input_dim = self.hyper_param['node_input_dim']
        out_activation = self.hyper_param['out_activation']
        hidden_activation = self.hyper_param['hidden_activation']
        num_neurons = self.hyper_param['num_neurons']

        self.move_module = MoveModule(node_dim=node_input_dim,
                                      move_dim=move_dim,
                                      num_neurons=num_neurons,
                                      hidden_activation=hidden_activation,
                                      out_activation=out_activation)

        self.hold_module = HoldModule(node_dim=node_input_dim,
                                      num_neurons=num_neurons,
                                      hidden_activation=hidden_activation,
                                      out_activation=out_activation)

        self.attack_module = AttackModule(node_dim=node_input_dim,
                                          num_neurons=num_neurons,
                                          hidden_activation=hidden_activation,
                                          out_activation=out_activation)

    def forward(self, graph, node_feature, maximum_num_enemy, attack_edge_type_index=EDGE_IN_ATTACK_RANGE):
        move_argument = self.move_module(graph, node_feature)
        hold_argument = self.hold_module(graph, node_feature)
        attack_argument = self.attack_module(graph, node_feature, maximum_num_enemy, attack_edge_type_index)
        return move_argument, hold_argument, attack_argument

    def compute_qs(self, graph, node_feature, maximum_num_enemy,
                   ally_node_type_index=NODE_ALLY,
                   attack_edge_type_index=EDGE_IN_ATTACK_RANGE):
        # get logits of each action
        move_arg, hold_arg, attack_arg = self(graph, node_feature, maximum_num_enemy, attack_edge_type_index)
        qs = torch.cat((move_arg, hold_arg, attack_arg), dim=-1)  # of all units including enemies

        ally_node_indices = get_filtered_node_index_by_type(graph, ally_node_type_index)
        qs = qs[ally_node_indices, :]  # of only ally units

        ally_tags = graph.ndata['tag']
        ally_tags = ally_tags[ally_node_indices]
        if 'enemy_tag' in graph.ndata.keys():
            enemy_tags = graph.ndata['enemy_tag']
        else:
            enemy_tags = torch.zeros_like(ally_tags).view(-1, 1)  # dummy

        enemy_tags = enemy_tags[ally_node_indices, :]

        return_dict = dict()
        # for RL training
        return_dict['qs'] = qs

        # for SC2 interfacing
        return_dict['ally_tags'] = ally_tags
        return_dict['enemy_tags'] = enemy_tags
        return return_dict

    def get_action(self, graph, node_feature, maximum_num_enemy,
                   ally_node_type_index=NODE_ALLY,
                   attack_edge_type_index=EDGE_IN_ATTACK_RANGE):

        info_dict = self.compute_qs(graph=graph,
                                    node_feature=node_feature,
                                    maximum_num_enemy=maximum_num_enemy,
                                    ally_node_type_index=ally_node_type_index,
                                    attack_edge_type_index=attack_edge_type_index)

        ally_qs = info_dict['qs']

        if 'enemy_tag' in graph.ndata.keys():
            _ = graph.ndata.pop('enemy_tag')

        if torch.rand(1) <= self.eps:
            raise NotImplementedError
            #nn_actions = dist.sample()
        else:
            nn_actions = ally_qs.argmax(dim=1)
        return nn_actions, info_dict
