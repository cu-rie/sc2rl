import torch
from sc2rl.rl.modules.Actions import MoveModule, HoldModule, AttackModule
from sc2rl.config.graph_configs import EDGE_IN_ATTACK_RANGE, NODE_ALLY, EDGE_ENEMY
from sc2rl.utils.graph_utils import get_filtered_node_index_by_type
from sc2rl.nn.diffPool import DiffPoolLayer
from sc2rl.utils.debug_utils import dn


class HierarchicalQnetActor(torch.nn.Module):

    def __init__(self,
                 conf,
                 move_dim: int = 4):
        super(HierarchicalQnetActor, self).__init__()
        self.conf = conf
        self.move_dim = move_dim
        self.use_concat_input = self.conf['use_concat_input']
        node_input_dim = self.conf['node_input_dim']
        if self.use_concat_input:
            node_input_dim = self.conf['node_input_dim'] + self.conf['init_node_dim']
            # self.ln = torch.nn.LayerNorm(node_input_dim)

        out_activation = self.conf['out_activation']
        hidden_activation = self.conf['hidden_activation']
        num_neurons = self.conf['num_neurons']
        spectral_norm = self.conf['spectral_norm']

        self.attack_edge_type_index = self.conf['attack_edge_type_index']

        # hierarchical pooling
        num_groups = self.conf['num_groups']
        pooling_op = self.conf['pooling_op']
        pooling_init = self.conf['pooling_init']
        self.pooling_using_initial = True

        use_hold = self.conf['use_hold']
        use_tanh = self.conf['use_tanh']
        use_noisy = self.conf['use_noisy']

        self.hold_module = HoldModule(node_dim=node_input_dim,
                                      num_neurons=num_neurons,
                                      hidden_activation=hidden_activation,
                                      out_activation=out_activation,
                                      spectral_norm=spectral_norm,
                                      use_hold=use_hold,
                                      use_noisy=use_noisy)

        self.attack_module = AttackModule(node_dim=node_input_dim,
                                          num_neurons=num_neurons,
                                          hidden_activation=hidden_activation,
                                          out_activation=out_activation,
                                          spectral_norm=spectral_norm,
                                          use_noisy=use_noisy)

        if use_tanh:
            self.move_module = MoveModule(node_dim=node_input_dim,
                                          move_dim=move_dim,
                                          num_neurons=num_neurons,
                                          hidden_activation='tanh',
                                          out_activation='tanh',
                                          spectral_norm=spectral_norm,
                                          use_noisy=use_noisy)

            self.grouping_module = DiffPoolLayer(node_dim=node_input_dim,
                                                 num_neurons=num_neurons,
                                                 pooling_op=pooling_op,
                                                 hidden_activation='tanh',
                                                 num_groups=num_groups,
                                                 spectral_norm=spectral_norm,
                                                 pooling_init=pooling_init,
                                                 use_noisy=use_noisy)

        else:

            self.move_module = MoveModule(node_dim=node_input_dim,
                                          move_dim=move_dim,
                                          num_neurons=num_neurons,
                                          hidden_activation=hidden_activation,
                                          out_activation=out_activation,
                                          spectral_norm=spectral_norm,
                                          use_noisy=use_noisy)

            self.grouping_module = DiffPoolLayer(node_dim=node_input_dim,
                                                 num_neurons=num_neurons,
                                                 pooling_op=pooling_op,
                                                 num_groups=num_groups,
                                                 spectral_norm=spectral_norm,
                                                 pooling_init=pooling_init,
                                                 use_noisy=use_noisy)

    def forward(self, graph, node_feature, maximum_num_enemy, attack_edge_type_index):
        if self.use_concat_input:
            node_feature = torch.cat([node_feature, graph.ndata['init_node_feature']], dim=1)

        move_argument = self.move_module(graph, node_feature)
        hold_argument = self.hold_module(graph, node_feature)
        attack_argument = self.attack_module(graph, node_feature, maximum_num_enemy, attack_edge_type_index)
        _, assignment, normalized_score = self.grouping_module(graph, node_feature)

        return move_argument, hold_argument, attack_argument, assignment, normalized_score

    def compute_qs(self, graph, node_feature, maximum_num_enemy,
                   ally_node_type_index=NODE_ALLY,
                   attack_edge_type_index=None):

        if attack_edge_type_index is None:
            attack_edge_type_index = self.attack_edge_type_index

        move_arg, hold_arg, attack_arg, assignment, normalized_score = self(graph, node_feature, maximum_num_enemy,
                                                                            attack_edge_type_index)
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
        return_dict['assignment'] = assignment
        return_dict['normalized_score'] = normalized_score

        # for SC2 interfacing
        return_dict['ally_tags'] = ally_tags
        return_dict['enemy_tags'] = enemy_tags
        return return_dict
