import torch
from sc2rl.rl.rl_modules.ActionModules import MoveModule, HoldModule, AttackModule
from sc2rl.utils.graph_utils import get_filtered_node_index_by_type

from sc2rl.config.nn_configs import VERY_SMALL_NUMBER
from sc2rl.config.graph_configs import EDGE_IN_ATTACK_RANGE, NODE_ALLY


class ActorModule(torch.nn.Module):

    def __init__(self,
                 node_input_dim,
                 out_activation,
                 hidden_activation,
                 num_neurons: list = [64, 64],
                 move_dim: int = 4):
        super(ActorModule, self).__init__()
        self.move_dim = move_dim
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

    def forward(self, graph, node_feature, maximum_num_enemy, attack_edge_type_index):
        move_argument = self.move_module(graph, node_feature)
        hold_argument = self.hold_module(graph, node_feature)
        attack_argument = self.attack_module(graph, node_feature, maximum_num_enemy, attack_edge_type_index)
        return move_argument, hold_argument, attack_argument

    def compute_probs(self, graph, node_feature, maximum_num_enemy,
                      ally_node_type_index=NODE_ALLY,
                      attack_edge_type_index=EDGE_IN_ATTACK_RANGE):
        # get logits of each action
        move_arg, hold_arg, attack_arg = self.forward(graph, node_feature, maximum_num_enemy, attack_edge_type_index)

        # Prepare un-normalized probability of attacks

        unnormed_ps = torch.cat((move_arg, hold_arg, attack_arg), dim=-1)  # of all units including enemies

        ally_node_indices = get_filtered_node_index_by_type(graph, ally_node_type_index)
        unnormed_ps = unnormed_ps[ally_node_indices, :]  # of only ally units

        ally_tags = graph.ndata.pop('tag')
        ally_tags = ally_tags[ally_node_indices]
        if 'enemy_tag' in graph.ndata.keys():
            enemy_tags = graph.ndata.pop('enemy_tag')
        else:
            enemy_tags = torch.zeros_like(ally_tags).view(-1, 1)  # dummy

        enemy_tags = enemy_tags[ally_node_indices, :]

        ps = torch.nn.functional.softmax(unnormed_ps, dim=-1)
        log_ps = torch.log(ps + VERY_SMALL_NUMBER)
        unit_entropy = - torch.sum(log_ps * ps, dim=-1)  # per unit entropy
        log_p_move, log_p_hold, log_p_attack = torch.split(log_ps, [self.move_dim, 1, maximum_num_enemy], dim=1)

        return_dict = dict()
        # for RL training
        return_dict['probs'] = ps
        return_dict['log_p_move'] = log_p_move
        return_dict['log_p_hold'] = log_p_hold
        return_dict['log_p_attack'] = log_p_attack
        return_dict['unit_entropy'] = unit_entropy
        # for SC2 interfacing
        return_dict['ally_tags'] = ally_tags
        return_dict['enemy_tags'] = enemy_tags
        return return_dict

    def get_action(self, graph, node_feature, mamximum_num_enemy,
                   ally_node_type_index=NODE_ALLY,
                   attack_edge_type_index=EDGE_IN_ATTACK_RANGE):

        info_dict = self.compute_probs(graph=graph,
                                       node_feature=node_feature,
                                       maximum_num_enemy=mamximum_num_enemy,
                                       ally_node_type_index=ally_node_type_index,
                                       attack_edge_type_index=attack_edge_type_index)
        ally_probs = info_dict['probs']

        if self.training:  # Sample from categorical dist
            dist = torch.distributions.Categorical(probs=ally_probs)
            nn_actions = dist.sample()
        else:
            nn_actions = ally_probs.argmax(dim=1)
        return nn_actions, info_dict
