from copy import deepcopy

import torch
import numpy as np

from sc2rl.rl.networks.MultiStepInputNetwork import MultiStepInputNetwork
from sc2rl.rl.networks.MultiStepInputGraphNetwork import MultiStepInputGraphNetwork
from sc2rl.rl.networks.RelationalGraphNetwork import RelationalGraphNetwork
from sc2rl.rl.modules.QnetActor import QnetActor

from sc2rl.config.ConfigBase import ConfigBase
from sc2rl.config.nn_configs import VERY_LARGE_NUMBER

from sc2rl.utils.debug_utils import dn


class MultiStepInputQnetConfig(ConfigBase):

    def __init__(self,
                 multi_step_input_qnet_conf=None,
                 qnet_actor_conf=None):
        super(MultiStepInputQnetConfig, self).__init__(multi_step_input_qnet_conf=multi_step_input_qnet_conf,
                                                       qnet_actor_conf=qnet_actor_conf)

        self.multi_step_input_qnet_conf = {
            'prefix': 'multi_step_input_qnet_conf',
            'use_attention': False,
            'eps': 1.0,
            'exploration_method': 'eps_greedy',
        }

        self.qnet_actor_conf = {
            'prefix': 'qnet_actor_conf',
            'node_input_dim': 17,
            'out_activation': None,
            'hidden_activation': 'mish',
            'num_neurons': [64, 64],
            'spectral_norm': False,
        }


# def generate_hierarchical_sampling_mask(action_spaces, q_mask=None):
#     n_agent = action_spaces.shape[0]
#     n_clusters = 2
#     action_start_indices = [0, 5]
#     action_end_indices = [5, None]
#
#     device = action_spaces.device
#
#     if n_agent / n_clusters >= 2.0:
#         n_agents_per_cluster = torch.randint(low=1, high=int(np.floor(n_agent / n_clusters)), size=(n_clusters - 1,))
#         _the_last_cluster_n = n_agent - torch.sum(n_agents_per_cluster).view(-1, )
#         n_agents_per_cluster = torch.cat([n_agents_per_cluster, _the_last_cluster_n], dim=0)
#         agent_indices = torch.randperm(n_agent)
#         splitted_agent_indices = torch.split(agent_indices, n_agents_per_cluster.tolist())
#
#         mask = torch.ones_like(action_spaces, device=device) * -VERY_LARGE_NUMBER
#         for agent_indices, action_start_index, action_end_index in zip(splitted_agent_indices,
#                                                                        action_start_indices,
#                                                                        action_end_indices):
#             mask[agent_indices, action_start_index:action_end_index] = 1
#     else:
#         mask = torch.ones_like(action_spaces, device=device)
#         mask[action_spaces <= -VERY_LARGE_NUMBER] = -VERY_LARGE_NUMBER
#
#     if q_mask is not None:
#         mask = mask * q_mask
#
#     return mask


def generate_hierarchical_sampling_mask(q_mask, use_hold):
    n_agent = q_mask.shape[0]
    n_clusters = 2  # Consider (Move & Hold) cluster and Attack cluster
    action_start_indices = [0, 5]
    action_end_indices = [5, None]
    can_attacks = (q_mask[:, 5:].sum(1) >= 1)

    if n_agent / n_clusters >= 2.0:
        n_agent_in_attack = torch.randint(low=1, high=int(np.floor(n_agent / n_clusters)), size=(n_clusters - 1,))
        n_agent_in_attack = min(n_agent_in_attack, can_attacks.sum())

        can_attack_agent_indices = can_attacks.nonzero()  # indices of agents who can attack
        should_move_hold = (~can_attacks).nonzero()

        mask = torch.ones_like(q_mask, device=q_mask.device)


        perm = torch.randperm(len(can_attack_agent_indices))
        attack_idx = perm[:n_agent_in_attack]
        move_hold_among_attackable = perm[n_agent_in_attack:]

        attack_agent_idx = can_attack_agent_indices[attack_idx]
        move_hold_agent_idx = torch.cat([can_attack_agent_indices[move_hold_among_attackable],
                                         should_move_hold],
                                        dim=0)

        # mask-out (make 0 prob. to be sampled) for move and hold
        mask[attack_agent_idx, action_start_indices[0]:action_end_indices[0]] = - VERY_LARGE_NUMBER

        # mask-out (make 0 prob. to be sampled) for attack
        mask[move_hold_agent_idx, action_start_indices[1]:action_end_indices[1]] = - VERY_LARGE_NUMBER

        # post process mask to be attack appropriate
        row, col = torch.where(q_mask == 0)
        mask[row, col] = -VERY_LARGE_NUMBER
    else:
        mask = torch.ones_like(q_mask, device=q_mask.device)
        mask[q_mask <= 0] = -VERY_LARGE_NUMBER

    if not use_hold:
        mask[:, 4] = -VERY_LARGE_NUMBER

    return mask


class MultiStepInputQnet(torch.nn.Module):

    def __init__(self, conf):
        super(MultiStepInputQnet, self).__init__()

        use_attention = conf.multi_step_input_qnet_conf['use_attention']
        qnet_actor_conf = conf.qnet_actor_conf

        self.eps = conf.multi_step_input_qnet_conf['eps']
        self.exploration_method = conf.multi_step_input_qnet_conf['exploration_method']

        if use_attention:
            self.multi_step_input_net = MultiStepInputNetwork(conf.gnn_conf)
        else:
            self.multi_step_input_net = MultiStepInputGraphNetwork(conf.gnn_conf)
            # self.multi_step_input_net = RelationalGraphNetwork(conf.gnn_conf)

        qnet_actor_conf['node_input_dim'] = self.multi_step_input_net.out_dim
        self.qnet = QnetActor(qnet_actor_conf)
        self.use_hold = qnet_actor_conf['use_hold']

    def forward(self, *args):
        pass

    def compute_qs(self,
                   num_time_steps,
                   hist_graph, hist_feature,
                   curr_graph, curr_feature, maximum_num_enemy):

        hist_current_encoded_node_feature = self.multi_step_input_net(num_time_steps,
                                                                      hist_graph,
                                                                      hist_feature,
                                                                      curr_graph,
                                                                      curr_feature)

        q_dict = self.qnet.compute_qs(curr_graph,
                                      hist_current_encoded_node_feature,
                                      maximum_num_enemy)

        q_dict['hidden_feat'] = hist_current_encoded_node_feature

        return q_dict

    def get_action(self,
                   num_time_steps,
                   hist_graph, hist_feature,
                   curr_graph, curr_feature, maximum_num_enemy,
                   eps):

        q_dict = self.compute_qs(num_time_steps,
                                 hist_graph, hist_feature,
                                 curr_graph, curr_feature, maximum_num_enemy)

        ally_qs = q_dict['qs']

        device = ally_qs.device

        if 'enemy_tag' in curr_graph.ndata.keys():
            _ = curr_graph.ndata.pop('enemy_tag')

        if self.training:
            if self.exploration_method == "eps_greedy":
                if torch.rand(1, device=device) <= eps:
                    sampling_mask = torch.ones_like(ally_qs, device=device)
                    sampling_mask[ally_qs <= -VERY_LARGE_NUMBER] = -VERY_LARGE_NUMBER
                    dist = torch.distributions.categorical.Categorical(logits=sampling_mask)
                    nn_actions = dist.sample()
                else:
                    nn_actions = ally_qs.argmax(dim=1)
            elif self.exploration_method == "clustered_random":
                if torch.rand(1, device=device) <= eps:
                    q_mask = torch.ones_like(ally_qs, device=device)
                    q_mask[ally_qs <= -VERY_LARGE_NUMBER] = 0
                    sampling_mask = generate_hierarchical_sampling_mask(q_mask, self.use_hold)
                    dist = torch.distributions.categorical.Categorical(logits=sampling_mask)
                    nn_actions = dist.sample()
                else:
                    nn_actions = ally_qs.argmax(dim=1)
            elif self.exploration_method == "noisy_net":
                nn_actions = ally_qs.argmax(dim=1)
            else:
                raise RuntimeError("Not admissible exploration methods.")
        else:
            nn_actions = ally_qs.argmax(dim=1)
        return nn_actions, q_dict


if __name__ == "__main__":
    conf = MultiStepInputQnetConfig()
    MultiStepInputQnet(conf)
