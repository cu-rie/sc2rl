import itertools

import dgl
import torch

from sc2rl.rl.agents.AgentBase import AgentBase
from sc2rl.utils.sc2_utils import nn_action_to_sc2_action
from sc2rl.utils.graph_utils import get_largest_number_of_enemy_nodes


class MultiStepActorCriticAgent(AgentBase):
    def __init__(self, brain, buffer):
        super(MultiStepActorCriticAgent, self).__init__(brain, buffer)

    def forward(self, *args, **kwargs):
        return self.get_action(*args, **kwargs)

    def get_action(self,
                   hist_graph,
                   curr_graph,
                   tag2unit_dict):
        assert isinstance(curr_graph, dgl.DGLGraph), "get action is designed to work on a single graph!"
        num_time_steps = hist_graph.batch_size
        hist_node_feature = hist_graph.ndata.pop['node_feautre']
        curr_node_feature = curr_graph.ndata.pop['node_feature']
        maximum_num_enemy = get_largest_number_of_enemy_nodes([curr_graph])

        nn_actions, info_dict = self.brain.get_action(num_time_steps,
                                                      hist_graph, hist_node_feature,
                                                      curr_graph, curr_node_feature, maximum_num_enemy)

        ally_tags = info_dict['ally_tags']
        enemy_tags = info_dict['enemy_tags']

        sc2_actions = nn_action_to_sc2_action(nn_actions=nn_actions,
                                              ally_tags=ally_tags,
                                              enemy_tags=enemy_tags,
                                              tag2unit_dict=tag2unit_dict)

        return nn_actions, sc2_actions

    def fit(self, batch_size, hist_num_time_steps):
        # the prefix 'c' indicates #current# time stamp inputs
        # the prefix 'n' indicates #next# time stamp inputs

        # expected specs:
        # bs = batch_size, nt = hist_num_time_steps
        # 'h_graph' = list of graph lists [[g_(0,0), g_(0,1), ... g_(0,nt)],
        #                                  [g_(1,0), g_(1,1), ..., g_(1,nt)],
        #                                  [g_(2,0), ..., g_(bs, 0), ... g_(bs, nt)]]
        # 'graph' = list of graphs  [g_(0), g_(1), ..., g_(bs)]

        c_h_graph, c_graph, actions, rewards, n_h_graph, n_graph, dones = self.buffer.sample(batch_size)

        # inferring maximum num enemies
        c_maximum_num_enemy = get_largest_number_of_enemy_nodes(c_graph)
        n_maximum_num_enemy = get_largest_number_of_enemy_nodes(n_graph)

        # batching graphs
        list_c_h_graph = itertools.chain.from_iterable(c_h_graph)
        list_n_h_graph = itertools.chain.from_iterable(n_h_graph)
        c_h_graph = dgl.batch(list_c_h_graph)
        n_h_graph = dgl.batch(list_n_h_graph)

        c_graph = dgl.batch(c_graph)
        n_graph = dgl.batch(n_graph)

        c_h_node_feature = c_h_graph.ndata.pop('node_feature')
        c_node_feature = c_graph.ndata.pop('node_feature')

        n_h_node_feature = n_h_graph.ndata.pop('node_feature')
        n_node_feature = n_graph.ndata.pop('node_feature')

        actor_loss, critic_loss = self.brain.fit(c_num_time_steps=hist_num_time_steps,
                                                 c_h_graph=c_h_graph,
                                                 c_h_node_feature=c_h_node_feature,
                                                 c_graph=c_graph,
                                                 c_node_feature=c_node_feature,
                                                 c_maximum_num_enemy=c_maximum_num_enemy,
                                                 n_num_time_steps=n_maximum_num_enemy,
                                                 n_h_graph=n_h_graph,
                                                 n_h_node_feature=n_h_node_feature,
                                                 n_graph=n_graph,
                                                 n_node_feature=n_node_feature,
                                                 n_maximum_num_enemy=n_maximum_num_enemy,
                                                 actions=actions,
                                                 rewards=rewards,
                                                 dones=dones)

        return actor_loss, critic_loss
