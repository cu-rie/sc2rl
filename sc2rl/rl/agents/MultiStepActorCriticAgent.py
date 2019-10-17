import dgl
import torch

from sc2rl.rl.agents.AgentBase import AgentBase
from sc2rl.utils.sc2_utils import nn_action_to_sc2_action
from sc2rl.utils.graph_utils import get_largest_number_of_enemy_nodes
from sc2rl.utils.graph_utils import get_filtered_node_index_by_type
from sc2rl.config.graph_configs import NODE_ALLY


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
        hist_node_feature = hist_graph.ndata.pop('node_feature')
        curr_node_feature = curr_graph.ndata.pop('node_feature')
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

        hist_graph.ndata['node_feature'] = hist_node_feature
        curr_graph.ndata['node_feature'] = curr_node_feature

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
        # for c_g in c_graph:
        #     if 'node_type' in c_g.ndata:
        #         pass
        #     else:
        #         print("node_type non exist")
        #         import pdb
        #         pdb.set_trace()
        #
        # for n_g in n_graph:
        #     if 'node_type' in n_g.ndata:
        #         pass
        #     else:
        #         print("node_type non exist")
        #         import pdb
        #         pdb.set_trace()

        c_maximum_num_enemy = get_largest_number_of_enemy_nodes(c_graph)
        n_maximum_num_enemy = get_largest_number_of_enemy_nodes(n_graph)

        # casting actions to one torch tensor
        actions = torch.cat(actions).long()

        # 'c_graph' is now list of graphs
        c_ally_units = [len(get_filtered_node_index_by_type(graph, NODE_ALLY)) for graph in c_graph]
        c_ally_units = torch.Tensor(c_ally_units).long()

        # prepare rewards
        rewards = torch.Tensor(rewards)
        rewards = rewards.repeat_interleave(c_ally_units, dim=0)

        # preparing dones
        dones = torch.Tensor(dones)
        dones = dones.repeat_interleave(c_ally_units, dim=0)

        # batching graphs
        list_c_h_graph = [g for L in c_h_graph for g in L]
        list_n_h_graph = [g for L in n_h_graph for g in L]
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
                                                 n_num_time_steps=hist_num_time_steps,
                                                 n_h_graph=n_h_graph,
                                                 n_h_node_feature=n_h_node_feature,
                                                 n_graph=n_graph,
                                                 n_node_feature=n_node_feature,
                                                 n_maximum_num_enemy=n_maximum_num_enemy,
                                                 actions=actions,
                                                 rewards=rewards,
                                                 dones=dones)

        return actor_loss, critic_loss
