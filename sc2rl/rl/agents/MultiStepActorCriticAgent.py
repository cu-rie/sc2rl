import dgl
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

    def fit(self, batch_size):
        #self.brain.fit()