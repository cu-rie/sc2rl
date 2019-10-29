import torch
from sc2rl.rl.networks.MultiStepInputNetwork import MultiStepInputNetwork
from sc2rl.rl.networks.MultiStepInputGraphNetwork import MultiStepInputGraphNetwork
from sc2rl.rl.modules.QnetActor import QnetActor


class MultiStepInputQnet(torch.nn.Module):

    def __init__(self, conf):
        super(MultiStepInputQnet, self).__init__()

        use_attention = conf.multi_step_input_qnet_conf['use_attention']
        multi_step_input_network_conf = conf.multi_step_input_network_conf
        qnet_conf = conf.qnet_conf

        self.eps = conf.multi_step_input_qnet_conf['eps']

        if use_attention:
            self.multi_step_input_net = MultiStepInputNetwork(multi_step_input_network_conf)
        else:
            self.multi_step_input_net = MultiStepInputGraphNetwork(multi_step_input_network_conf)

        self.qnet = QnetActor(qnet_conf)

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

        return q_dict

    def get_action(self,
                   num_time_steps,
                   hist_graph, hist_feature,
                   curr_graph, curr_feature, maximum_num_enemy):

        q_dict = self.compute_qs(num_time_steps,
                                 hist_graph, hist_feature,
                                 curr_graph, curr_feature, maximum_num_enemy)

        ally_qs = q_dict['qs']

        if 'enemy_tag' in curr_graph.ndata.keys():
            _ = curr_graph.ndata.pop('enemy_tag')

        if torch.rand(1) <= self.eps:
            raise NotImplementedError
            # nn_actions = dist.sample()
        else:
            nn_actions = ally_qs.argmax(dim=1)
        return nn_actions, q_dict
