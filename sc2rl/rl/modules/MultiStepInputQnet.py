import torch
from sc2rl.rl.networks.MultiStepInputNetwork import MultiStepInputNetwork, MultiStepInputNetworkConfig
from sc2rl.rl.networks.MultiStepInputGraphNetwork import MultiStepInputGraphNetwork, MultiStepInputGraphNetworkConfig
from sc2rl.rl.modules.QnetActor import QnetActor

from sc2rl.config.ConfigBase_refac import ConfigBase
from sc2rl.config.nn_configs import VERY_LARGE_NUMBER


class MultiStepInputQnetConfig(ConfigBase):

    def __init__(self, multi_step_input_qnet_conf=None, qnet_actor_conf=None):
        super(MultiStepInputQnetConfig, self).__init__(multi_step_input_qnet_conf=multi_step_input_qnet_conf,
                                                       qnet_actor_conf=qnet_actor_conf)
        self.multi_step_input_qnet_conf = {
            'prefix': 'multi_step_input_qnet_conf',
            'use_attention': False,
            'eps': 1.0
        }

        self.qnet_actor_conf = {
            'prefix': 'qnet_actor_conf',
            'node_input_dim': 17,
            'out_activation': None,
            'hidden_activation': 'mish',
            'num_neurons': [64, 64],
            'spectral_norm': False
        }


class MultiStepInputQnet(torch.nn.Module):

    def __init__(self, conf):
        super(MultiStepInputQnet, self).__init__()

        use_attention = conf.multi_step_input_qnet_conf['use_attention']
        qnet_actor_conf = conf.qnet_actor_conf

        self.eps = conf.multi_step_input_qnet_conf['eps']

        if use_attention:
            self.multi_step_input_net = MultiStepInputNetwork(MultiStepInputNetworkConfig())
        else:
            self.multi_step_input_net = MultiStepInputGraphNetwork(MultiStepInputGraphNetworkConfig())

        qnet_actor_conf['node_input_dim'] = self.multi_step_input_net.out_dim
        self.qnet = QnetActor(qnet_actor_conf)

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
                   curr_graph, curr_feature, maximum_num_enemy,
                   eps):

        q_dict = self.compute_qs(num_time_steps,
                                 hist_graph, hist_feature,
                                 curr_graph, curr_feature, maximum_num_enemy)

        ally_qs = q_dict['qs']

        device = ally_qs.device

        if 'enemy_tag' in curr_graph.ndata.keys():
            _ = curr_graph.ndata.pop('enemy_tag')

        if torch.rand(1, device=device) <= eps:
            sampling_mask = torch.ones_like(ally_qs, device=device)
            sampling_mask[ally_qs <= -VERY_LARGE_NUMBER] = -VERY_LARGE_NUMBER
            dist = torch.distributions.categorical.Categorical(logits=sampling_mask)
            nn_actions = dist.sample()
        else:
            nn_actions = ally_qs.argmax(dim=1)
        return nn_actions, q_dict


if __name__ == "__main__":
    conf = MultiStepInputQnetConfig()
    MultiStepInputQnet(conf)
