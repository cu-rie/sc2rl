import torch
from sc2rl.rl.modules.HierarchicalActor import HierarchicalActorModule, HierarchicalActorModuleConfig
from sc2rl.rl.modules.Actor import ActorModule, ActorModuleConfig
from sc2rl.rl.networks.MultiStepInputNetwork import MultiStepInputNetwork
from sc2rl.rl.networks.MultiStepInputGraphNetwork import MultiStepInputGraphNetwork
from sc2rl.config.ConfigBase import ConfigBase


class MultiStepInputActorConfig(ConfigBase):
    def __init__(self, multi_step_input_actor_conf=None, actor_conf=None):
        super(MultiStepInputActorConfig, self).__init__(
            multi_step_input_actor_conf=multi_step_input_actor_conf,
            actor_conf=actor_conf
        )

        self.multi_step_input_actor_conf = {
            'prefix': 'multi_step_input_actor_conf',
            'use_attention': False,
            'use_hierarchical_actor': True
        }
        self.actor_conf = {
            'prefix': 'actor_conf',
            'node_input_dim': 17,
            'out_activation': 'relu',
            'hidden_activation': 'mish',
            'num_neurons': [64, 64],
            'spectral_norm': False

        }


class MultiStepInputActor(torch.nn.Module):

    def __init__(self,
                 conf,
                 actor_conf=None,
                 # use_attention=False,
                 # use_hierarchical_actor=False
                 ):
        super(MultiStepInputActor, self).__init__()

        use_attention = conf.multi_step_input_actor_conf['use_attention']
        use_hierarchical_actor = conf.multi_step_input_actor_conf['use_hierarchical_actor']

        if actor_conf is None:
            rnn_hidden_dim = conf.gnn_conf.hist_rnn_conf['hidden_size']
            curr_enc_hidden_dim = conf.gnn_conf.curr_enc_conf['model_dim']
            if use_hierarchical_actor:
                actor_conf = HierarchicalActorModuleConfig().actor_conf
            else:
                actor_conf = ActorModuleConfig().actor_conf
            actor_conf['node_input_dim'] = rnn_hidden_dim + curr_enc_hidden_dim

        if use_attention:
            self.multi_step_input_net = MultiStepInputNetwork(conf.gnn_conf)
        else:
            self.multi_step_input_net = MultiStepInputGraphNetwork(conf.gnn_conf)

        if use_hierarchical_actor:
            self.actor = HierarchicalActorModule(actor_conf)
        else:
            self.actor = ActorModule(actor_conf)

    def forward(self,
                num_time_steps,
                hist_graph, hist_feature,
                curr_graph, curr_feature, maximum_num_enemy):
        hist_current_encoded_node_feature = self.multi_step_input_net(num_time_steps,
                                                                      hist_graph,
                                                                      hist_feature,
                                                                      curr_graph,
                                                                      curr_feature)
        move_argument, hold_argument, attack_argument = self.actor(curr_graph, hist_current_encoded_node_feature,
                                                                   maximum_num_enemy)
        return move_argument, hold_argument, attack_argument

    def compute_probs(self,
                      num_time_steps,
                      hist_graph, hist_feature,
                      curr_graph, curr_feature, maximum_num_enemy):
        hist_current_encoded_node_feature = self.multi_step_input_net(num_time_steps,
                                                                      hist_graph,
                                                                      hist_feature,
                                                                      curr_graph,
                                                                      curr_feature)

        _, prob_dict = self.actor.get_action(curr_graph,
                                             hist_current_encoded_node_feature,
                                             maximum_num_enemy)
        return prob_dict

    def get_action(self,
                   num_time_steps,
                   hist_graph, hist_feature,
                   curr_graph, curr_feature, maximum_num_enemy):

        info_dict = self.compute_probs(num_time_steps,
                                       hist_graph, hist_feature,
                                       curr_graph, curr_feature, maximum_num_enemy)

        ally_probs = info_dict['probs']
        if 'enemy_tag' in curr_graph.ndata.keys():
            _ = curr_graph.ndata.pop('enemy_tag')

        if self.training:  # Sample from categorical dist
            dist = torch.distributions.Categorical(probs=ally_probs)
            nn_actions = dist.sample()
        else:
            nn_actions = ally_probs.argmax(dim=1)
        return nn_actions, info_dict


if __name__ == "__main__":
    MultiStepInputActor(MultiStepInputActorConfig)
