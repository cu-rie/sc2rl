import torch
from sc2rl.rl.modules.Actor import ActorModule
from sc2rl.rl.networks.MultiStepInputNetwork import MultiStepInputNetwork


class MultiStepInputActor(torch.nn.Module):

    def __init__(self, multi_step_input_network_conf,
                 actor_conf):
        self.multi_step_input_net = MultiStepInputNetwork(multi_step_input_network_conf)
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

        prob_dict = self.actor.get_action(curr_graph,
                                          hist_current_encoded_node_feature,
                                          maximum_num_enemy)
        return prob_dict

    def get_action(self, num_time_steps,
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
