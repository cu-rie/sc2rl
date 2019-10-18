import torch
from sc2rl.rl.brains.BrainBase import BrainBase
from sc2rl.config.ConfigBase import ConfigBase

class MultiStepActorCriticBrainConfig(ConfigBase):

    def __init__(self,
                 MultiStepSha):



class MultiStepActorCriticBrain(BrainBase):

    def __init__(self, actor, critic, critic_target=None, critic2=None,
                 hyper_params):
        super(MultiStepActorCriticBrain, self).__init__()
        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        self.critic2 = critic2

        if critic2 is not None:
            self.double_q = True
        else:
            self.double_q = False

        if critic_target is not None:
            self.q_targeted = True
        else:
            self.q_targeted = False

    def fit_critic(self):
        pass

    def fit_actor(self):
        pass

    def compute_actor_loss(self,
                           num_time_steps,
                           hist_graph, hist_feature,
                           curr_graph, curr_feature, maximum_num_enemy):

        with torch.no_grad():
            qs = self.get_q(self.critic,
                            num_time_steps,
                            hist_graph, hist_feature,
                            curr_graph, curr_feature, maximum_num_enemy)

            if self.double_q:
                qs2 = self.get_q(self.critic2,
                                 num_time_steps,
                                 hist_graph, hist_feature,
                                 curr_graph, curr_feature, maximum_num_enemy)

        prob_dict = self.actor.compute_probs(num_time_steps,
                                             hist_graph, hist_feature,
                                             curr_graph, curr_feature, maximum_num_enemy)

        ps = prob_dict['probs']
        vs = (ps * qs).sum(1).deatch()
        policy_target = qs - vs

        if self.double_q:
            vs2 = (ps * qs2).sum(1).detach()
            policy_target2 = qs - vs2
            policy_target = torch.min(policy_target, policy_target2)





    @staticmethod
    def get_q(critic_net,
              num_time_steps,
              hist_graph, hist_feature,
              curr_graph, curr_feature, maximum_num_enemy):

        prob_dict = critic_net.compute_probs(num_time_steps,
                                             hist_graph, hist_feature,
                                             curr_graph, curr_feature, maximum_num_enemy)

        qs = prob_dict['unnormed_ps']
        return qs

    def get_exp_q(self, critic_net,
                  num_time_steps,
                  hist_graph, hist_feature,
                  curr_graph, curr_feature, maximum_num_enemy):

        q = self.get_q(critic_net,
                       num_time_steps,
                       hist_graph, hist_feature,
                       curr_graph, curr_feature, maximum_num_enemy)
        prob_dict = self.actor.compute_probs(num_time_steps,
                                             hist_graph, hist_feature,
                                             curr_graph, curr_feature, maximum_num_enemy)

        p = prob_dict['probs']
        exp_q = (q * p).sum(dim=1)
        return exp_q
