import torch
from sc2rl.rl.brains.BrainBase import BrainBase
from sc2rl.config.ConfigBase import ConfigBase
from sc2rl.config.nn_configs import VERY_SMALL_NUMBER
from sc2rl.utils.graph_utils import get_index_mapper


class MultiStepPolicyGradientBrainConfig(ConfigBase):

    def __init__(self,
                 brain_conf=None,
                 fit_conf=None,
                 entropy_conf=None):
        self._brain_conf = {
            'prefix': 'brain_conf',
            'optimizer': 'Adam',
            'actor_lr': 1e-4,
            'gamma': 1.0
        }
        self.set_configs(self._brain_conf, brain_conf)

        self._fit_conf = {
            'prefix': 'brain_fit',
            'num_critic_pre_fit_steps': 5,
            'tau': 0.005,
            'actor_norm_clip_val': 1.0
        }
        self.set_configs(self._fit_conf, fit_conf)

        self._entropy_conf = {
            'prefix': 'brain_entropy',
            'optimizer': 'Adam',
            'auto_tune': True,
            'alpha': 0.01,
            'target_alpha': -(4 + 1 + 1),  # Expected minimal action dim : Move 4 + Hold 1 + Attack 1
            'lr': 1e-4
        }
        self.set_configs(self._entropy_conf, entropy_conf)

    @property
    def brain_conf(self):
        return self.get_conf(self._brain_conf)

    @property
    def fit_conf(self):
        return self.get_conf(self._fit_conf)

    @property
    def entropy_conf(self):
        return self.get_conf(self._entropy_conf)


class MultiStepPolicyGradientBrain(BrainBase):

    def __init__(self, conf, actor):
        super(MultiStepPolicyGradientBrain, self).__init__()
        self.actor = actor

        self.brain_conf = conf.brain_conf
        self.gamma = self.brain_conf['gamma']

        optimizer = self.get_optimizer(self.brain_conf['optimizer'])
        self.actor_optimizer = optimizer(self.actor.parameters(), lr=self.brain_conf['actor_lr'])

        self.fit_conf = conf.fit_conf
        self.entropy_conf = conf.entropy_conf

        if self.entropy_conf['auto_tune']:
            self.target_alpha = self.entropy_conf['target_alpha']
            self.log_alpha = torch.zeros(1, requires_grad=True)
            optimizer = self.get_optimizer(self.entropy_conf['optimizer'])
            self.alpha_optimizer = optimizer([self.log_alpha], lr=self.entropy_conf['lr'])

        else:
            self.log_alpha = torch.log(torch.ones(1) * self.entropy_conf['alpha'])

        self.fit_steps = 0

    def get_action(self,
                   num_time_steps,
                   hist_graph,
                   hist_feature,
                   curr_graph,
                   curr_feature,
                   maximum_num_enemy):

        nn_actions, info_dict = self.actor.get_action(num_time_steps,
                                                      hist_graph, hist_feature,
                                                      curr_graph, curr_feature, maximum_num_enemy)
        return nn_actions, info_dict

    def fit(self,
            num_time_steps,
            c_hist_graph, c_hist_feature,
            c_curr_graph, c_curr_feature,
            c_maximum_num_enemy,
            actions,
            n_hist_graph, n_hist_feature,
            n_curr_graph, n_curr_feature,
            n_maximum_num_enemy,
            rewards,
            dones):

        fit_dict = dict()
        self.fit_steps += 1
        return fit_dict

    def fit_actor(self,
                  num_time_steps, rewards,
                  hist_graph, hist_feature,
                  curr_graph, curr_feature, maximum_num_enemy):

        loss = self.compute_actor_loss(num_time_steps, rewards,
                                       hist_graph, hist_feature,
                                       curr_graph, curr_feature, maximum_num_enemy)

        self.clip_and_optimize(self.actor_optimizer, self.actor.parameters(), loss,
                               clip_val=self.fit_conf['actor_norm_clip_val'])

        fit_dict = dict()
        fit_dict['actor_loss'] = loss.detach().cpu().numpy()
        return fit_dict

    def compute_actor_loss(self,
                           num_time_steps, rewards,
                           hist_graph, hist_feature,
                           curr_graph, curr_feature, maximum_num_enemy):

        prob_dict = self.actor.compute_probs(num_time_steps,
                                             hist_graph, hist_feature,
                                             curr_graph, curr_feature, maximum_num_enemy)

        log_p_move = prob_dict['log_p_move']
        log_p_hold = prob_dict['log_p_hold']
        log_p_attack = prob_dict['log_p_attack']

        log_ps = torch.cat([log_p_move, log_p_hold, log_p_attack], dim=1)
        ps = prob_dict['probs']

        vs = (ps * qs).sum(1).detach()
        policy_target = qs - vs.view(-1, 1)

        if self.double_q:
            vs2 = (ps * qs2).sum(1).detach()
            policy_target2 = qs - vs2.view(-1, 1)
            policy_target = torch.min(policy_target, policy_target2)

        unmasked_loss = log_ps * (self.log_alpha.exp() * log_ps - policy_target)
        loss_mask = (log_ps > torch.log(torch.tensor(VERY_SMALL_NUMBER))).float()
        loss = (unmasked_loss * loss_mask).sum() / loss_mask.sum()
        return loss

    def fit_alpha(self,
                  num_time_steps,
                  hist_graph, hist_feature,
                  curr_graph, curr_feature, maximum_num_enemy):

        prob_dict = self.actor.compute_probs(num_time_steps,
                                             hist_graph, hist_feature,
                                             curr_graph, curr_feature, maximum_num_enemy)

        log_ps = prob_dict['log_ps']

        alpha_loss = (-self.log_alpha * (log_ps * self.target_alpha).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        ret_dict = dict()
        ret_dict['alpha_loss'] = alpha_loss
        ret_dict['alpha'] = self.log_alpha.exp().detach().cpu().numpy()
        return ret_dict
