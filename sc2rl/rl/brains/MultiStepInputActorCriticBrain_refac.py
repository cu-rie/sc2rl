import torch
from sc2rl.rl.brains.BrainBase import BrainBase
from sc2rl.config.ConfigBase import ConfigBase
from sc2rl.config.nn_configs import VERY_SMALL_NUMBER
from sc2rl.utils.graph_utils import get_index_mapper


class MultiStepActorCriticBrainConfig(ConfigBase):

    def __init__(self,
                 brain_conf=None,
                 fit_conf=None,
                 entropy_conf=None):
        self._brain_conf = {
            'prefix': 'brain_conf',
            'optimizer': 'Adam',
            'actor_lr': 1e-4,
            'critic_lr': 1e-3,
            'gamma': 1.0
        }
        self.set_configs(self._brain_conf, brain_conf)

        self._fit_conf = {
            'prefix': 'brain_fit',
            'num_critic_pre_fit_steps': 5,
            'tau': 0.005,
            'critic_norm_clip_val': 1.0,
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


class MultiStepActorCriticBrain(BrainBase):

    def __init__(self, conf, actor, critic,
                 critic_target=None, critic2=None, critic2_target=None):
        super(MultiStepActorCriticBrain, self).__init__()
        self.actor = actor

        self.critic = critic
        self.critic_target = critic_target

        self.critic2 = critic2
        self.critic2_target = critic2_target

        self.brain_conf = conf.brain_conf
        self.gamma = self.brain_conf['gamma']

        optimizer = self.get_optimizer(self.brain_conf['optimizer'])
        self.actor_optimizer = optimizer(self.actor.parameters(), lr=self.brain_conf['actor_lr'])
        self.critic_optimizer = optimizer(self.critic.parameters(), lr=self.brain_conf['critic_lr'])

        if critic2 is not None:
            self.double_q = True
            self.critic2_optimizer = optimizer(self.critic2.parameters(), lr=self.brain_conf['critic_lr'])
        else:
            self.double_q = False

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
        if self.fit_steps >= self.fit_conf['num_critic_pre_fit_steps']:
            alpha_fit_dict = self.fit_alpha(num_time_steps,
                                            hist_graph=c_hist_graph,
                                            hist_feature=c_hist_feature,
                                            curr_graph=c_curr_graph,
                                            curr_feature=c_curr_feature,
                                            maximum_num_enemy=c_maximum_num_enemy)

            actor_fit_dict = self.fit_actor(num_time_steps=num_time_steps,
                                            hist_graph=c_hist_graph,
                                            hist_feature=c_hist_feature,
                                            curr_graph=c_curr_graph,
                                            curr_feature=c_curr_feature,
                                            maximum_num_enemy=c_maximum_num_enemy)
            fit_dict.update(alpha_fit_dict)
            fit_dict.update(actor_fit_dict)

        critic_fit_dict = self.fit_critic(num_time_steps=num_time_steps,

                                          c_hist_graph=c_hist_graph,
                                          c_hist_feature=c_hist_feature,
                                          c_curr_graph=c_curr_graph,
                                          c_curr_feature=c_curr_feature,
                                          c_maximum_num_enemy=c_maximum_num_enemy,
                                          actions=actions,
                                          n_hist_graph=n_hist_graph,
                                          n_hist_feature=n_hist_feature,
                                          n_curr_graph=n_curr_graph,
                                          n_curr_feature=n_curr_feature,
                                          n_maximum_num_enemy=n_maximum_num_enemy,
                                          rewards=rewards,
                                          dones=dones)
        fit_dict.update(critic_fit_dict)
        self.fit_steps += 1
        return fit_dict

    def fit_critic(self,
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

        loss = self.compute_critic_loss(self.critic,
                                        num_time_steps,
                                        c_hist_graph, c_hist_feature,
                                        c_curr_graph, c_curr_feature,
                                        c_maximum_num_enemy,
                                        actions,
                                        n_hist_graph, n_hist_feature,
                                        n_curr_graph, n_curr_feature,
                                        n_maximum_num_enemy,
                                        rewards,
                                        dones,
                                        target_net=self.critic_target)

        self.clip_and_optimize(self.critic_optimizer, self.critic.parameters(), loss,
                               clip_val=self.fit_conf['critic_norm_clip_val'])

        self.update_target_network(self.fit_conf['tau'], self.critic, self.critic_target)

        fit_dict['critic_loss'] = loss.detach().cpu().numpy()

        if self.double_q:
            loss2 = self.compute_critic_loss(self.critic2,
                                             num_time_steps,
                                             c_hist_graph, c_hist_feature,
                                             c_curr_graph, c_curr_feature,
                                             c_maximum_num_enemy,
                                             actions,
                                             n_hist_graph, n_hist_feature,
                                             n_curr_graph, n_curr_feature,
                                             n_maximum_num_enemy,
                                             rewards,
                                             dones,
                                             target_net=self.critic2_target)

            self.clip_and_optimize(self.critic2_optimizer, self.critic2.parameters(), loss2,
                                   clip_val=self.fit_conf['critic_norm_clip_val'])

            self.update_target_network(self.fit_conf['tau'], self.critic2, self.critic2_target)
            fit_dict['critic_loss2'] = loss2.detach().cpu().numpy()

        return fit_dict

    def compute_critic_loss(self,
                            critic_net,
                            num_time_steps,
                            c_hist_graph, c_hist_feature,
                            c_curr_graph, c_curr_feature,
                            c_maximum_num_enemy,
                            actions,
                            n_hist_graph, n_hist_feature,
                            n_curr_graph, n_curr_feature,
                            n_maximum_num_enemy,
                            rewards,
                            dones,
                            target_net=None):

        if target_net is None:
            target_net = critic_net

        # cur_q : [#. current ally units x #. actions]
        cur_q = self.get_q(critic_net,
                           num_time_steps,
                           c_hist_graph, c_hist_feature,
                           c_curr_graph, c_curr_feature, c_maximum_num_enemy)

        # cur_q : [#. current ally units]
        cur_q = cur_q.gather(-1, actions.unsqueeze(-1)).squeeze(dim=-1)

        # The number of allies in the current (batched) graph may differ from the one of the next graph
        target_q = torch.zeros_like(cur_q)

        with torch.no_grad():
            exp_target_q, entropy = self.get_exp_q(target_net,
                                                   num_time_steps,
                                                   n_hist_graph, n_hist_feature,
                                                   n_curr_graph, n_curr_feature,
                                                   n_maximum_num_enemy)

        unsorted_target_q = exp_target_q + self.log_alpha.exp().detach() * entropy  # [#. next ally_units]
        cur_idx, next_idx = get_index_mapper(c_curr_graph, n_curr_graph)
        target_q[cur_idx] = unsorted_target_q[next_idx]
        target_q = rewards + self.gamma * target_q * (1 - dones)
        loss = torch.nn.functional.mse_loss(input=cur_q, target=target_q)
        return loss

    def fit_actor(self,
                  num_time_steps,
                  hist_graph, hist_feature,
                  curr_graph, curr_feature, maximum_num_enemy):

        loss = self.compute_actor_loss(num_time_steps,
                                       hist_graph, hist_feature,
                                       curr_graph, curr_feature, maximum_num_enemy)

        self.clip_and_optimize(self.actor_optimizer, self.actor.parameters(), loss,
                               clip_val=self.fit_conf['actor_norm_clip_val'])

        fit_dict = dict()
        fit_dict['actor_loss'] = loss.detach().cpu().numpy()
        return fit_dict

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

        log_p_move = prob_dict['log_p_move']
        log_p_hold = prob_dict['log_p_hold']
        log_p_attack = prob_dict['log_p_attack']

        log_ps = torch.cat([log_p_move, log_p_hold, log_p_attack], dim=1)
        ps = prob_dict['probs']

        vs = (ps * qs).sum(1).detach()
        policy_target = qs - vs.view(-1,1)

        if self.double_q:
            vs2 = (ps * qs2).sum(1).detach()
            policy_target2 = qs - vs2.view(-1,1)
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
        entropy = prob_dict['ally_entropy']
        p = prob_dict['probs']
        exp_q = (q * p).sum(dim=1)
        return exp_q, entropy
