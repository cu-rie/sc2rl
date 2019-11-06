import warnings
import torch
from sc2rl.optim.Radam import RAdam
from sc2rl.rl.brains.BrainBase import BrainBase
from sc2rl.config.ConfigBase import ConfigBase
from sc2rl.config.nn_configs import VERY_SMALL_NUMBER


class QmixActorCriticBrainConfig(ConfigBase):
    def __init__(self, brain_conf=None, fit_conf=None, entropy_conf=None):
        super(QmixActorCriticBrainConfig, self).__init__(brain_conf=brain_conf,
                                                         fit_conf=fit_conf,
                                                         entropy_conf=entropy_conf)
        self.brain_conf = {
            'prefix': 'brain_conf',
            'optimizer': 'lookahead',
            'critic_lr': 1e-4,
            'actor_lr': 1e-5,
            'gamma': 1.0,
            'eps': 1.0,
            'eps_gamma': 0.995,
            'eps_min': 0.01,
            'use_double_q': True
        }

        self.fit_conf = {
            'prefix': 'fit_conf',
            'norm_clip_val': None,
            'num_critic_pre_fit_steps': 5,
            'tau': 0.1
        }

        self.entropy_conf = {
            'prefix': 'brain_entropy',
            'optimizer': 'radam',
            'auto_tune': True,
            'alpha': 0.01,
            'target_alpha': -(4 + 1 + 1),  # Expected minimal action dim : Move 4 + Hold 1 + Attack 1
            'lr': 1e-4
        }


class QMixActorCriticBrain(BrainBase):
    def __init__(self,
                 conf,
                 qnet,
                 mixer,
                 actor,
                 qnet_target=None,
                 mixer_target=None,
                 qnet2=None,
                 mixer2=None):
        super(QMixActorCriticBrain, self).__init__()
        self.qnet = qnet
        self.mixer = mixer
        self.actor = actor

        self.qnet_target = qnet_target
        self.mixer_target = mixer_target

        self.qnet2 = qnet2
        self.mixer2 = mixer2

        if self.qnet_target is None:
            self.use_target = False
        else:
            self.use_target = True
            self.update_target_network(1.0, self.qnet, self.qnet_target)
            self.update_target_network(1.0, self.mixer, self.mixer_target)

        if self.qnet2 is None:
            self.use_clipped_q = False
        else:
            self.use_clipped_q = True

        self.brain_conf = conf.brain_conf
        self.gamma = self.brain_conf['gamma']
        self.register_buffer('eps', torch.ones(1, ) * self.brain_conf['eps'])
        self.register_buffer('eps_min', torch.ones(1, ) * self.brain_conf['eps_min'])
        self.eps_gamma = self.brain_conf['eps_gamma']
        self.use_double_q = self.brain_conf['use_double_q']

        if int(self.use_double_q) + int(self.use_clipped_q) >= 2:
            warnings.warn("Either one of 'use_double_q' or 'clipped_q' can be true. 'use_double_q' set to be false.")
            self.use_double_q = False

        if self.use_double_q:
            assert self.use_target, "if 'use_double_q' true, then 'use_target' should be true."

        optimizer = self.get_optimizer(self.brain_conf['optimizer'])

        critic_params = list(self.qnet.parameters()) + list(self.mixer.parameters())
        actor_params = self.actor.parameters()
        if self.brain_conf['optimizer'] == 'lookahead':
            qnet_base_optimizer = RAdam(critic_params, lr=self.brain_conf['critic_lr'])
            actor_base_optimizer = RAdam(actor_params, lr=self.brain_conf['actor_lr'])
            self.qnet_optimizer = optimizer(qnet_base_optimizer)
            self.actor_optimizer = optimizer(actor_base_optimizer)
        else:
            self.qnet_optimizer = optimizer(critic_params, lr=self.brain_conf['critic_lr'])
            self.actor_optimizer = optimizer(actor_params, lr=self.brain_conf['actor_lr'])

        if self.use_clipped_q:
            params = list(self.qnet2.parameters()) + list(self.mixer2.parameters())
            if self.brain_conf['optimizer'] == 'lookahead':
                qnet_base_optimizer = RAdam(params, lr=self.brain_conf['critic_lr'])
                self.qnet2_optimizer = optimizer(qnet_base_optimizer)
            else:
                self.qnet2_optimizer = optimizer(params, lr=self.brain_conf['critic_lr'])

        self.fit_conf = conf.fit_conf
        self.entropy_conf = conf.entropy_conf

        if self.entropy_conf['auto_tune']:
            self.target_alpha = self.entropy_conf['target_alpha']
            self.log_alpha = torch.nn.Parameter(torch.zeros(1))
            optimizer = self.get_optimizer(self.entropy_conf['optimizer'])
            if self.entropy_conf['optimizer'] == 'lookahead':
                self.alpha_optimizer = optimizer(RAdam([self.log_alpha], lr=self.entropy_conf['lr']))
            else:
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
            fit_dict.update(alpha_fit_dict)
            actor_fit_dict = self.fit_actor(num_time_steps,
                                            c_hist_graph,
                                            c_hist_feature,
                                            c_curr_graph,
                                            c_curr_feature,
                                            c_maximum_num_enemy
                                            )
            fit_dict.update(actor_fit_dict)

        critic_fit_dict = self.fit_critic(num_time_steps,
                                          c_hist_graph, c_hist_feature,
                                          c_curr_graph, c_curr_feature,
                                          c_maximum_num_enemy,
                                          actions,
                                          n_hist_graph, n_hist_feature,
                                          n_curr_graph, n_curr_feature,
                                          n_maximum_num_enemy,
                                          rewards,
                                          dones)
        fit_dict.update(critic_fit_dict)
        self.fit_steps += 1

        return fit_dict

    def fit_actor(self,
                  num_time_steps,
                  hist_graph,
                  hist_feature,
                  curr_graph,
                  curr_feature,
                  maximum_num_enemy
                  ):
        with torch.no_grad():
            qs_dict = self.qnet.compute_qs(num_time_steps,
                                           hist_graph, hist_feature,
                                           curr_graph, curr_feature, maximum_num_enemy)
            qs = qs_dict['qs']
            if self.use_double_q:
                qs2_dict = self.qnet2.compute_qs(num_time_steps,
                                                 hist_graph, hist_feature,
                                                 curr_graph, curr_feature, maximum_num_enemy)
                qs2 = qs2_dict['qs']

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

        if self.use_double_q:
            vs2 = (ps * qs2).sum(1).detach()
            policy_target2 = qs - vs2.view(-1, 1)
            policy_target = torch.min(policy_target, policy_target2)

        device = log_ps.device

        unmasked_loss = log_ps * (self.log_alpha.exp() * log_ps - policy_target)
        loss_mask = (log_ps > torch.log(torch.tensor(VERY_SMALL_NUMBER, device=device))).float()
        loss = (unmasked_loss * loss_mask).sum() / loss_mask.sum()
        self.clip_and_optimize(optimizer=self.actor_optimizer, parameters=self.actor.parameters(), loss=loss,
                               clip_val=self.fit_conf['norm_clip_val'])

        fit_dict = dict()
        fit_dict['actor_loss'] = loss.detach().cpu().numpy()
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

        # [ # allies x # c_maximum_num_enemy]
        q_dict = self.qnet.compute_qs(num_time_steps,
                                      c_hist_graph, c_hist_feature,
                                      c_curr_graph, c_curr_feature, c_maximum_num_enemy)

        qs = q_dict['qs']

        qs = qs.gather(-1, actions.unsqueeze(-1).long()).squeeze(dim=-1)
        q_tot = self.mixer(c_curr_graph, c_curr_feature, qs)

        if self.use_clipped_q:
            q2_dict = self.qnet2.compute_qs(num_time_steps,
                                            c_hist_graph, c_hist_feature,
                                            c_curr_graph, c_curr_feature, c_maximum_num_enemy)

            q2s = q2_dict['qs']

            q2s = q2s.gather(-1, actions.unsqueeze(-1).long()).squeeze(dim=-1)
            q_tot_2 = self.mixer2(c_curr_graph, c_curr_feature, q2s)

        # compute q-target:
        with torch.no_grad():
            if self.use_double_q:
                next_q_targets_dict = self.qnet_target.compute_qs(num_time_steps,
                                                                  n_hist_graph, n_hist_feature,
                                                                  n_curr_graph, n_curr_feature, n_maximum_num_enemy)

                next_q_targets = next_q_targets_dict['qs']
                next_as = next_q_targets.argmax(dim=1)

                next_q_dict = self.qnet.compute_qs(num_time_steps,
                                                   n_hist_graph, n_hist_feature,
                                                   n_curr_graph, n_curr_feature, n_maximum_num_enemy)

                next_qs = next_q_dict['qs']
                next_qs = next_qs.gather(-1, next_as.unsqueeze(-1).long()).squeeze(dim=-1)
                next_q_tot = self.mixer_target(n_curr_graph, n_curr_feature, next_qs)

            elif self.use_clipped_q:
                next_q1_dict = self.qnet.compute_qs(num_time_steps,
                                                    n_hist_graph, n_hist_feature,
                                                    n_curr_graph, n_curr_feature, n_maximum_num_enemy)

                next_q1s = next_q1_dict['qs']

                next_q1s, _ = next_q1s.max(dim=1)
                next_q_tot_1 = self.mixer(n_curr_graph, n_curr_feature, next_q1s)

                next_q2_dict = self.qnet2.compute_qs(num_time_steps,
                                                     n_hist_graph, n_hist_feature,
                                                     n_curr_graph, n_curr_feature, n_maximum_num_enemy)

                next_q2s = next_q2_dict['qs']

                next_q2s, _ = next_q2s.max(dim=1)
                next_q_tot_2 = self.mixer2(n_curr_graph, n_curr_feature, next_q2s)
                next_q_tot = torch.min(next_q_tot_1, next_q_tot_2)

            else:
                if self.use_target:
                    q_net = self.qnet_target
                    mixer = self.mixer_target
                else:
                    q_net = self.qnet
                    mixer = self.mixer

                next_q_dict = q_net.compute_qs(num_time_steps,
                                               n_hist_graph, n_hist_feature,
                                               n_curr_graph, n_curr_feature, n_maximum_num_enemy)

                next_qs = next_q_dict['qs']

                next_qs, _ = next_qs.max(dim=1)
                next_q_tot = mixer(n_curr_graph, n_curr_feature, next_qs)

        q_targets = rewards + self.gamma * next_q_tot * (1 - dones)

        loss = torch.nn.functional.mse_loss(input=q_tot, target=q_targets)

        self.clip_and_optimize(optimizer=self.qnet_optimizer,
                               parameters=list(self.qnet.parameters()) + list(self.mixer.parameters()),
                               loss=loss,
                               clip_val=self.fit_conf['norm_clip_val'])

        self.update_target_network(self.fit_conf['tau'], self.qnet, self.qnet_target)
        self.update_target_network(self.fit_conf['tau'], self.mixer, self.mixer_target)

        fit_dict = dict()
        fit_dict['loss'] = loss.detach().cpu().numpy()

        if self.use_clipped_q:
            loss2 = torch.nn.functional.mse_loss(input=q_tot_2, target=q_targets)
            self.clip_and_optimize(optimizer=self.qnet2_optimizer,
                                   parameters=list(self.qnet2.parameters()) + list(self.mixer2.parameters()),
                                   loss=loss2,
                                   clip_val=self.fit_conf['norm_clip_val'])
            fit_dict['loss2'] = loss.detach().cpu().numpy()

        return fit_dict

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
