import warnings
import torch
from torch.optim.lr_scheduler import StepLR
from sc2rl.optim.Radam import RAdam
from sc2rl.rl.brains.BrainBase import BrainBase

from sc2rl.config.ConfigBase import ConfigBase


class QmixBrainConfig(ConfigBase):
    def __init__(self, brain_conf=None, fit_conf=None):
        super(QmixBrainConfig, self).__init__(brain_conf=brain_conf,
                                              fit_conf=fit_conf)
        self.brain_conf = {
            'prefix': 'brain_conf',
            'optimizer': 'lookahead',
            'lr': 1e-4,
            'gamma': 1.0,
            'eps': 1.0,
            'eps_gamma': 0.995,
            'eps_min': 0.01,
            'use_double_q': True,
            'scheduler_step_size': 30,
            'scheduler_gamma': 0.5
        }

        self.fit_conf = {
            'prefix': 'fit_conf',
            'norm_clip_val': None,
            'tau': 0.1
        }


class QMixBrain(BrainBase):
    def __init__(self,
                 conf,
                 qnet,
                 mixer,
                 qnet_target=None,
                 mixer_target=None,
                 qnet2=None,
                 mixer2=None):
        super(QMixBrain, self).__init__()
        self.qnet = qnet
        self.mixer = mixer

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
        self.scheduler_step_size = self.brain_conf['scheduler_step_size']
        self.scheduler_gamma = self.brain_conf['scheduler_gamma']

        if int(self.use_double_q) + int(self.use_clipped_q) >= 2:
            warnings.warn("Either one of 'use_double_q' or 'clipped_q' can be true. 'use_double_q' set to be false.")
            self.use_double_q = False

        if self.use_double_q:
            assert self.use_target, "if 'use_double_q' true, then 'use_target' should be true."

        optimizer = self.get_optimizer(self.brain_conf['optimizer'])

        params = list(self.qnet.parameters()) + list(self.mixer.parameters())
        if self.brain_conf['optimizer'] == 'lookahead':
            qnet_base_optimizer = RAdam(params, lr=self.brain_conf['lr'])
            self.qnet_optimizer = optimizer(qnet_base_optimizer)
            self.qnet_scheduler = StepLR(qnet_base_optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma)
        else:
            self.qnet_optimizer = optimizer(params, lr=self.brain_conf['lr'])
            self.qnet_scheduler = StepLR(self.qnet_optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma)

        if self.use_clipped_q:
            params = list(self.qnet2.parameters()) + list(self.mixer2.parameters())
            if self.brain_conf['optimizer'] == 'lookahead':
                qnet2_base_optimizer = RAdam(params, lr=self.brain_conf['lr'])
                self.qnet2_optimizer = optimizer(qnet2_base_optimizer)
                self.qnet2_scheduler = StepLR(qnet2_base_optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma)
            else:
                self.qnet2_optimizer = optimizer(params, lr=self.brain_conf['lr'])
                self.qnet2_scheduler = StepLR(self.qnet2_optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma)

        self.fit_conf = conf.fit_conf

    def get_action(self,
                   num_time_steps,
                   hist_graph,
                   hist_feature,
                   curr_graph,
                   curr_feature,
                   maximum_num_enemy):
        nn_actions, info_dict = self.qnet.get_action(num_time_steps,
                                                     hist_graph, hist_feature,
                                                     curr_graph, curr_feature, maximum_num_enemy,
                                                     self.eps)
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
                               parameters=list(self.qnet.parameters())+list(self.mixer.parameters()),
                               loss=loss,
                               clip_val=self.fit_conf['norm_clip_val'],
                               scheduler=self.qnet_scheduler)

        self.update_target_network(self.fit_conf['tau'], self.qnet, self.qnet_target)
        self.update_target_network(self.fit_conf['tau'], self.mixer, self.mixer_target)

        # decay epsilon
        self.eps = self.eps * self.eps_gamma
        if self.eps <= self.eps_min:
            self.eps.fill_(self.eps_min.data[0])
        self.qnet.eps = self.eps

        fit_dict = dict()
        fit_dict['loss'] = loss.detach().cpu().numpy()

        if self.use_clipped_q:
            loss2 = torch.nn.functional.mse_loss(input=q_tot_2, target=q_targets)
            self.clip_and_optimize(optimizer=self.qnet2_optimizer,
                                   parameters=list(self.qnet2.parameters()) + list(self.mixer2.parameters()),
                                   loss=loss2,
                                   clip_val=self.fit_conf['norm_clip_val'],
                                   scheduler=self.qnet2_scheduler)
            fit_dict['loss2'] = loss2.detach().cpu().numpy()

        return fit_dict
