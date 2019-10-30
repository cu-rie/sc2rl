import torch
from sc2rl.optim.Radam import RAdam
from sc2rl.rl.brains.BrainBase import BrainBase

from sc2rl.config.ConfigBase_refac import ConfigBase


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
            'eps_min': 0.01
        }

        self.fit_conf = {
            'prefix': 'fit_conf',
            'norm_clip_val': 0.5,
            'tau': 0.1
        }


class QMixBrain(BrainBase):
    def __init__(self, conf, qnet, mixer, qnet_target=None, mixer_target=None):
        super(QMixBrain, self).__init__()
        self.qnet = qnet
        self.mixer = mixer

        self.qnet_target = qnet_target
        self.mixer_target = mixer_target

        if self.qnet_target is None:
            self.use_target = False
            self.update_target_network(1.0, self.qnet, self.qnet_target)
            self.update_target_network(1.0, self.mixer, self.mixer_target)
        else:
            self.use_target = True

        self.brain_conf = conf.brain_conf
        self.gamma = self.brain_conf['gamma']
        # self.eps = self.brain_conf['eps']
        self.register_buffer('eps', torch.ones(1, ) * self.brain_conf['eps'])
        self.register_buffer('eps_min', torch.ones(1, ) * self.brain_conf['eps_min'])
        self.eps_gamma = self.brain_conf['eps_gamma']

        optimizer = self.get_optimizer(self.brain_conf['optimizer'])

        if self.brain_conf['optimizer'] == 'lookahead':
            qnet_base_optimzier = RAdam(self.qnet.parameters(), lr=self.brain_conf['lr'])
            self.qnet_optimizer = optimizer(qnet_base_optimzier)
        else:
            self.qnet_optimizer = optimizer(self.qnet.parameters(), lr=self.brain_conf['lr'])

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

        # compute q-target:
        with torch.no_grad():
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
                               parameters=self.qnet.parameters(),
                               loss=loss,
                               clip_val=self.fit_conf['norm_clip_val'])
        self.update_target_network(self.fit_conf['tau'], self.qnet, self.qnet_target)

        # decay epsilon
        self.eps = self.eps * self.eps_gamma
        if self.eps <= self.eps_min:
            self.eps.fill_(self.eps_min.data)
        self.qnet.eps = self.eps

        fit_dict = dict()
        fit_dict['loss'] = loss.detach().cpu().numpy()

        return fit_dict
