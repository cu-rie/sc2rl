import torch
from sc2rl.optim.Radam import RAdam
from sc2rl.rl.brains.BrainBase import BrainBase


class QMixBrain(BrainBase):
    def __init__(self, conf, qnet, mixer, qnet_target=None, mixer_target=None):
        self.qnet = qnet
        self.mixer = mixer

        self.qnet_target = qnet_target
        self.mixer_target = mixer_target

        if self.qnet_target is None:
            self.use_target = False
        else:
            self.qnet_target = True

        self.brain_conf = conf.brain_conf
        self.gamma = self.brain_conf['gamma']

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


        # [ # allies x # c_maximum_num_enemy]
        qs = self.qnet.compute_qs(num_time_steps,
                                  c_hist_graph, c_hist_feature,
                                  c_curr_graph, c_curr_feature, c_maximum_num_enemy)

        qs = qs.gather(-1, actions.unsqueeze(-1).long()).squeeze(dim=-1)
        q_tot = self.mixer(c_curr_graph, c_hist_feature, qs)

        # compute q-target:
        with torch.no_grad():
            if self.use_target:
                q_net = self.qnet_target
                mixer = self.mixer_target
            else:
                q_net = self.qnet
                mixer = self.mixer

            next_qs = q_net.compute_qs(num_time_steps,
                                         n_hist_graph, n_hist_feature,
                                         n_curr_graph, n_curr_feature, n_maximum_num_enemy)

            next_qs = next_qs.argmax(dim=1)
            next_q_tot = mixer(n_curr_graph, n_curr_feature, next_qs)

        q_targets = rewards + self.gamma * next_q_tot * (1-dones)

        loss = torch.nn.functional.mse_loss(input=q_tot, target=q_targets)
        self.clip_and_optimize(self.qnet_optimizer, loss, clip_val=self.fit_conf['norm_clip_val'])

        fit_dict = dict()
        fit_dict['actor_loss'] = loss.deatch().cpu().numpy()
        return fit_dict

