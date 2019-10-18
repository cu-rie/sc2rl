import torch
import dgl
from sc2rl.rl.brains.BrainBase import BrainBase

# for hinting
from sc2rl.rl.modules.ActorCritic import ActorCriticModule
from sc2rl.rl.networks.rnn_encoder import RNNEncoder
from sc2rl.rl.networks.RelationalNetwork import RelationalNetwork


class MultiStepSharedActorCriticBrain(BrainBase):
    def __init__(self,
                 actor_critic,
                 hist_encoder,
                 curr_encoder,
                 hyper_params: dict):
        """
        :param actor_critic:
        :param hist_encoder:
        :param curr_encoder:
        :param hyper_params: (dictionary)

        expected keys :
        'actor_update_freq' (int) update actor every 'actor_update_freq'
        'optimizer' (str)
        'actor_lr' (float) lr for actor related parameters
        'critic_lr' (float) lr for critic related parameters
        'auto_entropy' (bool) flag for auto entropy tuning
            'target_entropy' (float)
            'entropy_lr' (float)
        """

        super(MultiStepSharedActorCriticBrain, self).__init__()
        self.actor_critic = actor_critic  # type: ActorCriticModule
        self.hist_encoder = hist_encoder  # type: RNNEncoder
        self.curr_encoder = curr_encoder  # type: RelationalNetwork

        self.hyper_params = hyper_params

        self._actor_related_params = list(self.actor_critic.actor.parameters()) + \
                                     list(self.hist_encoder.parameters()) + \
                                     list(self.curr_encoder.parameters())

        self._critic_related_params = list(self.actor_critic.critic.parameters()) + \
                                      list(self.hist_encoder.parameters()) + \
                                      list(self.curr_encoder.parameters())

        optimizer = self.get_optimizer()
        self.actor_optimizer = optimizer(self._actor_related_params, lr=hyper_params['actor_lr'])
        self.critic_optimizer = optimizer(self._critic_related_params, lr=hyper_params['critic_lr'])

        if hyper_params['auto_entropy']:
            self.target_entropy = hyper_params['target_entropy']
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer([self.log_alpha], lr=hyper_params['entropy_lr'])

        self.update_steps = 0

    def get_optimizer(self):
        target_opt = self.hyper_params['optimizer']
        if target_opt in ['Adam', 'adam']:
            opt = torch.optim.Adam
        else:
            raise RuntimeError("Not supported optimizer type: {}".format(target_opt))
        return opt

    def forward(self, hist, state):
        pass

    def get_action(self, num_time_steps, hist_graph, hist_node_feature,
                   curr_graph, curr_node_feature, maximum_num_enemy):
        assert isinstance(curr_graph, dgl.DGLGraph), "get action is designed to work on a single graph!"

        encoded_node_feature = self.link_hist_to_curr(num_time_steps,
                                                      hist_graph, hist_node_feature, self.hist_encoder,
                                                      curr_graph, curr_node_feature, self.curr_encoder)

        return self.actor_critic.get_action(curr_graph, encoded_node_feature, maximum_num_enemy)

    def fit(self,
            c_num_time_steps, c_h_graph, c_h_node_feature, c_graph, c_node_feature, c_maximum_num_enemy,
            n_num_time_steps, n_h_graph, n_h_node_feature, n_graph, n_node_feature, n_maximum_num_enemy,
            actions, rewards, dones,
            target_critic=None, actor_clip_norm=None, critic_clip_norm=None):

        # the prefix 'c' indicates #current# time stamp inputs
        # the prefix 'n' indicates #next# time stamp inputs

        self.update_steps += 1

        # update critic
        c_encoded_node_feature = self.link_hist_to_curr(c_num_time_steps,
                                                        c_h_graph, c_h_node_feature, self.hist_encoder,
                                                        c_graph, c_node_feature, self.curr_encoder)

        n_encoded_node_feature = self.link_hist_to_curr(n_num_time_steps,
                                                        n_h_graph, n_h_node_feature, self.hist_encoder,
                                                        n_graph, n_node_feature, self.curr_encoder)

        critic_loss = self.actor_critic.compute_critic_loss(c_graph, c_encoded_node_feature, c_maximum_num_enemy,
                                                            actions,
                                                            n_graph, n_encoded_node_feature, n_maximum_num_enemy,
                                                            rewards, dones, target_critic)

        self.clip_and_optimize(self.critic_optimizer, self._critic_related_params, critic_loss, critic_clip_norm)

        # update actor
        if self.update_steps % self.hyper_params['actor_update_freq'] == 0:
            c_encoded_node_feature = self.link_hist_to_curr(c_num_time_steps,
                                                            c_h_graph, c_h_node_feature, self.hist_encoder,
                                                            c_graph, c_node_feature, self.curr_encoder)

            actor_loss = self.actor_critic.compute_actor_loss(c_graph, c_encoded_node_feature, c_maximum_num_enemy)

            self.clip_and_optimize(self.actor_optimizer, self._actor_related_params, actor_loss, actor_clip_norm)

        return actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()

    @staticmethod
    def clip_and_optimize(optimizer, parameters, loss, clip_val=None):
        optimizer.zero_grad()
        loss.backward()
        if clip_val is not None:
            torch.nn.utils.clip_grad_norm_(parameters, clip_val)
        optimizer.step()

    @staticmethod
    def link_hist_to_curr(num_time_steps,
                          h_graph, h_node_feature, h_encoder,
                          c_graph, c_node_feature, c_encoder):
        h_enc_out, h_enc_hidden = h_encoder(num_time_steps, h_graph, h_node_feature)

        # recent_hist_enc : slice of the last RNN layer's hidden
        recent_h_enc = h_enc_out[:, -1, :]  # [Batch size x rnn hidden]
        c_enc_out = c_encoder(c_graph, c_node_feature)

        if isinstance(c_graph, dgl.BatchedDGLGraph):
            c_units = c_graph.batch_num_nodes
            c_units = torch.Tensor(c_units).long()
            recent_h_enc = recent_h_enc.repeat_interleave(c_units, dim=0)
        else:
            c_unit = c_graph.number_of_nodes()
            recent_h_enc = recent_h_enc.repeat_interleave(c_unit, dim=0)
        c_encoded_node_feature = torch.cat([recent_h_enc, c_enc_out], dim=1)
        return c_encoded_node_feature


def get_hyper_param_dict(**kwargs):
    hyper_params = dict()
    hyper_params['actor_update_freq'] = 1
    hyper_params['optimizer'] = 'adam'
    hyper_params['actor_lr'] = 1e-4
    hyper_params['critic_lr'] = 1e-3
    hyper_params['auto_entropy'] = True
    hyper_params['target_entropy'] = 0.1
    hyper_params['entropy_lr'] = 1e-4

    return hyper_params
