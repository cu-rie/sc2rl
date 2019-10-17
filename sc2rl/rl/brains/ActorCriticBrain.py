import torch
import dgl
from sc2rl.rl.brains.BrainBase import BrainBase


class MultiStepActorCriticBrain(BrainBase):
    def __init__(self,
                 actor_critic,
                 hist_encoder,
                 curr_encoder,
                 opt: str = 'Adam',
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-3,
                 ):
        super(MultiStepActorCriticBrain, self).__init__()
        self.actor_critic = actor_critic
        self.hist_encoder = hist_encoder
        self.curr_encoder = curr_encoder

        self._actor_related_params = list(self.actor_critic.actor.parameters()) + \
                                     list(self.hist_encoder.parameters()) + \
                                     list(self.curr_encoder.parameters())

        self._critic_related_params = list(self.actor_critic.critic.parameters()) + \
                                      list(self.hist_encoder.parameters()) + \
                                      list(self.curr_encoder.parameters())
        if opt in ['Adam', 'adam']:
            self.actor_optimizer = torch.optim.Adam(self._actor_related_params, lr=actor_lr)
            self.critic_optimizer = torch.optim.Adam(self._critic_related_params, lr=critic_lr)
        else:
            raise RuntimeError("Not supported optimizer type")

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
            rewards, dones, target_critic,
            actor_clip_norm=None, critic_clip_norm=None,
            ):
        # the prefix 'c' indicates #current# time stamp inputs
        # the prefix 'n' indicates #next# time stamp inputs

        c_encoded_node_feature = self.link_hist_to_curr(c_num_time_steps,
                                                        c_h_graph, c_h_node_feature, self.hist_encoder,
                                                        c_graph, c_node_feature, self.curr_encoder)

        n_encoded_node_feature = self.link_hist_to_curr(n_num_time_steps,
                                                        n_h_graph, n_h_node_feature, self.hist_encoder,
                                                        n_graph, n_node_feature, self.curr_encoder)

        actor_loss, critic_loss = self.actor_critic.compute_loss(c_graph, c_encoded_node_feature, c_maximum_num_enemy,
                                                                 n_graph, n_encoded_node_feature, n_maximum_num_enemy,
                                                                 rewards, dones, target_critic)

        self.clip_and_optimize(self.actor_optimizer, self._actor_related_params, actor_loss, actor_clip_norm)
        self.clip_and_optimize(self.critic_optimizer, self._critic_related_params, critic_loss, critic_clip_norm)

        return actor_loss.detatch().cpu().numpy(), critic_loss.detatch().cpu().numpy()

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
        recent_h_enc = h_enc_out[-1, :, :]  # [Batch size x rnn hidden]
        c_enc_out = c_encoder(c_graph, c_node_feature)

        c_units = c_graph.number_of_nodes()
        recent_h_enc = recent_h_enc.repeat_interleave(c_units, dim=0)
        c_encoded_node_feature = torch.cat([recent_h_enc, c_enc_out], dim=1)
        return c_encoded_node_feature
