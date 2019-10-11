import torch
from sc2rl.rl.brains.BrainBase import BrainBase


class ActorCriticBrain(BrainBase):
    def __init__(self, actor, critic, target_net):
        super(ActorCriticBrain, self).__init__()
        self.actor = actor
        self.critic = critic
        self.target_net = target_net

    def compute_probs(self):
        pass

    def _get_action(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        pass

    def forward_critic(self):
        pass

    def forward_actor(self):
        pass


class MultiStepActorCriticBrain(BrainBase):
    def __init__(self, rnn_encoder, actor, critic, target_net):
        super(MultiStepActorCriticBrain, self).__init__()
        self.actor = actor
        self.critic = critic
        self.target_net = target_net
        self.rnn_encoder = rnn_encoder

    def _get_action(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        pass

    def forward_critic(self, hist, state):
        enc_out = self.rnn_encoder(hist)
        q_vals = self.critic(enc_out, state)
        return q_vals

    def forward_actor(self, hist, state):
        enc_out = self.rnn_encoder(hist)
        probs = self.actor(enc_out, state)
        return probs

    def forward(self, hist, state):
        enc_out = self.rnn_encoder(hist)
        probs = self.actor(enc_out, state)
        q_vals = self.critic(enc_out, state)


