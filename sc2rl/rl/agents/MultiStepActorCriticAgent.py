import torch
from sc2rl.rl.agents.AgentBase import AgentBase


class MultiStepMAACAgent(AgentBase):
    def __init__(self, brain, buffer):
        super(MultiStepMAACAgent, self).__init__(brain, buffer)

    def forward(self, *args, **kwargs):
        return self.get_action(*args, **kwargs)

    def get_action(self, *args, **kwargs):
        return self.brain.get_action(*args, **kwargs)






