import torch


class AgentBase(torch.nn.module):
    """
    An interface for RL agent
    """

    def __init__(self, brain, buffer):
        super(AgentBase, self).__init__()
        self.brain = brain
        self.buffer = buffer

    def forward(self, *args, **kwargs):
        raise NotImplementedError("This method will be implemented in the child class")

    def get_action(self, *args, **kwargs):
        """ Expected behaviour
        1. pre-process input depending on the selection of algorithm
        2. call get_action method of the RL brain
        3. post-process output to be compatible with environment
        """
        return self.brain.get_action(*args, **kwargs)

    def fit(self, *args, **kwargs):
        """ Expected behaviour
        1. get samples from the buffer
        2. pre-process the samples depending on algorithm specification
        3. call fit method of the rl brain
        4. post-process the outputs of RL brains 'fit'
        """
        raise NotImplementedError("This method will be implemented in the child class")

    def append_sample(self, sample):
        self.buffer.push(sample)
