import torch


class BrainBase(torch.nn.Module):
    def __init__(self, move_dim=4):
        super(BrainBase, self).__init__()
        self.move_dim = move_dim

    def get_action(self, *args, **kwargs):

        nn_args, sc2_args = self._get_action(*args, **kwargs)

        return nn_args, sc2_args

    def _get_action(self, *args, **kwargs):
        """
        :param args:
        :param kwargs:
        :return: nn_args, sc2_args
        """
        raise NotImplementedError("This method will be implemented in the child class")

    def get_probs(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        pass

    @staticmethod
    def update_target_network(tau, source, target):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
