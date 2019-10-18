import torch


class BrainBase(torch.nn.Module):
    def __init__(self, move_dim=4):
        super(BrainBase, self).__init__()
        self.move_dim = move_dim

    def get_action(self, *args, **kwargs):

        nn_args, sc2_args = self._get_action(*args, **kwargs)

        return nn_args

    def fit(self, *args, **kwargs):
        pass

    @staticmethod
    def update_target_network(tau, source, target):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def clip_and_optimize(optimizer, parameters, loss, clip_val=None):
        optimizer.zero_grad()
        loss.backward()
        if clip_val is not None:
            torch.nn.utils.clip_grad_norm_(parameters, clip_val)
        optimizer.step()

    def get_optimizer(self, target_opt):
        if target_opt in ['Adam', 'adam']:
            opt = torch.optim.Adam
        else:
            raise RuntimeError("Not supported optimizer type: {}".format(target_opt))
        return opt