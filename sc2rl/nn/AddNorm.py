import torch


class AddNormLayer(torch.nn.Module):

    def __init__(self, model_dim, use_norm=True):
        super(AddNormLayer, self).__init__()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = torch.nn.LayerNorm(model_dim)

    def forward(self, x, x_updated):
        ret = x + x_updated
        if self.use_norm:
            ret = self.norm(ret)
        return ret
