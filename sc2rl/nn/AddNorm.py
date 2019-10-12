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


class AddNormLayerHetero(torch.nn.Module):

    def __init__(self, model_dim, use_norm=True):
        super(AddNormLayerHetero, self).__init__()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = torch.nn.LayerNorm(model_dim)

    def forward(self, x_dict, x_updated_dict):
        return_dict = dict()

        for key in x_dict.keys():
            value1 = x_dict[key]
            value2 = x_updated_dict[key]

            updated_value = value1 + value2
            if self.use_norm:
                updated_value = self.norm(updated_value)
            return_dict[key] = updated_value

        return return_dict
