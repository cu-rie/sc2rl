import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def max_singular_value(w_mat, u, power_iterations):
    for _ in range(power_iterations):
        v = l2normalize(torch.mm(u, w_mat.data))

        u = l2normalize(torch.mm(v, torch.t(w_mat.data)))

    sigma = torch.sum(torch.mm(u, w_mat) * v)

    return u, sigma, v


class SNLinear(torch.nn.Linear):

    def __init__(self, *args, spectral_norm_pi=1, **kwargs):
        super(SNLinear, self).__init__(*args, **kwargs)
        self.spectral_norm_pi = spectral_norm_pi
        if spectral_norm_pi > 0:
            self.register_buffer("u", torch.randn((1, self.out_features), requires_grad=False))
        else:
            self.register_buffer("u", None)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias.data, 0)

    def forward(self, input):
        if self.spectral_norm_pi > 0:
            w_mat = self.weight.view(self.out_features, -1)
            u, sigma, _ = max_singular_value(w_mat, self.u, self.spectral_norm_pi)

            # w_bar = torch.div(w_mat, sigma)
            w_bar = torch.div(self.weight, sigma)
            if self.training:
                self.u = u
            # self.w_bar = w_bar.detach()
            # self.sigma = sigma.detach()
        else:
            w_bar = self.weight
        return F.linear(input, w_bar, self.bias)
