import torch


def dn(tensor: torch.Tensor):
    return tensor.detach().numpy()
