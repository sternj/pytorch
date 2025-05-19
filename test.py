import torch

@torch.compile
def f(x):
    return x[1].view(-1)

x = torch.randn(4, 4, requires_grad=True)
out = f(x)

