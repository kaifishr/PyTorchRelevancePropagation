"""Script tests different implementations of LRP for the linear layer for their equivalence.
"""
import torch
import time


def lrp_v1(
    layer: torch.nn.Linear, a: torch.tensor, r: torch.tensor, eps: float = 1e-5
) -> torch.tensor:
    z = layer.forward(a) + eps
    s = (r / z).data
    (z * s).sum().backward()
    c = a.grad
    r = (a * c).data
    return r


def lrp_v2(
    layer: torch.nn.Linear, a: torch.tensor, r: torch.tensor, eps: float = 1e-5
) -> torch.tensor:
    w = layer.weight
    b = layer.bias
    z = torch.mm(a, w.T) + b + eps
    s = r / z
    c = torch.mm(s, w)
    r = (a * c).data
    return r


def lrp_v3(
    layer: torch.nn.Linear, a: torch.tensor, r: torch.tensor, eps: float = 1e-5
) -> torch.tensor:
    z = layer.forward(a) + eps
    s = r / z
    c = torch.mm(s, layer.weight)
    r = (a * c).data
    return r


def main():
    torch.manual_seed(69)

    batch_size = 16
    in_features = 512
    out_features = 256

    # Linear layer, keep only positive weights
    linear = torch.nn.Linear(in_features=in_features, out_features=out_features)
    linear.weight = torch.nn.Parameter(linear.weight.clamp(min=0.0))
    linear.bias = torch.nn.Parameter(torch.zeros_like(linear.bias))

    # Random activations
    x = torch.randn(size=(batch_size, in_features))
    a = torch.relu(x)
    a.requires_grad_(True)

    # Random relevance, use softmax to ensure that relevance sums up to one
    r = torch.softmax(torch.rand(size=(batch_size, out_features)), dim=-1)
    print(r.sum(dim=-1))

    t0 = time.time()
    for _ in range(100):
        r1 = lrp_v1(linear, a, r)
        a.grad = torch.zeros_like(a)
    print(time.time() - t0)
    print(r1.sum(dim=-1))

    t0 = time.time()
    for _ in range(100):
        r2 = lrp_v2(linear, a, r)
    print(time.time() - t0)
    print(r2.sum(dim=-1))

    t0 = time.time()
    for _ in range(100):
        r3 = lrp_v3(linear, a, r)
    print(time.time() - t0)
    print(r3.sum(dim=-1))

    assert torch.allclose(r1, r2)
    assert torch.allclose(r1, r3)


if __name__ == "__main__":
    main()
