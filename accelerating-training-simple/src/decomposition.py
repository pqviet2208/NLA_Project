import torch.nn as nn
import torch as th
from tensorly.decomposition import partial_tucker
import numpy as np

def tucker_decomposition_conv_layer(layer, rank):
    if rank is None:
        return layer
    
    if rank == -1:
        if min(layer.weight.data.shape[:2]) in [1024, 2048]:
            rank = 64
        else:
            return layer

    if rank >= max(layer.weight.data.shape[:2]):
        return layer

    (core, [last, first]), _ = \
        partial_tucker(layer.weight.data, \
            modes=[0, 1], rank=rank, init='svd')
    
    first_layer = nn.Conv2d(
        first.shape[0], first.shape[1], 
        *(1, 1, 0),
        bias=False
    )
    core_layer = nn.Conv2d(
        core.shape[1], core.shape[0], 
        layer.kernel_size, layer.stride, layer.padding,
        bias=False
    )
    last_layer = nn.Conv2d(
        last.shape[1], last.shape[0], 
        *(1, 1, 0),
        bias=layer.bias is not None
    )
    if layer.bias is not None:
        last_layer.bias.data = layer.bias.data

    first_layer.weight.data = \
        th.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)

def svd_decomposition_linear_layer(layer, rank):
    if rank is None:
        return layer
    
    if rank == -1:
        if min(layer.weight.data.shape[:2]) in [1024, 2048]:
            rank = 64
        else:
            return layer

    if rank >= min(layer.weight.data.shape):
        return layer

    U, S, Vh = np.linalg.svd(layer.weight.data.cpu().numpy(), full_matrices=False)
    U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]

    first_layer = nn.Linear(in_features=Vh.shape[1], out_features=Vh.shape[0], bias=False)
    second_layer = nn.Linear(in_features=U.shape[1], out_features=U.shape[0], bias=layer.bias is not None)

    if layer.bias is not None:
        second_layer.bias.data = layer.bias.data

    first_layer.weight.data = th.tensor((Vh * S[..., None]), device=layer.weight.data.device, requires_grad=layer.weight.data.requires_grad)
    second_layer.weight.data = th.tensor(U, device=layer.weight.data.device, requires_grad=layer.weight.data.requires_grad)

    new_layers = [first_layer, second_layer]
    return nn.Sequential(*new_layers)

def tucker_decompose_model(model, linear_max_rank, conv_max_rank):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            model._modules[name] = tucker_decompose_model(module, linear_max_rank=linear_max_rank, conv_max_rank=conv_max_rank)
        elif isinstance(module, nn.Conv2d):
            decomposed = tucker_decomposition_conv_layer(module, rank=conv_max_rank)
            model._modules[name] = decomposed
        elif isinstance(module, nn.Linear):
            decomposed = svd_decomposition_linear_layer(module, rank=linear_max_rank)
            model._modules[name] = decomposed
    return model