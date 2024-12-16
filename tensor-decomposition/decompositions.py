import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
import torch
import torch.nn as nn
import traceback
from collections import OrderedDict
from VBMF import VBMF


# different criterions for sigma selection
# obtained from https://github.com/yuhuixu1993/Trained-Rank-Pruning
class EnergyThreshold(object):

    def __init__(self, threshold, eidenval=True):
        """
        :param threshold: float, threshold to filter small valued sigma:
        :param eidenval: bool, if True, use eidenval as criterion, otherwise use singular
        """
        self.T = threshold
        assert self.T < 1.0 and self.T > 0.0
        self.eiden = eidenval

    def __call__(self, sigmas):
        """
        select proper numbers of singular values
        :param sigmas: numpy array obj which containing singular values
        :return: valid_idx: int, the number of sigmas left after filtering
        """
        if self.eiden:
            energy = sigmas ** 2
        else:
            energy = sigmas

        sum_e = torch.sum(energy)
        valid_idx = sigmas.size(0)
        for i in range(energy.size(0)):
            if energy[:(i + 1)].sum() / sum_e >= self.T:
                valid_idx = i + 1
                break

        return valid_idx


class LinearRate(object):

    def __init__(self, rate):
        """
        filter out small valued singulars according to given proportion
        :param rate: value, left proportion
        """
        self.rate = rate

    def __call__(self, sigmas):
        return int(sigmas.size(0) * self.rate)


class ValueThreshold(object):

    def __init__(self, threshold):
        """
        filter out small valued singulars according to a given value threshold
        :param threshold: float, value threshold
        """
        self.T = threshold

    def __call__(self, sigmas):
        # input sigmas should be a sorted array from large to small
        valid_idx = sigmas.size(0)
        for i in range(len(sigmas)):
            if sigmas[i] < self.T:
                valid_idx = i
                break
        return valid_idx


def decompose_model(model, type, config):
    config["criterion"] = None
    if config["rank"] is not None and config["threshold"] is not None:
        raise Exception("Either threshold or rank can be set. Not both.")
    elif config["rank"] is None:
        if type in ["tucker", "cp"]:
            config["criterion"] = VBMF
        else:
            if config["threshold"] is None:
                config["threshold"] = 0.85
            config["criterion"] = EnergyThreshold(config["threshold"])

    layer_configs = get_per_layer_config(model, config, type)

    if type == 'tucker':
        return tucker_decompose_model(model, layer_configs)
    elif type == 'cp':
        return cp_decompose_model(model, exclude_first_conv, exclude_linears)
    elif type == 'channel':
        return channel_decompose_model(model, layer_configs)
    elif type == 'depthwise':
        return depthwise_decompose_model(model, layer_configs)
    elif type == 'spatial':
        return spatial_decompose_model(model, layer_configs)
    else:
        raise Exception(('Unsupported decomposition type passed: ' + type))


def get_per_layer_config(model, config, decomp_type, passed_first_conv=False):
    layer_configs = { }

    # TODO: handle conflicts in settings
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            layer_configs.update(get_per_layer_config(module, config, decomp_type, passed_first_conv))
        elif type(module) == nn.Conv2d:
            conv_layer = module

            # pop the mask list and check the value of current mask
            enable_current_conv = True
            if config["mask_conv_layers"] is not None:
                enable_current_conv = not mask_conv_layers.pop(0)

            if config["conv_ranks"] is not None:
                if decomp_type != "tucker" or passed_first_conv is False:
                    current_conv_rank = config["conv_ranks"].pop(0)
                else:
                    current_conv_rank = [config["conv_ranks"].pop(0), config["conv_ranks"].pop(0)]
            elif config["rank"] is not None:
                current_conv_rank = config["rank"]
            else:
                current_conv_rank = None

            if not passed_first_conv and config["exclude_first_conv"]:
                layer_configs.update({ conv_layer: (None, None) })
            elif enable_current_conv is False:
                layer_configs.update({ conv_layer: (None, None) })
            elif current_conv_rank is None:
                layer_configs.update({ conv_layer: (None, config["criterion"]) })
            elif current_conv_rank is not None:
                layer_configs.update({ conv_layer: (current_conv_rank, None) })

            if passed_first_conv is False:
                passed_first_conv = True

        elif type(module) == nn.Linear:
            linear_layer = module

            if config["exclude_linears"] is True:
                layer_configs.update({ linear_layer: (None, None) })
            else:
                layer_configs.update({ linear_layer: (None, config["criterion"]) })

    return layer_configs


def tucker_decompose_model(model, layer_configs):
    '''
    decompose filter NxCxHxW to 3 filters:
    R1xCx1x1 , R2xR1xHxW, and NxR2x1x1

    Unlike other decomposition methods, it requires 2 ranks
    '''
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = tucker_decompose_model(module, layer_configs)
        elif type(module) == nn.Conv2d:
            conv_layer = module
            print(conv_layer)

            (set_ranks, criterion) = layer_configs[conv_layer]

            if set_ranks is not None and criterion is not None:
                raise Exception("Can't have both pre-set rank and criterion for a layer")
            elif criterion is not None:
                ranks = tucker_ranks(conv_layer)
            elif set_ranks is not None:
                ranks = set_ranks
            elif set_ranks is None and criterion is None:
                print("\tExcluding layer")
                continue
            print("\tRanks: ", ranks)

            # check if Tucker-1 rank or Tucker-2 ranks
            if np.isscalar(ranks):
                rank = ranks
                is_tucker2 = False
            else:
                is_tucker2 = True

            if (is_tucker2):
                if (np.prod(ranks) >= conv_layer.in_channels * conv_layer.out_channels):
                    print("np.prod(ranks) >= conv_layer.in_channels * conv_layer.out_channels)")
                    continue

                if (any(r <= 0 for r in ranks)):
                    print("One of the estimated ranks is 0 or less. Skipping layer")
                    continue

                decomposed = tucker_decomposition_conv_layer(conv_layer, ranks)
            else:
                if (rank <= 0):
                    print("The estimated rank is 0 or less. Skipping layer")
                    continue

                decomposed = tucker1_decomposition_conv_layer(conv_layer, rank)

            model._modules[name] = decomposed
        elif type(module) == nn.Linear:
            linear_layer = module
            print(linear_layer)

            (set_rank, criterion) = layer_configs[linear_layer]

            if set_rank is not None and criterion is not None:
                raise Exception("Can't have both pre-set rank and criterion for a layer")
            elif criterion is not None:
                rank = tucker1_rank(linear_layer)

                print(linear_layer, "Tucker1 Estimated rank", rank)
                # hack to deal with the case when rank is very small (happened with ResNet56 on CIFAR10) and could deteriorate accuracy
                if rank < 2:
                    rank = svd_rank_linear(linear_layer)
                    print("Will instead use SVD Rank (using 90% rule) of ", rank, "for layer: ", linear_layer)
            elif set_rank is not None:
                rank = min(set_rank, dim[1])
            elif set_rank is None and criterion is None:
                print("\tExcluding layer")
                continue

            decomposed = svd_decomposition_linear_layer(linear_layer, rank)

            model._modules[name] = decomposed

    return model


def cp_decompose_model(model, exclude_first_conv=False, exclude_linears=False, passed_first_conv=False):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = cp_decompose_model(module, exclude_first_conv, exclude_linears, passed_first_conv)
        elif type(module) == nn.Conv2d:
            if passed_first_conv is False:
                passed_first_conv = True
                if exclude_first_conv is True:
                    continue

            conv_layer = module
            rank = cp_rank(conv_layer)
            print(conv_layer, "CP Estimated rank", rank)

            if (rank ** 2 >= conv_layer.in_channels * conv_layer.out_channels):
                print("(rank**2 >= conv_layer.in_channels * conv_layer.out_channels")
                continue

            decomposed = cp_decomposition_conv_layer(conv_layer, rank)

            model._modules[name] = decomposed
        elif type(module) == nn.Linear:
            if exclude_linears is True:
                continue

            # TODO: Revisit this part to decide how to deal with linear layer in CP Decomposition
            linear_layer = module
            rank = svd_rank_linear(linear_layer)
            print(linear_layer, "SVD Estimated Rank (using 90% rule): ", rank)

            decomposed = svd_decomposition_linear_layer(linear_layer, rank)

            model._modules[name] = decomposed

    return model


# This function was obtained from https://github.com/yuhuixu1993/Trained-Rank-Pruning/
def channel_decompose_model(model, layer_configs):
    '''
    a single NxCxHxW low-rank filter is decoupled
    into a NxRx1x1 kernel following a RxCxHxW kernel
    '''
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = channel_decompose_model(module, layer_configs)
        if type(module) == nn.Conv2d:
            conv_layer = module
            print(conv_layer)

            (set_rank, criterion) = layer_configs[conv_layer]

            if module.stride != (1, 1):
                print('\tNot supported stride (1,1)')
                continue

            param = conv_layer.weight.data
            dim = param.size()

            if set_rank is not None and criterion is not None:
                raise Exception("Can't have both pre-set rank and criterion for a layer")
            elif criterion is not None:
                rank = svd_rank_channel(conv_layer, criterion)
            elif set_rank is not None:
                rank = min(set_rank, dim[1])
            elif set_rank is None and criterion is None:
                print("\tExcluding layer")
                continue
            print("\tRank: ", rank)

            decomposed = channel_decomposition_conv_layer(conv_layer, rank)
            model._modules[name] = decomposed
        elif type(module) == nn.Linear:
            linear_layer = module
            print(linear_layer)

            (set_rank, criterion) = layer_configs[linear_layer]

            if set_rank is not None and criterion is not None:
                raise Exception("Can't have both pre-set rank and criterion for a layer")
            elif criterion is not None:
                rank = svd_rank_linear(linear_layer, criterion)
            elif set_rank is not None:
                rank = min(set_rank, dim[1])
            elif set_rank is None and criterion is None:
                print("\tExcluding layer")
                continue

            print("\tRank: ", rank)

            decomposed = svd_decomposition_linear_layer(linear_layer, rank)

            model._modules[name] = decomposed

    return model


# This function was obtained from https://github.com/yuhuixu1993/Trained-Rank-Pruning/
def depthwise_decompose_model(model, layer_configs):
    '''
    a single NxCxHxW low-rank filter is decoupled
    into a parrallel path composed of point-wise conv followed by depthwise conv
    '''
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = depthwise_decompose_model(module, layer_configs)
        elif type(module) == nn.Conv2d:
            conv_layer = module
            print(conv_layer)

            (set_rank, criterion) = layer_configs[conv_layer]

            if conv_layer.kernel_size == (1, 1):
                print("\tNot valid for filter size (1,1)")
                continue

            if conv_layer.groups > 1:
                print("\tCan't decompose depthwise conv")
                continue

            param = conv_layer.weight.data
            dim = param.size()

            if set_rank is not None and criterion is not None:
                raise Exception("Can't have both pre-set rank and criterion for a layer")
            elif criterion is not None:
                rank = svd_rank_depthwise(conv_layer, criterion)
            elif set_rank is not None:
                rank = min(set_rank, min(dim[2] * dim[3], dim[1]))
            elif set_rank is None and criterion is None:
                print("\tExcluding layer")
                continue
            print("\tRank: ", rank)

            decomposed = depthwise_decomposition_conv_layer(conv_layer, name, rank)
            model._modules[name] = decomposed
        elif type(module) == nn.Linear:
            linear_layer = module
            print(linear_layer)

            (set_rank, criterion) = layer_configs[linear_layer]

            if set_rank is not None and criterionc is not None:
                raise Exception("Can't have both pre-set rank and criterion for a layer")
            elif criterion is not None:
                rank = svd_rank_linear(linear_layer, criterion)
            elif set_rank is not None:
                rank = min(set_rank, min(dim[2] * dim[3], dim[1]))
            elif set_rank is None and criterion is None:
                print("\tExcluding layer")
                continue

            print("\tRank: ", rank)

            decomposed = svd_decomposition_linear_layer(linear_layer, rank)

            model._modules[name] = decomposed

    return model


# This function was obtained from https://github.com/yuhuixu1993/Trained-Rank-Pruning/
def spatial_decompose_model(model, layer_configs):
    '''
    a single NxCxHxW low-rank filter is decoupled
    into a RxCx1xW kernel and a NxRxHx1 kernel
    '''
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = spatial_decompose_model(module, layer_configs)
        elif type(module) == nn.Conv2d:
            conv_layer = module
            print(conv_layer)

            (set_rank, criterion) = layer_configs[conv_layer]

            if module.stride != (1, 1):
                print('Not decomposing', name, ' because its stride is not (1,1)')

            param = conv_layer.weight.data
            dim = param.size()

            if set_rank is not None and criterion is not None:
                raise Exception("Can't have both pre-set rank and criterion for a layer")
            elif criterion is not None:
                rank = svd_rank_spatial(conv_layer, criterion)
            elif set_rank is not None:
                rank = min(set_rank, dim[1])
            elif set_rank is None and criterion is None:
                print("\tExcluding layer")
                continue
            print("\tRank: ", rank)

            decomposed = spatial_decomposition_conv_layer(conv_layer, rank)
            model._modules[name] = decomposed
        elif type(module) == nn.Linear:
            linear_layer = module
            print(linear_layer)

            (set_rank, criterion) = layer_configs[linear_layer]

            if set_rank is not None and criterionc is not None:
                raise Exception("Can't have both pre-set rank and criterion for a layer")
            elif criterion is not None:
                rank = svd_rank_linear(linear_layer, criterion)
            elif set_rank is not None:
                rank = min(set_rank, min(dim[2] * dim[3], dim[1]))
            elif set_rank is None and criterion is None:
                print("\tExcluding layer")
                continue

            print("\tRank: ", rank)

            decomposed = svd_decomposition_linear_layer(linear_layer, rank)

            model._modules[name] = decomposed

    return model


def pd_conv(cin, cout, kernel, stride, pad, bias):
    return nn.Sequential(
        OrderedDict([
            ('pw', nn.Conv2d(cin, cout, 1, 1, 0, bias=False)),
            ('dw', nn.Conv2d(cout, cout, kernel, stride, pad, groups=cout, bias=bias))
        ])
    )


class MultiPathConv(nn.Module):
    '''
    a sub module structure used for network decouple as follows

         /--conv 1--\
        /            \
    --> ----conv 2--->+--->
        \            /
         \--conv n--/
    '''

    def __init__(self, n, cin, cout, kernel, pad, stride, bias):
        super(MultiPathConv, self).__init__()

        self.path_num = n
        self.path = nn.ModuleList([pd_conv(cin, cout, kernel, stride, pad, bias=(i == 0 and bias)) for i in range(n)])

    def forward(self, x):
        output = 0.0
        for m in self.path:
            output += m(x)
        return output


def cp_decomposition_conv_layer(layer, rank):
    """ Gets a conv layer and a target rank,
        returns a nn.Sequential object with the decomposition """

    # Perform CP decomposition on the layer weight tensorly.
    last, first, vertical, horizontal = \
        parafac(layer.weight.data, rank=rank, init='svd')

    pointwise_s_to_r_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
                                             out_channels=first.shape[1], kernel_size=1, stride=1, padding=0,
                                             dilation=layer.dilation, bias=False)

    depthwise_vertical_layer = torch.nn.Conv2d(in_channels=vertical.shape[1],
                                               out_channels=vertical.shape[1], kernel_size=(vertical.shape[0], 1),
                                               stride=1, padding=(layer.padding[0], 0), dilation=layer.dilation,
                                               groups=vertical.shape[1], bias=False)

    depthwise_horizontal_layer = \
        torch.nn.Conv2d(in_channels=horizontal.shape[1], \
                        out_channels=horizontal.shape[1],
                        kernel_size=(1, horizontal.shape[0]), stride=layer.stride,
                        padding=(0, layer.padding[0]),
                        dilation=layer.dilation, groups=horizontal.shape[1], bias=False)

    pointwise_r_to_t_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
                                             out_channels=last.shape[0], kernel_size=1, stride=1,
                                             padding=0, dilation=layer.dilation, bias=True)

    if layer.bias is not None:
        pointwise_r_to_t_layer.bias.data = layer.bias.data

    depthwise_horizontal_layer.weight.data = \
        torch.transpose(horizontal, 1, 0).unsqueeze(1).unsqueeze(1)
    depthwise_vertical_layer.weight.data = \
        torch.transpose(vertical, 1, 0).unsqueeze(1).unsqueeze(-1)
    pointwise_s_to_r_layer.weight.data = \
        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    pointwise_r_to_t_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)

    new_layers = [pointwise_s_to_r_layer, depthwise_vertical_layer, \
                  depthwise_horizontal_layer, pointwise_r_to_t_layer]

    return nn.Sequential(*new_layers)


def cp_decomposition_conv_layer_other(layer, rank):
    W = layer.weight.data

    last, first, vertical, horizontal = parafac(W, rank=rank, init='random')

    pointwise_s_to_r_layer = nn.Conv2d(in_channels=first.shape[0],
                                       out_channels=first.shape[1],
                                       kernel_size=1,
                                       padding=0,
                                       bias=False)

    depthwise_r_to_r_layer = nn.Conv2d(in_channels=rank,
                                       out_channels=rank,
                                       kernel_size=vertical.shape[0],
                                       stride=layer.stride,
                                       padding=layer.padding,
                                       dilation=layer.dilation,
                                       groups=rank,
                                       bias=False)

    pointwise_r_to_t_layer = nn.Conv2d(in_channels=last.shape[1],
                                       out_channels=last.shape[0],
                                       kernel_size=1,
                                       padding=0,
                                       bias=True)

    if layer.bias is not None:
        pointwise_r_to_t_layer.bias.data = layer.bias.data

    sr = first.t_().unsqueeze_(-1).unsqueeze_(-1)
    rt = last.unsqueeze_(-1).unsqueeze_(-1)
    rr = torch.stack([vertical.narrow(1, i, 1) @ torch.t(horizontal).narrow(0, i, 1) for i in range(rank)]).unsqueeze_(
        1)

    pointwise_s_to_r_layer.weight.data = sr
    pointwise_r_to_t_layer.weight.data = rt
    depthwise_r_to_r_layer.weight.data = rr

    new_layers = [pointwise_s_to_r_layer,
                  depthwise_r_to_r_layer, pointwise_r_to_t_layer]
    return new_layers


def svd_rank(weight, criterion):
    _, S, _ = torch.svd(weight, some=True)  # tl.partial_svd(weight, min(weight.shape))

    return criterion(S)


def svd_rank_linear(layer, criterion=EnergyThreshold(0.85)):
    return svd_rank(layer.weight.data, criterion)


def svd_rank_depthwise(conv_layer, criterion=EnergyThreshold(0.85)):
    param = conv_layer.weight.data
    dim = param.size()

    valid_idx = []
    # compute average rank according to criterion
    for i in range(dim[0]):
        W = param[i, :, :, :].view(dim[1], -1)
        U, sigma, V = torch.svd(W, some=True)
        valid_idx.append(criterion(sigma))

    item_num = min(max(valid_idx), min(dim[2] * dim[3], dim[1]))
    return item_num


def svd_rank_channel(conv_layer, criterion=EnergyThreshold(0.85)):
    param = conv_layer.weight.data
    dim = param.size()

    NC = param.view(dim[0], -1)  # [N x CHW]

    N, sigma, C = torch.svd(NC, some=True)
    C = C.t()
    # remain large singular value
    valid_idx = criterion(sigma)

    return valid_idx


def svd_rank_spatial(conv_layer, criterion=EnergyThreshold(0.85)):
    param = conv_layer.weight.data
    dim = param.size()

    VH = param.permute(1, 2, 0, 3).contiguous().view(dim[1] * dim[2], -1)

    V, sigma, H = torch.svd(VH, some=True)
    H = H.t()
    # remain large singular value
    valid_idx = criterion(sigma)

    return valid_idx


def tucker1_rank(layer):
    weights = layer.weight.data

    _, diag, _, _ = VBMF.EVBMF(weights)

    rank = diag.shape[0]
    return rank


def tucker_ranks(layer):
    """ Unfold the 2 modes of the Tensor the decomposition will
    be performed on, and estimates the ranks of the matrices using VBMF
    """

    weights = layer.weight.data

    unfold_0 = tl.base.unfold(weights, 0)
    unfold_1 = tl.base.unfold(weights, 1)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)

    ranks = [diag_0.shape[0], diag_1.shape[1]]
    return ranks


def cp_rank(layer):
    weights = layer.weight.data

    # Method used in previous repo
    # rank = max(layer.weight.shape)//3
    # return rank

    unfold_0 = tl.base.unfold(weights, 0)
    unfold_1 = tl.base.unfold(weights, 1)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)

    rank = max([diag_0.shape[0], diag_1.shape[0]])
    return rank


def tucker_decomposition_conv_layer(layer, ranks):
    """ Gets a conv layer,
        returns a nn.Sequential object with the Tucker decomposition.
        The ranks are estimated with a Python implementation of VBMF
        https://github.com/CasvandenBogaard/VBMF
    """
    core, [last, first] = \
        partial_tucker(layer.weight.data, \
                       modes=[0, 1], rank=ranks, init='svd')

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
                                  out_channels=first.shape[1], kernel_size=1,
                                  stride=1, padding=0, dilation=layer.dilation, bias=False)

    # A regular 2D convolution layer with R3 input channels
    # and R3 output channels
    core_layer = torch.nn.Conv2d(in_channels=core.shape[1], \
                                 out_channels=core.shape[0], kernel_size=layer.kernel_size,
                                 stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                                 bias=False)

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
                                 out_channels=last.shape[0], kernel_size=1, stride=1,
                                 padding=0, dilation=layer.dilation, bias=True)

    if layer.bias is not None:
        last_layer.bias.data = layer.bias.data

    first_layer.weight.data = \
        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)


def tucker1_decomposition_conv_layer(layer, rank):
    core, last = \
        partial_tucker(layer.weight.data, \
                       modes=[0], rank=rank, init='svd')

    '''
    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
            out_channels=first.shape[1], kernel_size=1,
            stride=1, padding=0, dilation=layer.dilation, bias=False)
    '''

    # A regular 2D convolution layer with R3 input channels
    # and R3 output channels
    core_layer = torch.nn.Conv2d(in_channels=core.shape[1], \
                                 out_channels=core.shape[0], kernel_size=layer.kernel_size,
                                 stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                                 bias=False)

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
                                 out_channels=last.shape[0], kernel_size=1, stride=1,
                                 padding=0, dilation=layer.dilation, bias=True)

    if layer.bias is not None:
        last_layer.bias.data = layer.bias.data

    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [core_layer, last_layer]
    return nn.Sequential(*new_layers)


def svd_decomposition_linear_layer(layer, rank):
    [U, S, V] = tl.partial_svd(layer.weight.data, rank)

    first_layer = torch.nn.Linear(in_features=V.shape[1], out_features=V.shape[0], bias=False)
    second_layer = torch.nn.Linear(in_features=U.shape[1], out_features=U.shape[0], bias=True)

    if layer.bias is not None:
        second_layer.bias.data = layer.bias.data

    first_layer.weight.data = (V.t() * S).t()
    second_layer.weight.data = U

    new_layers = [first_layer, second_layer]
    return nn.Sequential(*new_layers)


def channel_decomposition_conv_layer(module, rank):
    param = module.weight.data
    dim = param.size()

    if module.bias is not None:
        hasb = True
        b = module.bias.data
    else:
        hasb = False

    NC = param.view(dim[0], -1)  # [N x CHW]

    N, sigma, C = torch.svd(NC, some=True)
    C = C.t()
    N = N[:, :rank].contiguous()
    sigma = sigma[:rank]
    C = C[:rank, :]

    r = int(sigma.size(0))
    C = torch.mm(torch.diag(torch.sqrt(sigma)), C)
    N = torch.mm(N, torch.diag(torch.sqrt(sigma)))

    C = C.view(r, dim[1], dim[2], dim[3])
    N = N.view(dim[0], r, 1, 1)

    first_layer = nn.Conv2d(dim[1], r, dim[2], 1, 1, bias=False)
    first_layer.weight.data = C

    second_layer = nn.Conv2d(r, dim[0], 1, 1, 0, bias=hasb)
    second_layer.weight.data = N
    second_layer.bias = module.bias

    new_layers = [first_layer, second_layer]

    decomposed = nn.Sequential(*new_layers)
    return decomposed


def depthwise_decomposition_conv_layer(module, name, rank):
    param = module.weight.data
    dim = param.size()

    if module.bias is not None:
        hasb = True
        b = module.bias.data
    else:
        hasb = False

    try:
        pw = [param.new_zeros((dim[0], dim[1], 1, 1)) for i in range(rank)]
        dw = [param.new_zeros((dim[0], 1, dim[2], dim[3])) for i in range(rank)]

        # svd decoupling
        for i in range(dim[0]):
            W = param[i, :, :, :].view(dim[1], -1)
            U, sigma, V = torch.svd(W, some=True)
            V = V.t()
            U = U[:, :rank].contiguous()
            V = V[:rank, :].contiguous()
            sigma = torch.diag(torch.sqrt(sigma[:rank]))
            U = U.mm(sigma)
            V = sigma.mm(V)
            V = V.view(rank, dim[2], dim[3])
            for j in range(rank):
                pw[j][i, :, 0, 0] = U[:, j]
                dw[j][i, 0, :, :] = V[j, :, :]
    except:
        raise Exception('svd failed during decoupling')

    new_layers = MultiPathConv(rank, cin=dim[1], cout=dim[0], kernel=module.kernel_size, stride=module.stride,
                               pad=module.padding, bias=hasb)

    state_dict = new_layers.state_dict()
    for i in range(rank):
        dest = 'path.%d.pw.weight' % i
        src = '%s.weight' % name
        # print(dest+' <-- '+src)
        state_dict[dest].copy_(pw[i])

        dest = 'path.%d.dw.weight' % i
        # print(dest+' <-- '+src)
        state_dict[dest].copy_(dw[i])

        if i == 0 and hasb:
            dest = 'path.%d.dw.bias' % i
            src = '%s.bias' % name
            # print(dest+' <-- '+src)
            state_dict[dest].copy_(b)

    new_layers.load_state_dict(state_dict)

    return new_layers


def spatial_decomposition_conv_layer(module, rank):
    # the module should be decoupled
    param = module.weight.data
    if module.bias is not None:
        hasb = True
        b = module.bias.data  # Tensor size N
    else:
        hasb = False

    dim = param.size()
    VH = param.permute(1, 2, 0, 3).contiguous().view(dim[1] * dim[2], -1)

    try:
        V, sigma, H = torch.svd(VH, some=True)
        H = H.t()
        V = V[:, :rank].contiguous()
        sigma = sigma[:rank]
        H = H[:rank, :]
    except:
        raise Exception('svd failed during decoupling')

    # when decoupling, only conv with 1x1 stride is considered
    if module.stride != (1, 1):
        return module

    r = int(sigma.size(0))
    H = torch.mm(torch.diag(sigma), H).contiguous()

    H = H.view(r, dim[0], dim[3], 1).permute(1, 0, 3, 2)
    V = V.view(dim[1], 1, dim[2], r).permute(3, 0, 2, 1)

    new_layers = nn.Sequential(
        OrderedDict([
            ('V', nn.Conv2d(dim[1], r, kernel_size=(int(dim[2]), 1), stride=(1, 1), padding=(module.padding[0], 0),
                            bias=False)),
            ('H', nn.Conv2d(r, dim[0], kernel_size=(1, int(dim[3])), stride=(1, 1), padding=(0, module.padding[1]),
                            bias=hasb))])
    )

    state = new_layers.state_dict()
    # print(name+'.V.weight' + ' <-- ' + name+'.weight')
    state['V.weight'].copy_(V)
    # print(name+'.H.weight' + ' <-- ' + name+'.weight')
    state['H.weight'].copy_(H)

    if module.bias is not None:
        # print(name+'.H.bias' + ' <-- ' + name+'.bias')
        state['H.bias'].copy_(b)

    new_layers.load_state_dict(state)
    return new_layers
