import torch as th

def count_parameters(model, return_top=False):
    """
    Count model parameters.
    """
    total_params = 0
    params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        total_params += parameter.numel()
        params.append((parameter.numel(), name))
    if not return_top:
        return total_params

    params = sorted(params, key=lambda x: x[0], reverse=True)
    top = params[:5]
    return total_params, top

def load_weights(model, weights_path):
    current_model_dict = model.state_dict()
    loaded_state_dict = th.load(weights_path, map_location="cpu")

    new_state_dict = {k: loaded_state_dict[k] if k in loaded_state_dict and current_model_dict[k].size() == loaded_state_dict[k].size() else current_model_dict[k] for k in current_model_dict.keys()}
    model.load_state_dict(new_state_dict)