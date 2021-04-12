import torch
import collections


def tensors_to_device(tensors, device=torch.device("cpu")):
    """Place a collection of tensors in a specific device"""
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, (list, tuple)):
        return type(tensors)(
            tensors_to_device(tensor, device=device) for tensor in tensors
        )
    elif isinstance(tensors, (dict, collections.OrderedDict)):
        return type(tensors)(
            [
                (name, tensors_to_device(tensor, device=device))
                for (name, tensor) in tensors.items()
            ]
        )
    else:
        raise NotImplementedError()
