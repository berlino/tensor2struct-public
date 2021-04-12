import torch


def get_field_presence_info(ast_wrapper, node, field_infos):
    present = []
    for field_info in field_infos:
        field_value = node.get(field_info.name)
        is_present = field_value is not None and field_value != []

        maybe_missing = field_info.opt or field_info.seq
        is_builtin_type = field_info.type in ast_wrapper.primitive_types

        if maybe_missing and is_builtin_type:
            # TODO: make it posible to deal with "singleton?"
            present.append(is_present and type(field_value).__name__)
        elif maybe_missing and not is_builtin_type:
            present.append(is_present)
        elif not maybe_missing and is_builtin_type:
            present.append(type(field_value).__name__)
        elif not maybe_missing and not is_builtin_type:
            assert is_present
            present.append(True)
    return tuple(present)


def lstm_init(device, num_layers, hidden_size, *batch_sizes):
    init_size = batch_sizes + (hidden_size,)
    if num_layers is not None:
        init_size = (num_layers,) + init_size
    init = torch.zeros(*init_size, device=device)
    return (init, init)


def maybe_stack(items, dim=None):
    to_stack = [item for item in items if item is not None]
    if not to_stack:
        return None
    elif len(to_stack) == 1:
        return to_stack[0].unsqueeze(dim)
    else:
        return torch.stack(to_stack, dim)


def accumulate_logprobs(d, keys_and_logprobs):
    for key, logprob in keys_and_logprobs:
        existing = d.get(key)
        if existing is None:
            d[key] = logprob
        else:
            d[key] = torch.logsumexp(torch.stack((logprob, existing), dim=0), dim=0)
