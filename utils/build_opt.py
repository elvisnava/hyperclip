import torch

supported_optims = ["adam", "amsgrad", "sgd", "rmsprop", "adamw"]


def build_optimizer(parameters, config, loop="inner"):
    # loop should be either "inner" or "hyperclip"
    in_str = f"{loop}_"
    optim = config[in_str+"optimizer"]
    lr = config[in_str+"learning_rate"]
    weight_decay = config.get(in_str+"weight_decay", 0)
    adam_beta1 = config.get(in_str+"adam_beta1", 0.9)
    adam_beta2 = config.get(in_str+"adam_beta2", 0.999)
    momentum = config.get(in_str+"momentum", 0)
    sgd_dampening = config.get(in_str+"sgd_dampening", False)
    sgd_nesterov = config.get(in_str+"sgd_nesterov", False)
    rmsprop_alpha = config.get(in_str+"rmsprop_alpha", 0.99)

    if optim not in supported_optims:
        raise ValueError("Unsupported optim: {}. Must be one of {}".format(optim, supported_optims))

    if optim == "adam":
        optimizer = torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == "amsgrad":
        optimizer = torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif optim == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    elif optim == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )

    elif optim == "adamw":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    return optimizer
