import torch

def get_sgd_optimizer(
    policies, learning_rate, momentum=None, weight_decay=None
):
    return torch.optim.SGD(policies, learning_rate, momentum=momentum, weight_decay=weight_decay)

def get_adamw_optimizer(
    policies, learning_rate, weight_decay=None
):
    return torch.optim.AdamW(policies, learning_rate, weight_decay=weight_decay)
