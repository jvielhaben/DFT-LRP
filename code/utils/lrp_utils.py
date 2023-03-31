import numpy as np

import torch
import torch.nn as nn

import zennit.composites
import zennit.rules
import zennit.core
import zennit.attribution
from zennit.types import Linear

from functools import partial

import matplotlib.pyplot as plt


def one_hot(output, index=0, cuda=True):
    '''Get the one-hot encoded value at the provided indices in dim=1'''
    values = output[np.arange(output.shape[0]),index]
    mask = torch.eye(output.shape[1])[index]
    if cuda:
        mask = mask.cuda()
    out =  values[:, None] * mask
    return out


def zennit_relevance(input, model, target, attribution_method="lrp", zennit_choice="EpsilonPlus", rel_is_model_out=True, cuda=True):
    input = torch.tensor(input, dtype=torch.float32, requires_grad=True)
    if cuda:
        input = input.cuda()

    if attribution_method=="lrp":
        relevance = zennit_relevance_lrp(input, model, target, zennit_choice, rel_is_model_out, cuda)
    elif attribution_method=="gxi" or attribution_method=="sensitivity":
        attributer = zennit.attribution.Gradient(model)
        _, relevance = attributer(input, partial(one_hot, index=target, cuda=cuda))
        relevance = relevance.detach().cpu().numpy()
        if attribution_method=="gxi":
            relevance = relevance * input.detach().cpu().numpy()
        elif attribution_method=="sensitivity":
            relevance = relevance**2
    elif attribution_method=="ig":
        attributer = zennit.attribution.IntegratedGradients(model)
        _, relevance = attributer(input, partial(one_hot, index=target, cuda=cuda))
        relevance = relevance.detach().cpu().numpy()
    return relevance


def zennit_relevance_lrp(input, model, target, zennit_choice="EpsilonPlus", rel_is_model_out=True, cuda=True):
    """
    zennit_choice: str, zennit rule or composite
    """
    if zennit_choice=="EpsilonPlus":
        lrp = zennit.composites.EpsilonPlus()
    elif zennit_choice=="EpsilonAlpha2Beta1":
        lrp = zennit.composites.EpsilonAlpha2Beta1()

    # register hooks for rules to all modules that apply
    lrp.register(model)

    # execute the hooked/modified model
    output = model(input)

    target_output = one_hot(output.detach(), target, cuda) 
    if not rel_is_model_out:
        target_output[:,target] = torch.sign(output[:,target])
    # compute the attribution via the gradient
    relevance = torch.autograd.grad(output, input, grad_outputs=target_output)[0]

    # remove all hooks, undoing the modification
    lrp.remove()

    return relevance.cpu().numpy()
