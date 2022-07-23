"""
File adapted from https://github.com/JavierAntoran/Bayesian-Neural-Networks
"""

from torch.optim.optimizer import Optimizer, required
import numpy as np
import torch


class SGLD(Optimizer):
    """
    SGLD optimiser based on pytorch's SGD.
    Note that the weight decay is specified in terms of the gaussian prior sigma.
    """

    def __init__(self, params, lr=1e-2, prior_sigma=np.inf):

        weight_decay = 1 / (prior_sigma ** 2)

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)

        super(SGLD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """
        Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:

            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                langevin_noise = p.data.new(p.data.size()).normal_(mean=0, std=1) / np.sqrt(group['lr'])
                p.data = p.data.add(0.5 * d_p + langevin_noise, alpha=-group['lr'])

        return loss


class pSGLD(Optimizer):
    """
    RMSprop preconditioned SGLD using pytorch rmsprop implementation.
    """

    def __init__(self, params, lr=1e-2, prior_sigma=np.inf, alpha=0.99, eps=1e-8):

        weight_decay = 1 / (prior_sigma ** 2)

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, weight_decay=weight_decay, alpha=alpha, eps=eps)
        super(pSGLD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """
        Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:

            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']
                state['step'] += 1

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # sqavg x alpha + (1-alph) sqavg *(elemwise) sqavg
                square_avg.mul_(alpha).addcmul_(d_p, d_p, value=1 - alpha)

                avg = square_avg.sqrt().add_(group['eps'])
                langevin_noise = p.data.new(p.data.size()).normal_(mean=0, std=1) / np.sqrt(group['lr'])
                p.data = p.data.add(0.5 * d_p.div_(avg) + langevin_noise / torch.sqrt(avg), alpha=-group['lr'])

        return loss
