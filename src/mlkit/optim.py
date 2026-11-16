import numpy as np
import torch

class Optimizer:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
        self.t = 0

    def step(self, grads):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.buf = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        for p, g, b in zip(self.params, grads, self.buf):
            if self.momentum:
                b *= self.momentum
                b += g
                p -= self.lr * b
            else:
                p -= self.lr * g


class RMSprop(Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8):
        super().__init__(params, lr)
        self.alpha, self.eps = alpha, eps
        self.sq = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        for p, g, s in zip(self.params, grads, self.sq):
            s *= self.alpha
            s += (1 - self.alpha) * g * g
            p -= self.lr * g / (np.sqrt(s) + self.eps)


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, decoupled=False):
        super().__init__(params, lr)
        self.betas, self.eps = betas, eps
        self.weight_decay, self.decoupled = weight_decay, decoupled
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        b1, b2 = self.betas
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            if self.weight_decay:
                if self.decoupled:
                    p *= 1 - self.lr * self.weight_decay      # AdamW: shrink weights directly
                else:
                    g = g + self.weight_decay * p             # Adam + L2: add to the gradient
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            m_hat = m / (1 - b1 ** self.t)
            v_hat = v / (1 - b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(params, lr, betas, eps, weight_decay, decoupled=True)


class MyAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                g = p.grad

                p.mul_(1 - group["lr"] * group["weight_decay"])          # decoupled decay
                state["m"].mul_(beta1).add_(g, alpha=1 - beta1)
                state["v"].mul_(beta2).addcmul_(g, g, value=1 - beta2)

                m_hat = state["m"] / (1 - beta1 ** t)
                v_hat = state["v"] / (1 - beta2 ** t)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(group["eps"]), value=-group["lr"])
        return loss