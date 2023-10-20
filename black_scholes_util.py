import torch
import math


# Methods to calculate the black scholes price
def black_scholes_call(s, k, t, r, sigma):
    d1 = (torch.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * torch.sqrt(t))
    d2 = d1 - sigma * torch.sqrt(t)

    n_d1 = 0.5 * (1 + torch.erf(d1 / math.sqrt(2)))
    n_d2 = 0.5 * (1 + torch.erf(d2 / math.sqrt(2)))

    call_price = s * n_d1 - k * torch.exp(-r * t) * n_d2
    return call_price


def black_scholes_put(s, k, t, r, sigma):
    d1 = (torch.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * torch.sqrt(t))
    d2 = d1 - sigma * torch.sqrt(t)

    n_d1 = 0.5 * (1 + torch.erf(-d1 / math.sqrt(2)))
    n_d2 = 0.5 * (1 + torch.erf(-d2 / math.sqrt(2)))

    put_price = k * torch.exp(-r * t) * n_d2 - s * n_d1
    return put_price
