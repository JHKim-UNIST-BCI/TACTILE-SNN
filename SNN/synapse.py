import math

import torch
from torch import Tensor
from torch.nn import Module


class Synapse(Module):
    def __init__(
        self,
        weights: Tensor,
        tau_psp: float = 20,
        delays: Tensor = [],
        dt: float = 1,
        device: str = 'cpu',
    ):
        super().__init__()
        self.device = torch.device(device)
        self.tau_psp = tau_psp
        self.dt = dt
        self.delays_transposed = (torch.tensor(delays).t().long()).to(self.device)
        self.init_values(weights)

    def init_values(self, weights: Tensor) -> None:
        self.initial_weights = weights.clone().to(self.device)
        self.weights = weights.to(self.device)
        self.psp = torch.zeros(
            (weights.shape[0], weights.shape[1]), device=self.device)

    def reset(self) -> None:
        self.init_values(self.initial_weights)

    def cal_post_input(self, pre_spike_times: Tensor) -> Tensor:
        post_input = torch.matmul(self.weights, pre_spike_times)
        return post_input
    
    def forward(self, pre_spike_times: Tensor, buffer_size: int = 10) -> torch.Tensor:
        spikes_with_delay = torch.gather(pre_spike_times, 1, self.delays_transposed)
        self.psp.mul_(math.exp(-self.dt / self.tau_psp)).add_(spikes_with_delay.t())
        delayed_post_input = torch.einsum('ij,ij->i', self.weights, self.psp)
        return delayed_post_input
