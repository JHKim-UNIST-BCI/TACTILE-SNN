
import torch
import torch.nn as nn

class IzhikevichLayer(nn.Module):
    def __init__(
        self, a: float = 0.02, b: float = 0.2 , c: float = -65, d: float = 8, 
        size: int = 1,
        v_thres: float = 30,
        a_decay: float = 1,
        buffer_size: int = 10,
        noise_std: int = 2,
        device: str = 'cpu'
    ):
        super(IzhikevichLayer, self).__init__()

        self.device = torch.device(device)
        self.a, self.b, self.c, self.d = a, b, c, d
        self.size = size

        self.v_thres = v_thres
        self.noise_std = noise_std
        self.buffer_size = buffer_size
        self.a_decay = a_decay
        self.reset()

    def reset(self) -> None:
        self.v = torch.full((self.size,), self.c, dtype=torch.float32, device=self.device)
        self.u = torch.full((self.size,), self.b * self.c, dtype=torch.float32, device=self.device)
        self.spikes = torch.zeros((self.size,), device=self.device)
        self.decay_a = torch.full((self.size,), self.a, device=self.device)
        self.spike_buffer = torch.zeros((self.size, self.buffer_size), device=self.device)
        
    def forward(self, I):
        I = I.squeeze()
        dv = (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + I)
        du = (self.a * (self.b * self.v - self.u))
        
        self.v += dv
        self.u += du
        self.spikes = (self.v >= self.v_thres).float()
        
        self.v[self.v >= 30] = self.c
        self.u[self.v >= 30] += self.d
        
        self.spike_buffer = torch.roll(self.spike_buffer, shifts=-1, dims=1)
        self.spike_buffer[:, -1] = self.spikes

        return self.spikes