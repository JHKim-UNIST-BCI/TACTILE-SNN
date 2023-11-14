import torch
import torch.nn as nn
from SNN.Izhikevich import *
from SNN.synapse import *
from SNN.receptive_field import *

class SNN(nn.Module):
    def __init__(self, device):
        super(SNN, self).__init__()

        # Initialize receptive fields
        self.R = ReceptiveFieldWeights(pixel_h=320, pixel_w=240, device=device, type_output='single', plot_receptive_field=False)
        
        self.sa_rf_dim = [self.R.sa_cn_pn_step_height, self.R.sa_cn_pn_step_width]
        self.ra_rf_dim = [self.R.ra_cn_pn_step_height, self.R.ra_cn_pn_step_width]
        self.cn_rf_dim = [self.R.cn_pn_ra_rf_step_height, self.R.cn_pn_ra_rf_step_width]
        
        noise_std_val = 0 
        
        # Convert layers and synapses to ModuleList for better PyTorch optimization
        self.sa_layers = nn.ModuleList([
            IzhikevichLayer(0.02, 0.2, -65, 8, len(self.R.sa_rf), v_thres=30, a_decay=1, noise_std=noise_std_val, device=device),
            IzhikevichLayer(0.1, 0.2, -65, 6, len(self.R.sa_cn_in_rf), v_thres=30, a_decay=1, noise_std=noise_std_val, device=device),
            IzhikevichLayer(0.1, 0.2, -65, 6, len(self.R.sa_cn_pn_rf), v_thres=30, a_decay=1, noise_std=noise_std_val, device=device)
        ])
        
        self.ra_layers = nn.ModuleList([
            IzhikevichLayer(0.02, 0.2, -65, 2, len(self.R.ra_rf), v_thres=30, a_decay=1, noise_std=noise_std_val, device=device),
            IzhikevichLayer(0.1, 0.2, -65, 2, len(self.R.ra_cn_in_rf), v_thres=30, a_decay=1, noise_std=noise_std_val, device=device),
            IzhikevichLayer(0.1, 0.2, -65, 2, len(self.R.ra_cn_pn_rf), v_thres=30, a_decay=1, noise_std=noise_std_val, device=device)
        ])

        self.cn_layers = nn.ModuleList([
            IzhikevichLayer(0.02, 0.2, -65, 8, len(self.R.cn_in_sa_rf), v_thres=30, a_decay=1, noise_std=noise_std_val, device=device),
            IzhikevichLayer(0.02, 0.2, -65, 8, len(self.R.cn_pn_sa_rf), v_thres=30, a_decay=1, noise_std=noise_std_val, device=device)
        ])

        self.sa_synapses = nn.ModuleList([
            Synapse(self.R.sa_rf, device=device),
            Synapse(self.R.sa_cn_in_rf, delays=self.R.sa_cn_SD, device=device),
            Synapse(self.R.sa_cn_pn_rf, delays=self.R.sa_cn_SD, device=device),
            Synapse(self.R.sa_intopn_rf, delays=self.R.sa_intopn_DN, tau_psp=10, device=device)
        ])

        self.ra_synapses = nn.ModuleList([
            Synapse(self.R.ra_rf, device=device),
            Synapse(self.R.ra_cn_in_rf, delays=self.R.ra_cn_SD, device=device),
            Synapse(self.R.ra_cn_pn_rf, delays=self.R.ra_cn_SD, device=device),
            Synapse(self.R.ra_intopn_rf, delays=self.R.ra_intopn_DN, tau_psp=10, device=device)
        ])

        self.cn_synapses = nn.ModuleList([
            Synapse(self.R.cn_in_sa_rf, delays=self.R.cn_sa_SD, device=device),
            Synapse(self.R.cn_pn_sa_rf, delays=self.R.cn_sa_SD, device=device),
            Synapse(self.R.cn_in_ra_rf, delays=self.R.cn_ra_SD, device=device),
            Synapse(self.R.cn_pn_ra_rf, delays=self.R.cn_ra_SD, device=device),
            Synapse(self.R.cn_intopn_rf, delays=self.R.cn_intopn_DN, tau_psp=10, device=device)
        ])

        self.reset_model()

    def reset_model(self):
        for layer_set, synapse_set in zip([self.sa_layers, self.ra_layers, self.cn_layers], [self.sa_synapses, self.ra_synapses, self.cn_synapses]):
            for layer, synapse in zip(layer_set, synapse_set):
                layer.reset()
                synapse.reset()

    def forward(self, stim, diff_stim):
        sa_input = torch.matmul(self.R.sa_rf, stim.squeeze().reshape(-1))
        ra_input = torch.matmul(self.R.ra_rf, diff_stim.squeeze().reshape(-1))

        # SA updates
        self.sa_layers[0](sa_input)
        self.sa_layers[1](self.sa_synapses[1](self.sa_layers[0].spike_buffer))
        sa_rf_input_PN1 = self.sa_synapses[2](self.sa_layers[0].spike_buffer)
        sa_rf_input_PN2 = self.sa_synapses[3](self.sa_layers[1].spike_buffer)
        self.sa_layers[2](sa_rf_input_PN1 * 2 - sa_rf_input_PN2 * 1)

        # RA updates
        self.ra_layers[0](ra_input)
        self.ra_layers[1](self.ra_synapses[1](self.ra_layers[0].spike_buffer))
        ra_rf_input_PN1 = self.ra_synapses[2](self.ra_layers[0].spike_buffer)
        ra_rf_input_PN2 = self.ra_synapses[3](self.ra_layers[1].spike_buffer)
        self.ra_layers[2](ra_rf_input_PN1 * 2 - ra_rf_input_PN2 * 1)

        # CN updates
        cn_IN_sa_rf_input = self.cn_synapses[0](self.sa_layers[2].spike_buffer)
        cn_IN_ra_rf_input = self.cn_synapses[2](self.ra_layers[2].spike_buffer)
        self.cn_layers[0](cn_IN_sa_rf_input + cn_IN_ra_rf_input)
        cn_PN_sa_rf_input = self.cn_synapses[1](self.sa_layers[2].spike_buffer)
        cn_PN_ra_rf_input = self.cn_synapses[3](self.ra_layers[2].spike_buffer)
        cn_IN_rf_input = self.cn_synapses[4](self.cn_layers[0].spike_buffer)
        input_value = cn_PN_sa_rf_input * 2 + cn_PN_ra_rf_input * 2 - cn_IN_rf_input
        self.cn_layers[1](input_value)

        return self.sa_layers[0].spikes, self.ra_layers[0].spikes
