import torch
import numpy as np
import matplotlib.pyplot as plt

class ReceptiveFieldWeights:
    def __init__(self, pixel_h=320, pixel_w=240, device = 'cpu', type_output = 'single', plot_receptive_field = False, plot_ind = 0):
        self.pixel_h = pixel_h
        self.pixel_w = pixel_w
        self.device = device
        self.type_output = type_output
        print(self.type_output)

        self.generate_primary_receptive_field_weights()
        self.generate_cuneate_nucleus_receptive_field_weights()
        self.generate_cortex_receptive_field_weights()

        if plot_receptive_field is True:
            self.plot_receptive_field(sa_ind = plot_ind, ra_ind = plot_ind, no_legend = False)

    def pick_points_in_rf(self, num_points=28, kernel_h=10, kernel_w=10, device='cpu'):
        arr = torch.zeros((kernel_h, kernel_w), device=device)
        center = np.array([kernel_h / 2, kernel_w / 2])
        
        # Adjust scale to spread points more widely
        scale = min(kernel_h, kernel_w) / 8
        
        kernel_dims = np.array([kernel_h, kernel_w])
        
        # Function to generate indices based on scale
        def generate_indices(scale_factor, num_pts):
            idx = np.random.normal(loc=center, scale=scale * scale_factor, size=(num_pts, 2)).astype(int)
            return np.clip(idx, 0, kernel_dims - 1)
        
        # Generate points with smaller scale
        indices = generate_indices(1, num_points)
        values = torch.rand(num_points).uniform_(0.1, 1).to(device)
        arr[indices[:, 0], indices[:, 1]] = values
        
        # Generate additional points with larger scale for outer region
        outer_indices = generate_indices(2, num_points)
        outer_values = torch.rand(num_points).uniform_(0.1, 1).to(device)
        arr[outer_indices[:, 0], outer_indices[:, 1]] = outer_values
        
        return arr
    
    def generate_mechanoreceptor_to_afferent_rf(self, pixel_h=320, pixel_w=240, kernel_w=9, kernel_h=11, step_size=6, device='cpu'):
        num_horizontal_steps = (pixel_w - kernel_w) // step_size + 1
        num_vertical_steps = (pixel_h - kernel_h) // step_size + 1

        receptive_fields = []
        for step_v in range(0, num_vertical_steps * step_size, step_size):
            for step_h in range(0, num_horizontal_steps * step_size, step_size):
                temp_rf = torch.zeros((pixel_h, pixel_w), device=device)
                temp_arr = self.pick_points_in_rf(num_points=28, kernel_h=kernel_h, kernel_w=kernel_w, device=device)
                temp_rf[step_v:step_v + kernel_h, step_h:step_h + kernel_w] = temp_arr
                receptive_fields.append(temp_rf)

        stacked_rf = torch.stack(receptive_fields)
        reshaped_rf = stacked_rf.reshape(stacked_rf.shape[0], -1)

        return reshaped_rf, (num_vertical_steps, num_horizontal_steps)
    
    def generate_receptive_field(self, rf_list, pixel_h=320, pixel_w=240, step_size=2, device='cpu'):
        rf_array = []
        rf_lengths = []

        for rf in rf_list:
            kernel_h, kernel_w = rf.shape
            num_horizontal_steps = (pixel_w - kernel_w) // step_size + 1
            num_vertical_steps = (pixel_h - kernel_h) // step_size + 1

            rf_lengths.append(num_horizontal_steps * num_vertical_steps)

            for step_v in range(0, pixel_h - kernel_h + 1, step_size):
                for step_h in range(0, pixel_w - kernel_w + 1, step_size):
                    temp_rf = torch.zeros((pixel_h, pixel_w), device=device)
                    temp_rf[step_v:step_v + kernel_h, step_h:step_h + kernel_w] = rf
                    rf_array.append(temp_rf)

        stacked_rf = torch.stack(rf_array).to(device)
        reshaped_rf = stacked_rf.reshape(stacked_rf.shape[0], -1)

        return reshaped_rf, (num_vertical_steps, num_horizontal_steps)

    
    def generate_neuron_connection_weight(self, input_neurons, output_neurons, connection_probability=0.2, device='cpu'):
        # Create a binary connectivity mask using the connection_probability
        connectivity_mask = (torch.rand(output_neurons, input_neurons, device=device) < connection_probability).float()

        # Directly initialize weights using the connectivity mask
        weights = torch.rand(output_neurons, input_neurons, device=device) * connectivity_mask

        # Create a delay matrix. Currently, all delays are set to 0.
        delays = torch.zeros(output_neurons, input_neurons, device=device)

        return weights, delays
    
    def generate_primary_receptive_field_weights(self):
        scale_factor = 5 if self.pixel_h == 320 else 1
        
        # Define a dictionary to store the configurations for SA and RA
        configurations = {
            "sa": {
                "kernel_h": 11 * scale_factor,
                "kernel_w": 9 * scale_factor,
                "step_size": 5 * scale_factor,
                "rf_pixel_h": self.pixel_h,
                "rf_pixel_w": self.pixel_w
            },
            "ra": {
                "kernel_h": 14 * scale_factor,
                "kernel_w": 11 * scale_factor,
                "step_size": 4 * scale_factor,
                "rf_pixel_h": self.pixel_h,
                "rf_pixel_w": self.pixel_w
            }
        }
        
        for key, config in configurations.items():
            rf, [rf_height, rf_width] = self.generate_mechanoreceptor_to_afferent_rf(
                pixel_h=config["rf_pixel_h"], pixel_w=config["rf_pixel_w"], 
                kernel_w=config["kernel_w"], kernel_h=config["kernel_h"],
                step_size=config["step_size"], device=self.device
            )
            setattr(self, f"{key}_rf", rf)
            setattr(self, f"{key}_rf_height", rf_height)
            setattr(self, f"{key}_rf_width", rf_width)
            
            # Print the shape of the rf variable
            print(f"{key}_rf shape:", rf.shape, 'with height =', rf_height, 'with width =', rf_width)

    def generate_cuneate_nucleus_receptive_field_weights(self):
        time_delay = 2

        cn_pn_rf = [torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]],device=self.device) * 4]
        cn_in_rf = [torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]],device=self.device)]
        cn_SD = [torch.full((3, 3), time_delay, device=self.device)]

        cn_intopn_rf = []

        # Check if the sizes of the inner tensors are different and print the index
        for i, (pn,IN) in enumerate(zip(cn_pn_rf, cn_in_rf)):
            if pn.size() != IN.size():
                raise ValueError(
                    f"The inner tensors at index {i} have different sizes: {pn.size()} != {IN.size()}")
            
        self.sa_cn_pn_rf, [self.sa_cn_pn_step_height, self.sa_cn_pn_step_width] = self.generate_receptive_field(cn_pn_rf, pixel_h=self.sa_rf_height,pixel_w=self.sa_rf_width, step_size=1, device=self.device)
        self.sa_cn_in_rf, [self.sa_cn_in_step_height, self.sa_cn_in_step_width] = self.generate_receptive_field(cn_in_rf, pixel_h=self.sa_rf_height,pixel_w=self.sa_rf_width, step_size=1, device=self.device)
        self.sa_cn_SD, [self.sa_cn_SD_step_height, self.sa_cn_SD_step_width]  = self.generate_receptive_field(cn_SD, pixel_h=self.sa_rf_height,pixel_w=self.sa_rf_width, step_size=1, device=self.device)
        self.ra_cn_pn_rf, [self.ra_cn_pn_step_height, self.ra_cn_pn_step_width] = self.generate_receptive_field(cn_pn_rf, pixel_h=self.ra_rf_height,pixel_w=self.ra_rf_width, step_size=1, device=self.device)
        self.ra_cn_in_rf, [self.ra_cn_in_step_height, self.ra_cn_in_step_width] = self.generate_receptive_field(cn_in_rf, pixel_h=self.ra_rf_height,pixel_w=self.ra_rf_width, step_size=1, device=self.device)
        self.ra_cn_SD, [self.ra_cn_SD_step_height, self.ra_cn_SD_step_width] = self.generate_receptive_field(cn_SD, pixel_h=self.ra_rf_height, pixel_w=self.ra_rf_width, step_size=1, device=self.device)

        self.sa_intopn_rf, self.sa_intopn_DN = self.generate_neuron_connection_weight(len(self.sa_cn_in_rf), len(self.sa_cn_pn_rf), connection_probability=0.2, device=self.device)
        self.ra_intopn_rf, self.ra_intopn_DN = self.generate_neuron_connection_weight(len(self.ra_cn_in_rf), len(self.ra_cn_pn_rf), connection_probability=0.2, device=self.device)

        print("sa_cn_pn_rf shape: ", self.sa_cn_pn_rf.shape,"sa_cn_pn_step_height:", self.sa_cn_pn_step_height,"sa_cn_pn_step_width:", self.sa_cn_pn_step_width)
        print("sa_cn_in_rf shape: ", self.sa_cn_in_rf.shape,"sa_cn_in_step_height:", self.sa_cn_in_step_height,"sa_cn_in_step_width:", self.sa_cn_in_step_width)
        print("ra_cn_pn_rf shape: ", self.ra_cn_pn_rf.shape,"ra_cn_pn_step_height:", self.ra_cn_pn_step_height,"ra_cn_pn_step_width:", self.ra_cn_pn_step_width)
        print("ra_cn_in_rf shape: ", self.ra_cn_in_rf.shape,"ra_cn_in_step_height:", self.ra_cn_in_step_height,"ra_cn_in_step_width:", self.ra_cn_in_step_width)
        print("sa_intopn_rf shape: ", self.sa_intopn_rf.shape)
        print("ra_intopn_rf shape: ", self.ra_intopn_rf.shape)

    def generate_gabor_kernels(self, ksize, theta_range,neg_scale_factor = 1):
        kernels_pos = []
        kernels_neg = []
        kernels_twos = []

        scale_factor = neg_scale_factor
        time_delay = 2
        for theta in theta_range:
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            kern -= np.mean(kern)  # Set the center value to 0
            kern /= np.abs(kern).sum()
            kern *= 20   # Adjust weight to maintain the same total sum

            pos = torch.tensor(np.maximum(0, kern), dtype=torch.float32)  * 1
            neg = torch.tensor(np.maximum(0, -kern), dtype=torch.float32) * 1
            twos = torch.tensor(np.full_like(kern, time_delay), dtype=torch.float32)
            # Zero out one half of the 'negative' kernel
            # mid_point = neg.shape[1] // 2
            # neg[:, :mid_point] = 0

            # Scale down the 'neg' tensor
            neg *= scale_factor

            kernels_pos.append(pos)
            kernels_neg.append(neg)
            kernels_twos.append(twos)

        # fig, axs = plt.subplots(3, 6, figsize=(15, 8))

        # for i, ax in enumerate(axs.flatten()):
        #     if i < len(kernels_pos):
        #         ax.imshow(kernels_pos[i], cmap='gray')
        #         ax.set_title(f"{(i*10)} degrees")
        #         ax.axis('off')

        # plt.tight_layout()
        # plt.show()

        # fig, axs = plt.subplots(3, 6, figsize=(15, 8))

        # for i, ax in enumerate(axs.flatten()):
        #     if i < len(kernels_neg):
        #         ax.imshow(kernels_neg[i], cmap='gray')
        #         ax.set_title(f"{(i*10)} degrees")
        #         ax.axis('off')

        # plt.tight_layout()
        # plt.show()

        # print(kernels_pos[2])
        # print(kernels_neg[2])

        return kernels_pos, kernels_neg, kernels_twos


    def generate_cortex_receptive_field_weights(self):
        if isinstance(self.type_output, list):  # Check if type_output is a list
            cn_pn_rf_set = []
            cn_in_rf_set = []
            cn_SD_set = []
            cn_pn_rf_ra_set = []
            cn_in_rf_ra_set = []
            cn_sd_rf_set = []

            for angle in self.type_output:
                # Generate receptive field for each angle
                ksize = 3
                theta_range = np.array([angle])
                kernels_pos, kernels_neg, kernels_twos = self.generate_gabor_kernels(ksize, theta_range, neg_scale_factor=0.5)
                cn_pn_rf_set.extend(kernels_pos)
                cn_in_rf_set.extend(kernels_neg)
                cn_SD_set.extend(kernels_twos)

                ksize = 5
                theta_range = np.array([angle])
                kernels_pos, kernels_neg, kernels_twos = self.generate_gabor_kernels(ksize, theta_range, neg_scale_factor=0.5)
                cn_pn_rf_ra_set.extend(kernels_pos)
                cn_in_rf_ra_set.extend(kernels_neg)
                cn_sd_rf_set.extend(kernels_twos)

        else:
            if self.type_output == 'single':
                cn_pn_rf_set = [torch.tensor([[0, 0, 0], [0, 0, 0], [1, 1 ,1]], device=self.device)]
                cn_in_rf_set = [torch.tensor([[0, 0, 0], [1, 1 ,1], [0, 0 ,0]], device=self.device)]
                cn_SD_set = [torch.tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]], device=self.device)]

                cn_pn_rf_ra_set = [torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[1, 1, 1 ,1, 1], [1, 1, 1 ,1, 1]], device=self.device)/10*1]

                cn_in_rf_ra_set = [torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1 ,1, 1], [1, 1, 1 ,1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], device=self.device)/10*1]

                cn_sd_rf_set = [torch.tensor([[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2]], device=self.device)]

            elif  self.type_output == 'angle0to180':
                ksize = 3

                theta_range = np.arange(0, np.pi, np.pi / 18)
                kernels_pos, kernels_neg, kernels_twos = self.generate_gabor_kernels(ksize, theta_range, neg_scale_factor= 0.5)
                cn_pn_rf_set = kernels_pos
                cn_in_rf_set = kernels_neg
                cn_SD_set = kernels_twos

                ksize = 5
                theta_range = np.arange(0, np.pi, np.pi / 18)
                kernels_pos, kernels_neg, kernels_twos = self.generate_gabor_kernels(ksize, theta_range, neg_scale_factor= 0.5)
                cn_pn_rf_ra_set = kernels_pos
                cn_in_rf_ra_set = kernels_neg
                cn_sd_rf_set = kernels_twos

            elif self.type_output == 'speed':
                cn_pn_rf_set = [torch.tensor([[1, 1 ,1], [0, 0, 0], [0, 0 ,0]], device=self.device)]
                cn_in_rf_set = [torch.tensor([[0, 0, 0], [1, 1 ,1], [0, 0 ,0]], device=self.device)]
                cn_SD_set = [torch.tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]], device=self.device)]

                cn_pn_rf_ra_set = [torch.tensor([[1, 1, 1 ,1, 1], [1, 1, 1 ,1, 1],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], device=self.device)/10*3]

                cn_in_rf_ra_set = [torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1 ,1, 1], [1, 1, 1 ,1, 1]], device=self.device)/10*3]

                cn_sd_rf_set = [torch.tensor([[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2]], device=self.device)]

            else:
                cn_pn_rf_set = [torch.tensor([[0, 0, 0], [0, 0, 0], [1, 1 ,1]], device=self.device),torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=self.device),torch.tensor([[0, 0, 1], [0, 0, 1], [0, 0 ,1]], device=self.device),
                    torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=self.device)]
                cn_in_rf_set = [torch.tensor([[0, 0, 0], [1, 1 ,1], [0, 0 ,0]], device=self.device),torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]], device=self.device)*3/2,torch.tensor([[0, 1, 0], [0, 1 ,0], [0, 1 ,0]], device=self.device),
                                torch.tensor([[0, 1, 0], [0, 0, 1], [0, 0, 0]], device=self.device)*3/2]
                cn_SD_set = [torch.tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]], device=self.device),torch.tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]], device=self.device),
                            torch.tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]], device=self.device),torch.tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]], device=self.device)]

                cn_pn_rf_ra_set = [torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[1, 1, 1 ,1, 1], [1, 1, 1 ,1, 1]], device=self.device)/10*3,
                                torch.tensor([[0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], device=self.device)/5*3,
                                torch.tensor([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1],[0, 0, 0, 1, 1],[0, 0, 0 ,1, 1], [0, 0, 0 ,1, 1]], device=self.device)/10*3,
                                torch.tensor([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], device=self.device)/5*3]

                cn_in_rf_ra_set = [torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1 ,1, 1], [1, 1, 1 ,1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], device=self.device)/10*3,
                                torch.tensor([[0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], device=self.device)/5*3,
                                torch.tensor([[0, 1, 1, 0, 0], [0, 1, 1, 0, 0],[0, 1, 1, 0, 0],[0, 1, 1, 0, 0], [0, 1, 1, 0, 0]], device=self.device)/10*3,
                                torch.tensor([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]], device=self.device)/5*3,
                                ]

                cn_sd_rf_set = [torch.tensor([[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2]], device=self.device),
                                torch.tensor([[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2]], device=self.device),
                                torch.tensor([[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2]], device=self.device),
                                torch.tensor([[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2],[2, 2, 2 ,2, 2]], device=self.device)]


        
        # cn_pn_rf, cn_pn_DN = create_weight_matrix(len(sa_cn_pn_rf)+len(ra_cn_pn_rf),num_output_neuron,connection_probability = 0.2, device = device)
        self.cn_pn_sa_rf, [self.cn_pn_sa_rf_step_height, self.cn_pn_sa_rf_step_width] = self.generate_receptive_field(cn_pn_rf_set, pixel_h=self.sa_cn_pn_step_height, pixel_w=self.sa_cn_pn_step_width, step_size=1, device=self.device)
        self.cn_in_sa_rf, [self.cn_in_sa_rf_step_height, self.cn_in_sa_rf_step_width] = self.generate_receptive_field(cn_in_rf_set, pixel_h=self.sa_cn_pn_step_height, pixel_w=self.sa_cn_pn_step_width, step_size=1, device=self.device)
        self.cn_sa_SD, [self.cn_sa_SD_step_height, self.cn_sa_SD_step_width] = self.generate_receptive_field(
            cn_SD_set, pixel_h=self.sa_cn_pn_step_height, pixel_w=self.sa_cn_pn_step_width, step_size=1, device=self.device)

        self.cn_pn_ra_rf, [self.cn_pn_ra_rf_step_height, self.cn_pn_ra_rf_step_width] = self.generate_receptive_field(cn_pn_rf_ra_set, pixel_h=self.ra_cn_pn_step_height, pixel_w=self.ra_cn_pn_step_width, step_size=1, device=self.device)
        self.cn_in_ra_rf, [self.cn_in_ra_rf_step_height, self.cn_in_ra_rf_step_width] = self.generate_receptive_field(cn_in_rf_ra_set, pixel_h=self.ra_cn_pn_step_height, pixel_w=self.ra_cn_pn_step_width, step_size=1, device=self.device)
        self.cn_ra_SD, [self.cn_ra_SD_step_height, self.cn_ra_SD_step_width] = self.generate_receptive_field(
            cn_sd_rf_set, pixel_h=self.ra_cn_pn_step_height, pixel_w=self.ra_cn_pn_step_width, step_size=1, device=self.device)

        self.cn_intopn_rf, self.cn_intopn_DN = self.generate_neuron_connection_weight(len(self.cn_in_sa_rf), len(self.cn_pn_sa_rf), connection_probability=0.2, device=self.device)

        print("cn_pn_sa_rf shape: ", self.cn_pn_sa_rf.shape, "cn_pn_sa_rf_step_height:", self.cn_pn_sa_rf_step_height, "cn_pn_sa_rf_step_width:", self.cn_pn_sa_rf_step_width)
        print("cn_in_sa_rf shape: ", self.cn_in_sa_rf.shape, "cn_in_sa_rf_step_height:", self.cn_in_sa_rf_step_height, "cn_in_sa_rf_step_width:", self.cn_in_sa_rf_step_width)
        print("cn_pn_ra_rf shape: ", self.cn_pn_ra_rf.shape, "cn_pn_ra_rf_step_height:", self.cn_pn_ra_rf_step_height, "cn_pn_ra_rf_step_width:", self.cn_pn_ra_rf_step_width)
        print("cn_in_ra_rf shape: ", self.cn_in_ra_rf.shape, "cn_in_ra_rf_step_height:", self.cn_in_ra_rf_step_height, "cn_in_ra_rf_step_width:", self.cn_in_ra_rf_step_width)
        print("cn_intopn_rf shape: ", self.cn_intopn_rf.shape)


    def plot_receptive_field(self, plot_vmax=10, plot_vmin=-10, sa_ind=10, ra_ind=10, no_legend=False):
        cmap = 'coolwarm'
        empty_ticks = []

        figures = [
            (self.cn_pn_sa_rf[sa_ind].reshape(self.sa_cn_pn_step_height, self.sa_cn_pn_step_width) * 1 - self.cn_in_sa_rf[sa_ind].reshape(self.sa_cn_pn_step_height, self.sa_cn_pn_step_width), 10, -10),
            (self.cn_pn_ra_rf[ra_ind].reshape(self.ra_cn_pn_step_height, self.ra_cn_pn_step_width) * 1 - self.cn_in_ra_rf[ra_ind].reshape(self.ra_cn_pn_step_height, self.ra_cn_pn_step_width), 10, -10),
            (torch.matmul(self.cn_pn_sa_rf[sa_ind].unsqueeze(0),self.sa_cn_pn_rf).squeeze(0).reshape(self.sa_rf_height, self.sa_rf_width)
             -torch.matmul(self.cn_in_sa_rf[sa_ind].unsqueeze(0),self.sa_cn_in_rf).squeeze(0).reshape(self.sa_rf_height, self.sa_rf_width),10,-10),
            (torch.matmul(self.cn_pn_ra_rf[ra_ind].unsqueeze(0),self.ra_cn_pn_rf).squeeze(0).reshape(self.ra_rf_height, self.ra_rf_width)
             -torch.matmul(self.cn_in_ra_rf[ra_ind].unsqueeze(0),self.ra_cn_in_rf).squeeze(0).reshape(self.ra_rf_height, self.ra_rf_width),10,-10),
        ]


        for fig, vmax, vmin in figures:
            plt.figure()
            plt.imshow(fig, cmap=cmap, vmax=vmax, vmin=vmin)
            plt.xticks(empty_ticks)
            plt.yticks(empty_ticks)
            if not no_legend:
                plt.colorbar()
            plt.show()
            print(torch.sum(fig))

        sa_weight = torch.matmul(self.cn_pn_sa_rf[sa_ind].unsqueeze(0),self.sa_cn_pn_rf).squeeze(0)-torch.matmul(self.cn_in_sa_rf[sa_ind].unsqueeze(0),self.sa_cn_in_rf).squeeze(0)
        self.sa_receptive_field = torch.matmul(sa_weight, self.sa_rf)

        ra_weight = torch.matmul(self.cn_pn_ra_rf[ra_ind].unsqueeze(0),self.ra_cn_pn_rf).squeeze(0)-torch.matmul(self.cn_in_ra_rf[ra_ind].unsqueeze(0),self.ra_cn_in_rf).squeeze(0)
        self.ra_receptive_field = torch.matmul(ra_weight, self.ra_rf)

        for field in [self.sa_receptive_field, self.ra_receptive_field]:
            plt.figure()
            plt.imshow(field.reshape(self.pixel_h, self.pixel_w), cmap=cmap, vmax=plot_vmax, vmin=plot_vmin)
            plt.xticks(empty_ticks)
            plt.yticks(empty_ticks)
            if not no_legend:
                plt.colorbar()
            plt.show()
            print(torch.sum(self.sa_receptive_field),torch.sum(self.ra_receptive_field))

