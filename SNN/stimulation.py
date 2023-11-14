import torch
import numpy as np
import matplotlib.pyplot as plt

def create_edge_stimulus(center_x, center_y, max_sensor_height=19, max_sensor_width=16, 
                         image_height=640, image_width=480, width=1, amplitude=10, rotation_angle=0):
    """
    Create an edge stimulus using Gaussian distribution around the center.
    """
    y_coords, x_coords = torch.meshgrid(torch.linspace(0, max_sensor_height, image_height),
                                        torch.linspace(0, max_sensor_width, image_width))
    x_coords = x_coords-center_x  # shift origin to (center_x, center_y)
    y_coords = y_coords-center_y

    # Calculate the Gaussian distribution around the center
    return amplitude * torch.exp(-torch.square(x_coords * np.sin(rotation_angle) + y_coords * np.cos(rotation_angle)) / 
                                 (2 * np.square(width)))

def generate_stimulus_series(rotation_degree, num_frames, image_height, image_width, amplitude=5, visualize=False, device='cpu'):
    """
    Generate a series of edge stimuli frames with specified rotation.
    """
    stimuli_series = torch.zeros((image_height, image_width, num_frames), device=device)
    theta = rotation_degree * np.pi / 180.0

    for frame_num in range(num_frames):
        center_x = 500 * 0.3 / 17  # Move the stimulus by 0.3mm for each frame
        center_y = 500 * 0.3 / 15
        
        single_stimulus = create_edge_stimulus(center_x, center_y, amplitude=amplitude, 
                                               image_height=image_height, image_width=image_width, 
                                               rotation_angle=theta)
        stimuli_series[:, :, frame_num] = single_stimulus

        if visualize and frame_num == num_frames // 2:
            plt.imshow(stimuli_series[:, :, frame_num].cpu(), cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.show()

    return stimuli_series

