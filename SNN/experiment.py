import numpy as np
import time
import datetime
import os
import random
import math
import matplotlib.pyplot as plt
import torch
import cv2
import threading

plt.style.use('ggplot')

BASELINE_SCALE = 1
DIFF_SCALE = 1
STIMULI=["soft_base", "baseball", "tennis_ball", "soft_tennis"]


def process_sensor_data(snn, robot, duration=1, display=True):
    snn.reset_model()
    sa_spikes_2d_list, ra_spikes_2d_list, cn_spikes_2d_list = [], [], []
    prev_stimulation = None
    baseline_tensor = torch.tensor(robot.latest_frame[:, :, np.newaxis]).float()

    loop_count = 0
    start_time = time.time()
    while time.time() - start_time < duration:
        current_frame = robot.latest_frame
        input_data = current_frame[:, :, np.newaxis]
        stimulation = (torch.tensor(input_data).float() - baseline_tensor) / BASELINE_SCALE
        
        if prev_stimulation is None:
            prev_stimulation = stimulation.clone()

        diff_stimulation = torch.abs(stimulation - prev_stimulation) * DIFF_SCALE

        snn(stimulation, diff_stimulation)

        sa_spikes_2d_list.append(snn.sa_layers[2].spikes.reshape(snn.sa_rf_dim))
        ra_spikes_2d_list.append(snn.ra_layers[2].spikes.reshape(snn.ra_rf_dim))
        cn_spikes_2d_list.append(snn.cn_layers[1].spikes.reshape(snn.cn_rf_dim))

        prev_stimulation = stimulation.clone()

        if loop_count == 0:
            robot.pos = 255

        loop_count += 1

    robot.pos = 0
    time.sleep(0.5)

    cn_spikes_2d_tensor = torch.stack(cn_spikes_2d_list)
    cn_firing_rate_2d = cn_spikes_2d_tensor.mean(dim=0) * 100

    if display:
        display_results(sa_spikes_2d_list, ra_spikes_2d_list, cn_spikes_2d_list)

    return cn_spikes_2d_list, cn_firing_rate_2d, robot.pos

def process_sensor_data_with_nogrip(snn, robot, baseline_tensor ,duration=1, display=True):
    snn.reset_model()
    sa_spikes_2d_list, ra_spikes_2d_list, cn_spikes_2d_list = [], [], []
    prev_stimulation = None

    loop_count = 0
    start_time = time.time()
    while time.time() - start_time < duration:
        current_frame = robot.latest_frame
        input_data = current_frame[:, :, np.newaxis]
        stimulation = (torch.tensor(input_data).float() - baseline_tensor) / BASELINE_SCALE
        
        if prev_stimulation is None:
            prev_stimulation = stimulation.clone()

        diff_stimulation = torch.abs(stimulation - prev_stimulation) * DIFF_SCALE

        snn(stimulation, diff_stimulation)

        sa_spikes_2d_list.append(snn.sa_layers[2].spikes.reshape(snn.sa_rf_dim))
        ra_spikes_2d_list.append(snn.ra_layers[2].spikes.reshape(snn.ra_rf_dim))
        cn_spikes_2d_list.append(snn.cn_layers[1].spikes.reshape(snn.cn_rf_dim))

        prev_stimulation = stimulation.clone()

        loop_count += 1

    cn_spikes_2d_tensor = torch.stack(cn_spikes_2d_list)
    cn_firing_rate_2d = cn_spikes_2d_tensor.mean(dim=0) * 100

    if display:
        display_results(sa_spikes_2d_list, ra_spikes_2d_list, cn_spikes_2d_list)

    return cn_spikes_2d_list, cn_firing_rate_2d


def display_results(sa_spikes_2d_list, ra_spikes_2d_list, cn_spikes_2d_list):
    plt.rcParams['font.family'] = 'Arial'  # or 'Times New Roman'
    plt.rcParams['font.size'] = 14

    loop_count = len(sa_spikes_2d_list)

    sa_firing_rate = torch.stack(sa_spikes_2d_list).reshape(loop_count, -1).mean(dim=1) * 100
    ra_firing_rate = torch.stack(ra_spikes_2d_list).reshape(loop_count, -1).mean(dim=1) * 100
    cn_firing_rate = torch.stack(cn_spikes_2d_list).reshape(loop_count, -1).mean(dim=1) * 100

    # Plot average firing rates
    plt.figure(figsize=(6, 3))
    plt.plot(sa_firing_rate, label='SA Firing Rate', color='red')
    plt.plot(ra_firing_rate, label='RA Firing Rate', color='blue')
    plt.plot(cn_firing_rate, label='CN Firing Rate', color='green')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Average Firing Rate (Hz)', fontsize=14)
    plt.title('Average Firing Rate Over Iterations', fontsize=16)
    plt.ylim([0, 100])  # Setting the y-axis limits
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # Display heatmaps
    heatmaps = [
        (torch.stack(sa_spikes_2d_list).mean(dim=0) * 100, "SA Firing Rate Heatmap"),
        (torch.stack(ra_spikes_2d_list).mean(dim=0) * 100, "RA Firing Rate Heatmap"),
        (torch.stack(cn_spikes_2d_list).mean(dim=0) * 100, "CN (PN) Firing Rate Heatmap")
    ]

    plt.figure(figsize=(9, 3))
    for idx, (data, title) in enumerate(heatmaps, start=1):
        plt.subplot(1, 3, idx)
        plt.imshow(data, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Firing Rate (Hz)').ax.tick_params(labelsize=12)
        plt.title(title, fontsize=16)
        plt.xlabel('X Coordinate', fontsize=14)
        plt.ylabel('Y Coordinate', fontsize=14)
        plt.clim(0, 100)  # Setting the colorbar limits
    plt.tight_layout()
    plt.show()
    

def record_experiment_randomAngle_v2_with_position(snn, robot, stimuli=STIMULI, repetitions=10, duration=1.5, display=False, record=False):
    data_dir = "Robot_RandomAngle_SNN_CNoutput_with_position_1101(2)"
    os.makedirs(data_dir, exist_ok=True)

    for idx, stimulus in enumerate(stimuli, start=1):
        robot.move_to_location(idx)
        current_l = robot.robot.getl()
        current_j = robot.robot.getj()
        time.sleep(2)

        for rep in range(repetitions):
            # Z축 랜덤이동
            random_z_l = robot.robot.getl()
            z_noise = random.uniform(0.05, 0.07)
            random_z_l[2] = z_noise
            robot.robot.movel(random_z_l, acc=1, vel=1, wait=True)

            # 각도 변경만 수행
            rel_angle = random.uniform(-90, 90)
            radians = math.radians(rel_angle)
            new_j = robot.robot.getj()
            new_j[-1] = current_j[-1]
            new_j[-1] += radians
            robot.robot.movej(new_j, acc=1, vel=0.5, wait=True)
            time.sleep(0.5)
            
            current_display = display if rep == 0 else False
            cn_spikes_2d_list,cn_firing_rate_2d ,position = process_sensor_data(snn, robot, duration=duration, display=current_display)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if record:
                filename_spikes = f"{stimulus}_{duration}s_duration_spikes_{timestamp}.npy"
                filepath_spikes = os.path.join(data_dir, filename_spikes)
                np.save(filepath_spikes, cn_spikes_2d_list)
                
                # Save the CN's firing rate with timestamp in filename
                filename_firing_rate = f"{stimulus}_{duration}s_duration_firing_rate_{timestamp}.npy"
                filepath_firing_rate = os.path.join(data_dir, filename_firing_rate)
                np.save(filepath_firing_rate, cn_firing_rate_2d)
                
                # Save the position with timestamp in filename
                filename_position = f"{stimulus}_{duration}s_duration_position_{timestamp}.npy"
                filepath_position = os.path.join(data_dir, filename_position)
                np.save(filepath_position, position)


def record_experiment_shake(snn, robot, stimuli=STIMULI, repetitions=10, duration=3, shake_intensity=0.05,shake_times=3, display=False, record=False):
    data_dir = "Robot_Shake_SNN_CNoutput_1101"
    os.makedirs(data_dir, exist_ok=True)
    robot.pos = 0
    for idx, stimulus in enumerate(stimuli, start=1):
        robot.move_to_location(idx)  # 로봇이 물체를 집으러 이동
        current_l = robot.robot.getl()  # 현재 위치 가져오기
        current_j = robot.robot.getj()  # 현재 각도 가져오기
        time.sleep(2)

        for rep in range(repetitions):
            baseline_tensor = torch.tensor(robot.latest_frame[:, :, np.newaxis]).float()
            robot.pos = 255
            time.sleep(0.5)
            position = robot.pos
            elevated_loc = current_l.copy()
            elevated_loc[2] += 0.05  # Z축을 조금 들어올림
            robot.robot.movel(elevated_loc, acc=1, vel=1, wait=True)
            
            current_display = display if rep == 0 else False
            sensor_thread = threading.Thread(target=process_sensor_data_with_nogrip, args=(snn, robot, duration, current_display))
            sensor_thread.start()
            # 로봇이 물체를 지정된 횟수만큼 흔드는 코드
            for _ in range(shake_times):
                shake_amplitude = random.uniform(-shake_intensity, shake_intensity)
                shaken_location = elevated_loc.copy()
                shaken_location[2] += shake_amplitude
                robot.robot.movel(shaken_location, acc=1, vel=1, wait=True)
                robot.robot.movel(elevated_loc, acc=1, vel=1, wait=True) 
            sensor_thread.join()

            # 센서 데이터 처리
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            robot.robot.movel(current_l, acc=1, vel=1, wait=True)
            robot.pos = 0
            time.sleep(0.5)

            # if record:
            #     filename_spikes = f"{stimulus}_{duration}s_duration_spikes_{timestamp}.npy"
            #     filepath_spikes = os.path.join(data_dir, filename_spikes)
            #     np.save(filepath_spikes, cn_spikes_2d_list)
                
            #     filename_firing_rate = f"{stimulus}_{duration}s_duration_firing_rate_{timestamp}.npy"
            #     filepath_firing_rate = os.path.join(data_dir, filename_firing_rate)
            #     np.save(filepath_firing_rate, cn_firing_rate_2d)
                
            #     filename_position = f"{stimulus}_{duration}s_duration_position_{timestamp}.npy"
            #     filepath_position = os.path.join(data_dir, filename_position)
            #     np.save(filepath_position, position)
