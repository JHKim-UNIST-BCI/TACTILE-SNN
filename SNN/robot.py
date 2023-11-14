#Robot,Gripper and DIGIT Tactile sensor
import cv2
import os
import math
import time
import urx
import logging
import socket
import keyboard
import numpy as np
import datetime
import matplotlib.pyplot as plt
import random
import threading
import torch

# logging.basicConfig(level=logging.INFO)

class Robot:
    RESPONSE_SIZE = 2**14
    RETRY_DELAY = 3

    def __init__(self, robot_ip, gripper_host, gripper_port):
        self._position = 0
        self.robot = self._connect_with_retry(lambda: self._initialize_robot_connection(robot_ip))
        self.gripper = self._connect_with_retry(lambda: self._initialize_gripper_connection(gripper_host, gripper_port))
        self.home_j = [math.radians(value) for value in [90, -90, 90, -90, -90, 180]]
        self.home_l = [0.11, -0.3, 0.2, -math.radians(180), 0, 0]
        self.grip_l = [0.11, -0.3, 0.065, -math.radians(180), 0, 0]
        self._connect_with_retry(lambda: self._initialize_digit_sensor())

        self.latest_frame = None
        self.latest_frame_torch = None
        self.capture_thread = threading.Thread(target=self._capture_digit_image)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        if not self.is_gripper_active():
            self.grip_init()

    def _capture_digit_image(self):
        while True:
            ret, frame = self.digit1.read()
            if ret:
                rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                grayscale_frame = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)
                self.latest_frame = grayscale_frame
                self.latest_tensor = torch.tensor(self.latest_frame[:, :, np.newaxis]).float()

    def is_gripper_active(self):
        self.gripper.sendall(b'GET ACT\n')
        response = self.gripper.recv(self.RESPONSE_SIZE).decode()
        return "ACT 1" in response

    def get_joint_positions(self):
        radian_positions = self.robot.getj()
        joint_positions = [math.degrees(pos) for pos in radian_positions]
        return joint_positions

    def _connect_with_retry(self, connection_method):
        while True:
            try:
                return connection_method()
            except Exception as e:
                logging.error(f"Failed to connect, retrying in {self.RETRY_DELAY} seconds: {e}")
                time.sleep(self.RETRY_DELAY)

    def _initialize_digit_sensor(self, camera_index=0):
        self.digit1 = cv2.VideoCapture(camera_index)
        
        desired_width = 320
        desired_height = 240
        self.digit1.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
        self.digit1.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
        self.digit1.set(cv2.CAP_PROP_FPS, 60)

        if not self.digit1.isOpened():
            raise Exception("Failed to initialize DIGIT Tactile sensor.")
        else:
            print(f"DIGIT Tactile sensor initialized at camera index: {camera_index} with {desired_width}x{desired_height}@{self.digit1.get(cv2.CAP_PROP_FPS)}FPS")


    def _initialize_robot_connection(self, ip):
        rob = urx.Robot(ip)
        rob.set_tcp((0, 0, 0.19, 0, 0, 0))
        rob.set_payload(0.5, (0, 0, 0))
        return rob
    
    def _initialize_gripper_connection(self, host, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        return s
    
    def move_to(self, position, acc=1, vel=1, wait= True, relative = False):
        self.robot.movel(position, acc, vel, wait = wait, relative = relative)

    def move_to_thread(self, position, acc=1, vel=1, wait=True, relative=False):
        threading.Thread(target=self._move_to_thread, args=(position, acc, vel, wait, relative)).start()

    def _move_to_thread(self, position, acc=1, vel=1, wait=True, relative=False):
        self.robot.movel(position, acc, vel, wait = wait, relative = relative)

    def control_with_arrows(self, step_size=0.005): 
        print("Press arrow keys to control the robot. Press 'q' to quit.")
        while True:
            if keyboard.is_pressed('up'):
                self.move_to([0, 0, step_size, 0, 0, 0], relative=True)
                time.sleep(0.05) 
            elif keyboard.is_pressed('down'):
                self.move_to([0, 0, -step_size, 0, 0, 0], relative=True)
                time.sleep(0.05)
            elif keyboard.is_pressed('left'):
                self.move_to([-step_size, 0, 0, 0, 0, 0], relative=True)
                time.sleep(0.05)
            elif keyboard.is_pressed('right'):
                self.move_to([step_size, 0, 0, 0, 0, 0], relative=True)
                time.sleep(0.05)
            elif keyboard.is_pressed('w'):
                self.move_to([0, step_size, 0, 0, 0, 0], relative=True)
                time.sleep(0.05)
            elif keyboard.is_pressed('s'):
                self.move_to([0, -step_size, 0, 0, 0, 0], relative=True)
                time.sleep(0.05)
            elif keyboard.is_pressed('space'):
                self.pos = 255 - self.pos
            elif keyboard.is_pressed('q'):
                print("Exiting control mode.")
                break
            elif keyboard.is_pressed('h'):
                self.robot.movej(self.home_j, acc=0.1, vel=0.1, wait=True, relative=False)
    
        ####################### gripper ########################
    def grip_init(self, FOR=100, SPE=255, POS=0):
        # 사용자에게 그리퍼 내에 오브젝트가 있는지 확인
        response = input("Is there an object inside the gripper? (yes/no): ")
        if response.strip().lower() == 'yes':
            print("Please remove the object from the gripper and try again.")
            return

        # 오브젝트가 없는 경우, 그리퍼 초기화 진행
        self.gripper.sendall(b'SET ACT 1\n')
        self.gripper.recv(2**10)
        self.gripper.sendall(b'SET GTO 1\n')
        self.gripper.recv(2**10)
        time.sleep(3)
        self.gripper.sendall(b'SET FOR ' + str(FOR).encode() + b'\n')
        self.gripper.recv(2**10)
        self.gripper.sendall(b'SET SPE ' + str(SPE).encode() + b'\n')
        self.gripper.recv(2**10)
        self.gripper.sendall(b'SET POS ' + str(POS).encode() + b'\n')
        self.gripper.recv(2**10)

        self._position = 0

    def get_grip_settings(self):
        settings = ['ACT', 'GTO', 'FOR', 'SPE', 'POS','OBJ', 'STA']
        for setting in settings:
            self.gripper.sendall(b'GET ' + setting.encode() + b'\n')
            response = self.gripper.recv(2**14).decode()
            print(f"{response}")

    def set_grip(self,name,val):
        self.gripper.sendall(b'SET '+name.encode()+b' '+str(val).encode()+b'\n')
        self.gripper.recv(2**10)

    @property
    def pos(self):
        return self._get_current_gripper_pos()

    @property
    def obj(self):
        return self._get_current_gripper_obj()

    @pos.setter
    def pos(self, val):
        threading.Thread(target=self._set_pos_thread, args=(val,)).start()

    def _set_pos_thread(self, val):
        self.set_grip('POS', val)

    def close_connections(self):
        try:
            self.robot.close()
        except Exception as e:
            logging.error("Failed to close connections: " + str(e))

    def _get_current_gripper_pos(self):
        try:
            self.gripper.sendall(b'GET POS\n')
            response = self.gripper.recv(self.RESPONSE_SIZE).decode()
            return int(float(response.split(' ')[1]))
        except Exception as e:
            logging.error(f"Error getting gripper position: {e}")
            return 0  # or some default or previous value
        
    def _get_current_gripper_obj(self):
        try:
            self.gripper.sendall(b'GET OBJ\n')
            response = self.gripper.recv(self.RESPONSE_SIZE).decode()
            return int(float(response.split(' ')[1]))
        except Exception as e:
            logging.error(f"Error getting gripper OBJ: {e}")
            return 0  # or some default or previous value
        
    def get_digit_image(self, display=True):
        if display and self.latest_frame is not None:
            cv2.imshow("DIGIT Tactile Sensor Image", self.latest_frame)
        return self.latest_frame
        
    def record_stimuli(self, stimuli, record=False, display=False, repetitions=10, relative_angles=[0, 45, 90]):
        data_dir = "Data"
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        for stimulus in stimuli:
            input(f"Please change to the {stimulus}. Press Enter when ready...")
            initial_joint_positions = self.robot.getj()  # 초기 관절 각도를 저장

            first_rep_images = []  # 첫 번째 repetition의 이미지만 저장

            for i in range(repetitions):
                frames = []
                for rel_angle in relative_angles:
                    # Adjust the joint angle
                    self.set_relative_joint_angle(initial_joint_positions, rel_angle)
                    
                    self.pos = 210
                    time.sleep(1)

                    frame = self.get_digit_image(display=False)
                    if frame is not None:
                        if record:
                            frames.append(frame)

                        if i == 0:  # 첫 번째 repetition만 저장
                            first_rep_images.append(frame)

                    self.pos = 0
                    time.sleep(1)

                    if record:
                        self.save_digit1_data(stimulus, frames, i+1, rel_angle)

                self.robot.movej(initial_joint_positions, acc=1, vel=1, wait=True)

            # 첫 번째 repetition의 이미지를 각도별로 한 figure에 보여줌
            if display:
                fig, axes = plt.subplot(1, len(relative_angles), figsize=(15, 5))
                fig.suptitle(stimulus)
                for ax, image, angle in zip(axes, first_rep_images, relative_angles):
                    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    ax.set_title(f"{angle} degrees")
                    ax.axis("off")
                plt.tight_layout()
                plt.show()

    def set_relative_joint_angle(self, initial_joint_positions, relative_angle):
        adjusted_joint_positions = list(initial_joint_positions)
        adjusted_joint_positions[-1] -= math.radians(relative_angle)
        self.robot.movej(adjusted_joint_positions, acc=1, vel=1, wait=True)

    def save_digit1_data(self, stimulus, frames, repetition, angle):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stimulus}_{timestamp}_{repetition}reps_{angle}deg.npy"
        filepath = os.path.join("Data", filename)
        np.save(filepath, frames)

    def angle_grip(self, relative_angles=[0, 45, 90], capture=False, display=False):
        initial_joint_positions = self.robot.getj()
        captured_images = []
        grip_positions = []

        for rel_angle in relative_angles:
            self.set_relative_joint_angle(initial_joint_positions, rel_angle)

            self.pos = 210
            time.sleep(1)

            if capture or display:
                frame = self.get_digit_image(display=False)
                if frame is not None:
                    captured_images.append(frame)
                    grip_positions.append(int(self.pos))

            self.pos = 0
            time.sleep(1)

        self.robot.movej(initial_joint_positions, acc=1, vel=1, wait=True)
        
        if display:
            fig, axes = plt.subplot(1, len(relative_angles), figsize=(15, 5))
            fig.suptitle("Captured Images per Angle")
            for ax, (angle, image) in zip(axes, captured_images):
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                ax.set_title(f"{angle} degrees")
                ax.axis("off")
            plt.tight_layout()
            plt.show()
        
        return captured_images, grip_positions
    
    def slow_press(self, capture=False, display=False):
        captured_images = []
        grip_positions = []

        self.set_grip('SPE', 0)
        before_press_frame = self.get_digit_image(display=False)

        self.pos = 255
        time.sleep(5)

        if capture or display:
            after_press_frame = self.get_digit_image(display=False)
            if after_press_frame is not None:
                captured_images.append(after_press_frame)
                grip_positions.append(int(self.pos))

        self.pos = 0
        self.set_grip('SPE', 127)

        if display:
            fig, axes = plt.subplot(1, 3, figsize=(20, 5))
            
            # Before Press Image
            axes[0].imshow(cv2.cvtColor(before_press_frame, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Before Press")
            axes[0].axis("off")
            
            # After Press Image
            axes[1].imshow(cv2.cvtColor(after_press_frame, cv2.COLOR_BGR2RGB))
            axes[1].set_title("After Press")
            axes[1].axis("off")
            
            # Difference between Before and After Press Image
            difference_frame = cv2.absdiff(before_press_frame, after_press_frame)
            amplified_difference = cv2.convertScaleAbs(difference_frame, alpha=5, beta=0)  # 증폭
            axes[2].imshow(cv2.cvtColor(amplified_difference, cv2.COLOR_BGR2RGB))
            axes[2].set_title("Amplified Difference (After - Before)")
            axes[2].axis("off")
            
            plt.tight_layout()
            plt.show()

        return captured_images, grip_positions

    def sweep_move(self, step_size=0.005, capture=False, display=False, repetitions=1):
        captured_frames = []
        initial_position = self.robot.getl()  # 로봇의 초기 위치 저장
        initial_grip_pos = self.pos  # 그리퍼의 초기 위치 저장

        self.pos = 210
        time.sleep(2)
        self.pos = int(self.pos) - 19
        time.sleep(2)

        for _ in range(repetitions):

            self.robot.movel([0, 0, step_size, 0, 0, 0], relative=True, acc=0.5, vel=0.5)
            if capture or display:
                frame = self.get_digit_image(display=False)
                if frame is not None:
                    captured_frames.append(frame)
            
            self.robot.movel([0, 0, -step_size, 0, 0, 0], relative=True, acc=0.5, vel=0.5)
            if capture or display:
                frame = self.get_digit_image(display=False)
                if frame is not None:
                    captured_frames.append(frame)

        self.robot.movel(initial_position, acc=0.5, vel=0.5, wait=True)
        self.pos = 0

        if display:
            fig, axes = plt.subplot(1, len(captured_frames), figsize=(15, 5))
            for ax, image in zip(axes, captured_frames):
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                ax.axis("off")
            plt.tight_layout()
            plt.show()

        return captured_frames

    def move_to_location(self, idx):
        loc_list = [
            [0.21, -0.21, 0.06, 3.14, 0.01, 0.00],
            [0.21, -0.38, 0.06, 3.13, -0.00, -0.00],
            [-0.04, -0.38, 0.06, 3.13, -0.02, -0.01],
            [-0.04, -0.21, 0.06, 3.13, -0.02, -0.01]
        ]
        
        current_loc = self.robot.getl()
        elevated_loc = current_loc.copy()
        elevated_loc[2] = 0.14
        
        target_loc = loc_list[idx-1].copy()
        elevated_target_loc = target_loc.copy()
        elevated_target_loc[2] = 0.14
        
        self.robot.movel(elevated_loc, acc=0.5, vel=0.5, wait=True)
        self.robot.movel(elevated_target_loc, acc=0.5, vel=0.5, wait=True)
        self.robot.movel(target_loc, acc=0.5, vel=0.5, wait=True)

    def grip_until_contact(self, threshold=150000):
        initial_frame = self.get_digit_image(display=False)
        
        while self.pos < 255:
            self.pos = int(self.pos) + 1
            time.sleep(0.001)  # 그리퍼 움직임 간격 조절
            
            current_frame = self.get_digit_image(display=False)
            
            difference_frame = cv2.absdiff(initial_frame, current_frame)
            diff_value = np.sum(difference_frame)
            print(diff_value)
            
            if diff_value > threshold:
                break  # 센서 변화가 임계값 이상이면 그리퍼 움직임 중지

    def record_experiment(self, stimuli=["soft_base", "baseball", "tennis_ball", "soft_tennis"], 
                        record=True, display=False, repetitions=10, relative_angles=[0, 45, 90]):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = f"Experimental_Data_{timestamp}"
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        for idx, stimulus in enumerate(stimuli, start=1):
            self.move_to_location(idx)  # Move the robot to the stimulus location

            for rep in range(repetitions):
                angle_grip_images, angle_grip_positions = self.angle_grip(relative_angles=relative_angles, capture=True, display=display)
                time.sleep(1)
                slow_press_images, slow_press_positions = self.slow_press(capture=True, display=display)
                time.sleep(1)

                if record:
                    # Save images and grip positions for angle grip experiment
                    np.save(os.path.join(data_dir, f"{stimulus}_{rep+1}_ang_images.npy"), angle_grip_images)
                    np.save(os.path.join(data_dir, f"{stimulus}_{rep+1}_ang_positions.npy"), angle_grip_positions)

                    # Save images and grip positions for slow press experiment
                    np.save(os.path.join(data_dir, f"{stimulus}_{rep+1}_slow_images.npy"), slow_press_images)
                    np.save(os.path.join(data_dir, f"{stimulus}_{rep+1}_slow_positions.npy"), slow_press_positions)



    def record_experiment_randomAngle(self, stimuli=["soft_base", "baseball", "tennis_ball", "soft_tennis"], 
                                    record=True, display=False, repetitions=10):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = f"Experimental_Data_{timestamp}_randomangles"
        self.set_grip("FOR", 0)
        self.set_grip("SPE", 127)
        self.pos = 0
        if record and not os.path.exists(data_dir):
            os.mkdir(data_dir)

        for idx, stimulus in enumerate(stimuli, start=1):
            self.move_to_location(idx)  # Move the robot to the stimulus location
            current_j = self.robot.getj()
            time.sleep(2)
            for rep in range(repetitions):
                # -90에서 90 사이의 랜덤한 각도를 얻습니다.
                rel_angle = random.uniform(-90, 90)
                radians = math.radians(rel_angle)
                print("Random Angle in Degrees:", rel_angle)
                print("Random Angle in Radians:", radians)

                # 마지막 관절의 현재 각도에 랜덤 각도를 더합니다.
                
                new_j = current_j.copy()
                new_j[-1] += radians

                # 로봇을 새로운 관절 각도로 움직입니다.
                print("Moving robot to new joint angles:", new_j)
                self.robot.movej(new_j, acc=1, vel=0.5, wait=True)  # 움직임의 속도와 가속도를 조절할 수 있습니다.

                # 그립을 255로 설정하여 오브젝트를 잡습니다.
                self.pos = 255
                time.sleep(1)  # 오브젝트를 잡는 데 시간이 필요합니다.

                # 데이터 캡쳐 및 저장
                angle_grip_images = self.get_digit_image(display=False)
                angle_grip_positions = self.pos
                
                if record:
                    # Save images and grip positions for angle grip experiment
                    np.save(os.path.join(data_dir, f"{stimulus}_{rep+1}_randang_images.npy"), angle_grip_images)
                    np.save(os.path.join(data_dir, f"{stimulus}_{rep+1}_randang_positions.npy"), angle_grip_positions)
                
                # 그립을 해제합니다.
                self.pos = 0
                time.sleep(1)

                # Short pause between iterations for safety and clarity
                time.sleep(0.5)

    def record_experiment_randomAngle_v2(self, stimuli=["soft_base", "baseball", "tennis_ball", "soft_tennis"], 
                                    record=True, display=False, repetitions=10):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = "Experimental_Data_randomangles"  # 폴더명을 고정합니다.
        self.set_grip("FOR", 0)
        self.set_grip("SPE", 127)
        self.pos = 0
        if record and not os.path.exists(data_dir):
            os.mkdir(data_dir)

        for idx, stimulus in enumerate(stimuli, start=1):
            self.move_to_location(idx)  # Move the robot to the stimulus location
            current_j = self.robot.getj()
            time.sleep(2)
            for rep in range(repetitions):
                rel_angle = random.uniform(-90, 90)
                radians = math.radians(rel_angle)
                # print("Random Angle in Degrees:", rel_angle)
                # print("Random Angle in Radians:", radians)
                
                new_j = current_j.copy()
                new_j[-1] += radians

                # print("Moving robot to new joint angles:", new_j)
                self.robot.movej(new_j, acc=1, vel=0.5, wait=True)

                self.pos = 255
                time.sleep(1)  

                angle_grip_images = self.get_digit_image(display=False)
                angle_grip_positions = self.pos
                
                if record:
                    # 파일명에 timestamp를 포함시킵니다.
                    np.save(os.path.join(data_dir, f"{stimulus}_{rep+1}_randang_images_{timestamp}.npy"), angle_grip_images)
                    np.save(os.path.join(data_dir, f"{stimulus}_{rep+1}_randang_positions_{timestamp}.npy"), angle_grip_positions)
                
                self.pos = 0
                time.sleep(1)


    def record_experiment_randomAngle_randomZ(self, stimuli=["soft_base", "baseball", "tennis_ball", "soft_tennis"], 
                                    record=True, display=False, repetitions=10):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = "Experimental_Data_randomangles_randomz"  # 폴더명을 고정합니다.
        self.set_grip("FOR", 0)
        self.set_grip("SPE", 127)
        self.pos = 0
        if record and not os.path.exists(data_dir):
            os.mkdir(data_dir)

        for idx, stimulus in enumerate(stimuli, start=1):
            self.move_to_location(idx)  # Move the robot to the stimulus location
            current_l = self.robot.getl()
            current_j = self.robot.getj()
            time.sleep(2)

            for rep in range(repetitions):
                # Z축 랜덤이동
                random_z_l = self.robot.getl()
                z_noise = random.uniform(0.05, 0.07)
                random_z_l[2] = z_noise
                self.robot.movel(random_z_l, acc=1, vel=1, wait=True)

                # 각도 변경만 수행
                rel_angle = random.uniform(-90, 90)
                radians = math.radians(rel_angle)
                new_j = self.robot.getj()
                new_j[-1] = current_j[-1]
                new_j[-1] += radians
                self.robot.movej(new_j, acc=1, vel=0.5, wait=True)

                self.pos = 255
                time.sleep(1)

                angle_grip_images = self.get_digit_image(display=display)
                angle_grip_positions = self.pos
                
                if record:
                    # 파일명에 timestamp를 포함시킵니다.
                    np.save(os.path.join(data_dir, f"{stimulus}_{rep+1}_randang_images_{timestamp}.npy"), angle_grip_images)
                    np.save(os.path.join(data_dir, f"{stimulus}_{rep+1}_randang_positions_{timestamp}.npy"), angle_grip_positions)
                
                self.pos = 0
                time.sleep(1)

    