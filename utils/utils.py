import numpy as np

def euler_to_angular_vel(euler, euler_vel):
    "zyz euler to angular velocity"
    euler_to_angular = np.array([
        [0, -np.sin(euler[0]), np.cos(euler[0]) * np.sin(euler[1])],
        [0, np.cos(euler[0]), np.sin(euler[0]) * np.sin(euler[1])],
        [1, 0, np.cos(euler[1])]
    ])
    return euler_to_angular @ euler_vel
