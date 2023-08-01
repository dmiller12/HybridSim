import numpy as np
import mujoco
from math import sin, cos
from scipy.spatial.transform import Rotation as R

from utils.utils import euler_to_angular_vel


class TaskSpaceHybridController:
    def __init__(self, model, data, robot, Kp, Ki, Kd, Kfp, Kfi):

        self.model = model
        self.data = data
        self.robot = robot

        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_site")

        self.ef_sum = np.zeros(1)
        self.e_sum = np.zeros(4)

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Kfp = Kfp
        self.Kfi = Kfi

    def compute(self, s_d, lambda_d):
        Sf = np.array([[0, 0, 0, 0, 0, 1]]).T  # feasible wrench
        Sv = np.concatenate([np.identity(5), np.array([[0, 0, 0, 0, 0]])])  # feasible twist
        # Note Sv should have off diagonal components since df < 6

        M = np.zeros((4, 4))
        mujoco.mj_fullM(self.model, M, self.data.qM)

        J_temp = np.array(self.robot.get_J(self.data.qpos))
        J = J_temp[[3, 4, 5, 0, 1, 2]]  # angular, linear

        J_inv = np.linalg.pinv(J)
        mass_task = J_inv.T @ M @ J_inv

        J_dot_temp = self.robot.get_Jdot(self.data.qpos, self.data.qvel)
        J_dot = np.array(J_dot_temp)[[3, 4, 5, 0, 1, 2]]
        cor_grav_task = J_inv.T @ self.data.qfrc_bias - mass_task @ J_dot @ self.data.qvel

        V = J @ self.data.qvel
        V = np.array(V)

        s_dot = np.linalg.pinv(Sv) @ V[[3, 4, 5, 0, 1, 2]]

        desired_angular = euler_to_angular_vel(s_d[:3, 2], s_d[:3, 1])

        s_d[:3, 1] = desired_angular
        e_dot = s_d[:, 1] - s_dot

        pose = self.robot.fk(self.data.qpos)
        ee_rot = R.from_matrix(pose[:3, :3])

        # compute orientation error from quaternions which permits the use of geometric jacobian instead of analytic
        ee_rot_d = R.from_euler("zyz", s_d[:3, 2])
        rot_error = ee_rot_d * ee_rot.inv()
        error_quat = rot_error.as_quat()
        if error_quat[-1] < 0:  # canonical form scalar +ve
            error_quat = -1 * error_quat

        e = np.zeros(5)
        e[:3] = error_quat[:3]
        e[3:] = s_d[3:, 2] - pose[:2, -1]

        a_s = s_d[:, 0] + (self.Kd @ e_dot) + (self.Kp @ e)
        twist_comp = mass_task @ Sv @ a_s

        if self.data.ncon > 0:
            force = np.empty(6, dtype=np.double)
            mujoco.mj_contactForce(self.model, self.data, 0, force)
            # convert force in contact frame to tool frame
            contact_to_world = self.data.contact[0].frame.reshape(3, 3, order="F")
            contact_to_tool = pose[:3, :3].T @ contact_to_world
            force_tool = (
                np.c_[np.vstack([contact_to_tool, np.zeros((3, 3))]), np.vstack([np.zeros((3, 3)), contact_to_tool])]
                @ -force[[3, 4, 5, 0, 1, 2]]
            )
        else:
            force_tool = np.zeros(6)

        e_f = lambda_d - np.linalg.pinv(Sf) @ force_tool

        self.ef_sum += e_f * self.model.opt.timestep
        a_lambda = lambda_d + self.Kfp @ e_f + self.Kfi @ self.ef_sum
        wrench_comp = Sf @ a_lambda

        u = J.T @ (twist_comp + wrench_comp + cor_grav_task)
        return u


class JointSpaceHybridController:
    def __init__(self, model, data, Kp, Ki, Kd, Kfp, Kfi):

        self.model = model
        self.data = data

        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_site")

        self.ef_sum = np.zeros(3)
        self.e_sum = np.zeros(4)

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Kfp = Kfp
        self.Kfi = Kfi

    def compute(self, q_d, lambda_d):

        M = np.zeros((4, 4))
        mujoco.mj_fullM(self.model, M, self.data.qM)

        A = self.calc_A(self.data.qpos)
        M_inv = np.linalg.inv(M)
        P = np.identity(4) - A.T @ np.linalg.inv(A @ M_inv @ A.T) @ A @ M_inv

        e_dot = q_d[:, 1] - self.data.qvel
        e = q_d[:, 2] - self.data.qpos

        self.e_sum += e * self.model.opt.timestep

        tau_prime_motion = q_d[:, 0] + self.Kp @ e + self.Ki @ self.e_sum + self.Kd @ e_dot
        u_motion = P @ (M @ tau_prime_motion)

        J = np.zeros((3, 4))
        mujoco.mj_jacSite(self.model, self.data, J, None, self.ee_id)

        if self.data.ncon > 0:
            contact_rot = self.data.contact[0].frame.reshape(3, 3).T
            force = np.empty(6, dtype=np.double)
            mujoco.mj_contactForce(self.model, self.data, 0, force)
            force_base = contact_rot @ -force[:3]
        else:
            force_base = np.zeros(3)

        ee_orein = self.data.site("end_effector_site").xmat.reshape(3, 3)
        e_f = lambda_d[:, 0] - ee_orein.T @ force_base

        self.ef_sum += e_f * self.model.opt.timestep
        u_force = (np.identity(4) - P) @ J.T @ (lambda_d[:, 0] + self.Kfp @ e_f + self.Kfi @ self.ef_sum)

        u = u_motion + u_force + self.data.qfrc_bias

        return u

    def calc_A(self, q):
        q0, q1, q2, q3 = q
        return np.array(
            [
                [
                    0,
                    -0.045 * cos(q1) * cos(q2)
                    + (0.045 * cos(q1) * cos(q2) - 0.31 * sin(q1)) * cos(q3)
                    + (-0.31 * cos(q1) * cos(q2) - 0.045 * sin(q1)) * sin(q3)
                    - 0.55 * sin(q1),
                    -0.045 * cos(q3) * sin(q1) * sin(q2)
                    + 0.31 * sin(q1) * sin(q2) * sin(q3)
                    + 0.045 * sin(q1) * sin(q2),
                    (-0.31 * cos(q2) * sin(q1) + 0.045 * cos(q1)) * cos(q3)
                    - (0.045 * cos(q2) * sin(q1) + 0.31 * cos(q1)) * sin(q3),
                ]
            ]
        )
