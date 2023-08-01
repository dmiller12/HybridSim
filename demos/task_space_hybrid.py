import numpy as np
import mujoco
from pathlib import Path
import mujoco.viewer
import time
import pandas as pd
from scipy.spatial.transform import Rotation as R

from controllers.hybrid_controller import TaskSpaceHybridController
from robots.wam import WAM4


def linear_cart_traj(xstart, xend, t, max_time):
    xpos_start, xorein_start = xstart[:3, -1], R.from_matrix(xstart[:3, :3])
    xpos_end, xorein_end = xend[:3, -1], R.from_matrix(xend[:3, :3])

    xorein_start = xorein_start.as_euler('zyz')
    xorein_end = xorein_end.as_euler('zyz')

    start = np.concatenate([xorein_start, xpos_start])
    end = np.concatenate([xorein_end, xpos_end])
    cart = start + (t / max_time) * (end - start)
    cartdot = (end - start) * (1 / max_time)
    cartdotdot = np.zeros(6)

    return cartdotdot, cartdot, cart


if __name__ == "__main__":

    xml_path = Path("assets/scenes/scene.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    MAX_TIME = 10  # seconds

    data = mujoco.MjData(model)

    model.vis.scale.contactwidth = 0
    model.vis.scale.contactheight = 0
    model.vis.scale.forcewidth = 0.02
    model.vis.map.force = 4

    robot = WAM4()

    # motion gains, orientation (x, y, z), linear (x, y)
    Kp = 10 * np.diag([1, 1, 1, 100, 100])
    Ki = 0 * np.diag([1, 1, 1, 1, 1])
    Kd = 1 * np.diag([0.1, 0.1, 0.1, 5, 5])
    # force gains
    Kfp = 1 * np.diag([1])
    Kfi = 0.1 * np.diag([1])

    controller = TaskSpaceHybridController(model, data, robot, Kp=Kp, Ki=Ki, Kd=Kd, Kfp=Kfp, Kfi=Kfi)

    initial_x = np.array([0.45, 0, 0.346])
    final_x = np.array([0.65, 0, 0.346])

    res_initial = robot.ik([0, 0.38426975110262895, 0, 2.3233065269466247], initial_x)
    res_final = robot.ik([0, 0.38426975110262895, 0, 2.3233065269466247], final_x)

    data.qpos = np.array([0, res_initial[0], 0, res_initial[1]])

    pose_initial = robot.fk(data.qpos)
    pose_final = robot.fk(np.array([0, res_final[0], 0, res_final[1]]))

    freq = 1 / model.opt.timestep
    logDf = pd.DataFrame(index=range(int(freq * MAX_TIME)))

    with mujoco.viewer.launch_passive(model, data) as viewer:

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        start_time = time.time()
        prev_q = data.qpos
        time_step = 0

        while viewer.is_running() and data.time < MAX_TIME:
            step_start = time.time()

            mujoco.mj_step1(model, data)

            cartdotdot, cartdot, cart = linear_cart_traj(
                pose_initial,
                pose_final,
                data.time,
                MAX_TIME
            )

            x_d = np.zeros((5, 3), dtype=np.double)
            x_d[:, 0] = cartdotdot[0:5]
            x_d[:, 1] = cartdot[0:5]
            x_d[:, 2] = cart[0:5]

            force_d_base = np.array([[0, 0, 10]]).T
            force_rot = data.site("end_effector_site").xmat.reshape(3, 3)
            lambda_d = force_rot.T @ force_d_base

            data.ctrl = controller.compute(x_d, lambda_d[-1, 0])

            # logging
            logDf.loc[time_step, "time"] = data.time
            logDf.loc[time_step, ["x", "y", "z"]] = data.site("end_effector_site").xpos
            logDf.loc[time_step, ["q_0", "q_1", "q_2", "q_3"]] = data.qpos

            logDf.loc[time_step, ["x_d", "y_d", "z_d"]] = cart[3:]

            if data.ncon > 0:
                contact_rot = data.contact[0].frame.reshape(3, 3).T
                force = np.empty(6, dtype=np.double)
                mujoco.mj_contactForce(model, data, 0, force)
                force_base = force_rot.T @ contact_rot @ -force[:3]
            else:
                force_base = np.zeros(3)

            logDf.loc[time_step, ['f_x', 'f_y', 'f_z']] = force_base
            logDf.loc[time_step, ['f_x_d', 'f_y_d', 'f_z_d']] = lambda_d[:, 0]

            mujoco.mj_step2(model, data)

            logDf.loc[time_step, ["torque_0", "torque_1", "torque_2", "torque_3"]] = data.actuator_force

            viewer.sync()

            # roughly sync sim time to real time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            time_step += 1

    logDf.to_csv("data/res.csv", index=False)
