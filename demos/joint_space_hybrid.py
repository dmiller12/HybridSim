import numpy as np
import mujoco
from pathlib import Path
import mujoco.viewer
import time
import pandas as pd
from scipy.optimize import root

from controllers.hybrid_controller import JointSpaceHybridController
from robots.wam import PlanarWAM


def linear_cart_traj(xstart, xend, t, prev_pos, robot):
    cart = xstart + (t / 10) * (xend - xstart)
    cartdot = (xend - xstart) * (1 / 10)

    res = root(lambda x: robot.fk(x, cart), prev_pos, jac=robot.get_J, tol=1e-5, options={"maxfev": 1000})

    qdot = np.linalg.solve(robot.get_J(res.x), cartdot[[0, 2]])
    qdotdot = np.linalg.solve(robot.get_J(res.x), -robot.get_Jdot(res.x, qdot[:, 0]) @ qdot)
    return res.x, qdot, qdotdot, cart


if __name__ == "__main__":

    xml_path = Path("assets/scenes/scene.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    MAX_TIME = 10  # seconds

    data = mujoco.MjData(model)

    model.vis.scale.contactwidth = 0
    model.vis.scale.contactheight = 0
    model.vis.scale.forcewidth = 0.02
    model.vis.map.force = 5

    # motion gains
    Kp = 1000 * np.diag([1, 1, 1, 1])
    Ki = 0 * np.diag([1, 1, 1, 1])
    Kd = 50 * np.diag([1, 1, 1, 1])
    # force gains
    Kfp = 1 * np.diag([1, 1, 1])
    Kfi = 10 * np.diag([1, 1, 1])
    controller = JointSpaceHybridController(model, data, Kp=Kp, Ki=Ki, Kd=Kd, Kfp=Kfp, Kfi=Kfi)

    initial_x = np.array([[0.45, 0, 0.346]]).T
    final_x = np.array([[0.65, 0, 0.346]]).T

    robot = PlanarWAM()

    res = root(
        lambda x: robot.fk(x, initial_x + np.array([[0, 0, 0.01]]).T),
        [0.78, 0.78],
        jac=robot.get_J,
        tol=1e-5,
        options={"maxfev": 1000},
    )

    data.qpos = np.array([0, res.x[0], 0, res.x[1]])
    
    freq = int(1 / model.opt.timestep)
    logDf = pd.DataFrame(index=range(freq * MAX_TIME))

    with mujoco.viewer.launch_passive(model, data) as viewer:

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        start_time = time.time()
        prev_q = data.qpos[[1, 3]]
        time_step = 0

        while viewer.is_running() and data.time < MAX_TIME:
            step_start = time.time()

            mujoco.mj_step1(model, data)

            q, qdot, qdotdot, cart = linear_cart_traj(initial_x, final_x, data.time, prev_q, robot)
            prev_q = q

            q_d = np.empty((4, 3), dtype=np.double)
            q_d[:, 0] = [0, qdotdot[0, 0], 0, qdotdot[1, 0]]
            q_d[:, 1] = [0, qdot[0, 0], 0, qdot[1, 0]]
            q_d[:, 2] = [0, q[0], 0, q[1]]

            force_d_base = np.array([[0, 0, 10]]).T
            force_rot = data.site("end_effector_site").xmat.reshape(3, 3)
            lambda_d = force_rot.T @ force_d_base

            data.ctrl = controller.compute(q_d, lambda_d)

            # logging
            logDf.loc[time_step, "time"] = data.time
            logDf.loc[time_step, ["x", "y", "z"]] = data.site("end_effector_site").xpos
            logDf.loc[time_step, ["q_0", "q_1", "q_2", "q_3"]] = data.qpos
            
            logDf.loc[time_step, ["x_d", "y_d", "z_d"]] = cart[:, 0]
        
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
