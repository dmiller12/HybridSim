from math import cos, sin
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from scipy.optimize import root


class PlanarWAM:
    def __init__(self):
        pass

    def get_J(self, q):
        q1, q3 = q

        J = np.array([
            [0.3835*cos(q1 + q3) + 0.55*cos(q1) + 0.045*sin(q1 + q3) - 0.045*sin(q1), 0.3835*cos(q1 + q3) + 0.045*sin(q1 + q3)],
            [0.045*cos(q1 + q3) - 0.045*cos(q1) - 0.3835*sin(q1 + q3) - 0.55*sin(q1), 0.045*cos(q1 + q3) - 0.3835*sin(q1 + q3)]
        ])
        return J

    def get_Jdot(self, q, qdot):
        q1, q3 = q
        qdot1, qdot3 = qdot
        Jdot = np.array([
            [ 0.045*qdot1*cos(q1 + q3) + 0.045*qdot3*cos(q1 + q3) - 0.045*qdot1*cos(q1) - 0.3835*qdot1*sin(q1 + q3) - 0.3835*qdot3*sin(q1 + q3) - 0.55*qdot1*sin(q1), 0.045*qdot1*cos(q1 + q3) + 0.045*qdot3*cos(q1 + q3) - 0.3835*qdot1*sin(q1 + q3) - 0.3835*qdot3*sin(q1 + q3)],
            [-0.3835*qdot1*cos(q1 + q3) - 0.3835*qdot3*cos(q1 + q3) - 0.55*qdot1*cos(q1) - 0.045*qdot1*sin(q1 + q3) - 0.045*qdot3*sin(q1 + q3) + 0.045*qdot1*sin(q1), -0.3835*qdot1*cos(q1 + q3) - 0.3835*qdot3*cos(q1 + q3) - 0.045*qdot1*sin(q1 + q3) - 0.045*qdot3*sin(q1 + q3)]
        ])
        return Jdot

    def fk(self, q, des):
        q1, q3 = q
        x = -0.045*cos(q1 + q3) + 0.045*cos(q1) + 0.3835*sin(q1 + q3) + 0.55*sin(q1) - des[0]
        z = 0.3835*cos(q1 + q3) + 0.55*cos(q1) + 0.045*sin(q1 + q3) - 0.045*sin(q1) + 0.346 - des[2]

        return np.array([x, z])[:, 0]


class WAM4:
    def __init__(self):

        self.DH = [
            {"a": 0, "alpha": -jnp.pi / 2, "d": 0.346},
            {"a": 0, "alpha": jnp.pi / 2, "d": 0},
            {"a": 0.045, "alpha": -jnp.pi / 2, "d": 0.55},
            {"a": -0.045, "alpha": jnp.pi / 2, "d": 0},
            {"a": 0, "alpha": 0, "d": 0.3835},
        ]
        
    @partial(jax.jit, static_argnums=(0,))
    def fk(self, q):
        T = jnp.identity(4)
        q = jnp.append(q, jnp.array([0]))
        for i, row in enumerate(self.DH):
            T = T @ self.calc_DH_transformation(**row, q=q[i])
        return T

    def calc_DH_transformation(self, a, alpha, d, q):
        return jnp.array([
            [jnp.cos(q), -jnp.sin(q)*jnp.cos(alpha), jnp.sin(q)* jnp.sin(alpha), a * jnp.cos(q)],
            [jnp.sin(q), jnp.cos(q)*jnp.cos(alpha), -jnp.cos(q)*jnp.sin(alpha), a * jnp.sin(q)],
            [0, jnp.sin(alpha), jnp.cos(alpha), d],
            [0, 0, 0, 1]
        ])


    @partial(jax.jit, static_argnums=(0,))
    def get_J(self, q):
        T = jnp.identity(4)
        z_list = []
        p_list = []
        q = jnp.append(q, jnp.array([0]))
        for i, row in enumerate(self.DH):
            z_list.append(T[:3, 2])
            p_list.append(T[:3, -1])
            T = T @ self.calc_DH_transformation(**row, q=q[i])

        ee_pos = T[:3, -1]
        J_rows = []
        for i in range(len(z_list)-1):
            row1 = jnp.cross(z_list[i], ee_pos - p_list[i])
            row2 = z_list[i]
            J_rows.append(jnp.concatenate([row1, row2]))

        J = jnp.stack(J_rows, axis=1)
        return J

    @partial(jax.jit, static_argnums=(0,))
    def get_Jdot(self, q, qdot):
        partialJ = jax.jacobian(self.get_J)
        Jdot = jnp.einsum('ijk,k->ij', partialJ(q), qdot)
        return Jdot

    def ik(self, start, desired):
        res = root(
            lambda x: self.fk(np.array([0, x[0], 0, x[1]]))[:3, -1] - desired,
            [start[1], start[3]],
            jac=lambda x: self.get_J(np.array([0, x[0], 0, x[1]]))[:3, [1, 3]],
            method='lm',
            options={'xtol': 1e-9}
        )
        return res.x
