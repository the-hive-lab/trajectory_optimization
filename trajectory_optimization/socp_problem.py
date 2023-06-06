# Standard library imports
from dataclasses import dataclass

# Third party imports
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


@dataclass
class Results:
    status: str
    objective: float
    s: np.ndarray
    nu: np.ndarray
    z: np.ndarray
    u: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    trav_time: float


class SocpProblem:
    def __init__(self, path, K):
        self.path = path

        nu = cp.Variable(K)  # DAE control input
        z = cp.Variable(K + 1, pos=True)  # DAE differential state
        u = cp.Variable((2, K))  # DAE algebraic state (original control input)

        a = cp.Variable(K + 1)  # Replaces sum of square roots of z
        b = cp.Variable(K)  # Replaces square root of z
        c = cp.Variable(K, pos=True)  # Replaces travel time component

        s_colloc = np.linspace(0, 1, num=K + 1, endpoint=True)  # Collocation points
        delta_s = np.diff(s_colloc)  # Distance between collocation points
        sqrt_z_sum = cp.sqrt(z[1:]) + cp.sqrt(z[:-1])  # Sum of square roots of z

        # Traversal time (CVXPY parameter so we can sweep over a range of them)
        traversal_time = cp.Parameter(nonneg=True)

        # Limits
        z_overline = 10.5  # Upper limit on z
        u_overline = 3  # Control upper limit
        u_underline = -u_overline  # Control lower limit

        constraints = list()  # Holds optimization constraints


        # Objective function constraints
        for k in range(K):
            norm_expr = cp.norm(cp.vstack([2 * u[0, k],
                                        2 * u[1, k],
                                        a[k + 1] + a[k] - b[k]]))
            constraints.append(norm_expr <= a[k + 1] + a[k] + b[k])

        for k in range(K + 1):
            norm_expr = cp.norm(cp.vstack([2 * a[k],
                                        z[k] - 1]))
            constraints.append(norm_expr <= z[k] + 1)

        objective = cp.Minimize(cp.sum(2 * cp.multiply(delta_s, b)))


        # Dynamics constraints
        for k in range(K - 1):
            delta_z_k = z[k + 1] - z[k]
            s_k = s_colloc[k]

            constraints.append(self.h(s_k) * nu[k] == self.f(s_k) * z[k] + self.G(s_k) @ u[:, k])
            constraints.append(delta_z_k == 2 * nu[k] * delta_s[k])


        # Traversal time constraints
        for k in range(K):
            norm_expr = cp.norm(cp.vstack([2,
                                        a[k + 1] + a[k] - c[k]]))
            constraints.append(norm_expr <= a[k + 1] + a[k] + c[k])

        constraints.append(
            cp.norm(cp.sum(cp.multiply(2 * delta_s, c))) <= traversal_time)


        # Terminal constraints (Matthew Kelly term)
        constraints.append(z[0] == 0)  # Initial path-velocity squared
        constraints.append(z[K] == 0)  # Final path-velocity squared

        # Path constraints (Matthew Kelly term)
        constraints.append(u[0, :] <= u_overline)
        constraints.append(u[1, :] <= u_overline)
        constraints.append(u[0, :] >= u_underline)
        constraints.append(u[1, :] >= u_underline)
        constraints.append(z >= 0)  # Positivity constraint
        constraints.append(z <= z_overline)


        # Create SOCP problem and solve
        self.problem = cp.Problem(objective, constraints)
        self.traversal_time = traversal_time
        self.s = s_colloc
        self.delta_s = delta_s
        self.nu = nu
        self.z = z
        self.u = u
        self.a = a
        self.b = b
        self.c = c

    def v(self, s):
        q_ds_1, q_ds_2, _ = self.path.tau_ds(s)

        return np.linalg.norm([q_ds_1, q_ds_2], axis=0)


    def h(self, s):
        return np.copy(self.path.tau_ds(s))


    def f(self, s):
        _, _, p_3 = self.path.tau(s)
        _, _, p_ds_3 = self.path.tau_ds(s)
        p_dds_1, p_dds_2, p_dds_3 = self.path.tau_dds(s)

        f_1 = -p_dds_1 - p_ds_3 * self.v(s) * np.sin(p_3)
        f_2 = -p_dds_2 + p_ds_3 * self.v(s) * np.cos(p_3)
        f_3 = -p_dds_3

        return np.array([f_1, f_2, f_3])


    def G(self, s):
        _, _, p_3 = self.path.tau(s)

        G_1 = np.array([np.cos(p_3), 0])
        G_2 = np.array([np.sin(p_3), 0])
        G_3 = np.array([0, 1])

        return np.vstack([G_1, G_2, G_3])

    def solve(self, trav_time, verbose: bool=False, **solver_opts: dict):
        try:
            self.traversal_time.value = trav_time
            result = self.problem.solve(verbose=verbose, **solver_opts)
        except Exception as e:
            # print(e)
            return None

        if (self.problem.status != 'optimal' and
            self.problem.status != 'optimal_inaccurate'):
            # print('Problem could not be solved')
            return None

        return Results(status=self.problem.status,
                       objective=result,
                       s=self.s,
                       nu=self.nu.value,
                       z=self.z.value,
                       u=self.u.value,
                       a=self.a.value,
                       b=self.b.value,
                       c=self.c.value,
                       trav_time=np.sum(2 * self.delta_s * self.c.value))
