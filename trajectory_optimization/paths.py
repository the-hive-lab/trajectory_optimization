# Third party imports
import casadi
import matplotlib.pyplot as plt
import numpy as np


class Se2Path:
    def __init__(self, name: str, s: casadi.SX.sym, x_expr, y_expr, theta_expr) -> None:
        self.name = name

        self.path = casadi.Function('path', [s], [x_expr, y_expr, theta_expr])

        x, y, theta = self.path(s)
        self.path_ds = casadi.Function('path_ds', [s], [casadi.jacobian(x, s),
                                                        casadi.jacobian(y, s),
                                                        casadi.jacobian(theta, s)])

        x_ds, y_ds, theta_ds = self.path_ds(s)  # velocity
        self.path_dds = casadi.Function('path_dds', [s], [casadi.jacobian(x_ds, s),
                                                          casadi.jacobian(y_ds, s),
                                                          casadi.jacobian(theta_ds, s)])

    @property
    def tau_sx(self):
        return self.path

    @property
    def tau_ds_sx(self):
        return self.path_ds

    @property
    def tau_dds_sx(self):
        return self.path_dds

    def tau(self, s):
        return np.asarray(self.path(s)).squeeze()

    def tau_ds(self, s):
        return np.asarray(self.path_ds(s)).squeeze()

    def tau_dds(self, s):
        return np.asarray(self.path_dds(s)).squeeze()

    def x(self, s):
        return np.asarray(self.path(s)).squeeze()[0]

    def x_ds(self, s):
        return np.asarray(self.path_ds(s)).squeeze()[0]

    def x_dds(self, s):
        return np.asarray(self.path_dds(s)).squeeze()[0]

    def y(self, s):
        return np.asarray(self.path(s)).squeeze()[1]

    def y_ds(self, s):
        return np.asarray(self.path_ds(s)).squeeze()[1]

    def y_dds(self, s):
        return np.asarray(self.path_dds(s)).squeeze()[1]

    def theta(self, s):
        return np.asarray(self.path(s)).squeeze()[2]

    def theta_ds(self, s):
        return np.asarray(self.path_ds(s)).squeeze()[2]

    def theta_dds(self, s):
        return np.asarray(self.path_dds(s)).squeeze()[2]


class StraightLinePath(Se2Path):
    def __init__(self, length: float=5) -> None:
        s = casadi.SX.sym('s')

        x_expr = 1
        y_expr = length * s
        theta_expr = casadi.atan2(casadi.jacobian(y_expr, s),
                                  casadi.jacobian(x_expr, s))

        super().__init__('StraightLine', s, x_expr, y_expr, theta_expr)


class LeftTurnPath(Se2Path):
    def __init__(self, radius: float=5) -> None:
        s = casadi.SX.sym('s')

        x_expr = radius * casadi.cos(casadi.pi / 2 * s) - radius
        y_expr = radius * casadi.sin(casadi.pi / 2 * s)
        theta_expr = casadi.atan2(casadi.jacobian(y_expr, s),
                                  casadi.jacobian(x_expr, s))

        super().__init__('LeftTurn', s, x_expr, y_expr, theta_expr)


class LeftTurnLongerPath(Se2Path):
    def __init__(self, radius: float=5) -> None:
        s = casadi.SX.sym('s')

        x_expr = radius * casadi.cos(3 * casadi.pi / 2 * s) - radius
        y_expr = radius * casadi.sin(3 * casadi.pi / 2 * s)
        theta_expr = casadi.atan2(casadi.jacobian(y_expr, s),
                                  casadi.jacobian(x_expr, s))

        super().__init__('LeftTurnLonger', s, x_expr, y_expr, theta_expr)


class LeftTurnSharpPath(Se2Path):
    def __init__(self) -> None:
        s = casadi.SX.sym('s')

        x_expr = casadi.cos(s) - 1
        y_expr = 2 * casadi.sin(casadi.pi * s)
        theta_expr = casadi.atan2(casadi.jacobian(y_expr, s),
                                  casadi.jacobian(x_expr, s))

        super().__init__('LeftTurnSharp', s, x_expr, y_expr, theta_expr)


class RightTurnPath(Se2Path):
    def __init__(self, radius: float=5) -> None:
        s = casadi.SX.sym('s')

        x_expr = - radius * casadi.cos(casadi.pi / 2 * s)
        y_expr = radius * casadi.sin(casadi.pi / 2 * s)
        theta_expr = casadi.atan2(casadi.jacobian(y_expr, s),
                                  casadi.jacobian(x_expr, s))

        super().__init__('RightTurn', s, x_expr, y_expr, theta_expr)


class RightTurnQuadraticPath(Se2Path):
    def __init__(self) -> None:
        s = casadi.SX.sym('s')

        x_expr = 10 * s
        y_expr = -25 * s ** 2 + 30 * s
        theta_expr = casadi.atan2(casadi.jacobian(y_expr, s),
                                  casadi.jacobian(x_expr, s))

        super().__init__('RightTurnQuadratic', s, x_expr, y_expr, theta_expr)


class RightTurnSharpPath(Se2Path):
    def __init__(self) -> None:
        s = casadi.SX.sym('s')

        x_expr = -casadi.cos(s)
        y_expr = 2 * casadi.sin(casadi.pi * s)
        theta_expr = casadi.atan2(casadi.jacobian(y_expr, s),
                                  casadi.jacobian(x_expr, s))

        super().__init__('RightTurnSharp', s, x_expr, y_expr, theta_expr)


def plot_components(path):
    fig, axes = plt.subplots(3, sharex=True)

    s_values = np.linspace(0, 1, num=1000, endpoint=True)
    axes[0].plot(s_values, path.x(s_values))
    axes[0].set_title('$x(s)$', fontsize='small', loc='right')

    axes[1].plot(s_values, path.y(s_values))
    axes[1].set_title('$y(s)$', fontsize='small', loc='right')

    axes[2].plot(s_values, path.theta(s_values))
    axes[2].set_title('$\\theta(s)$', fontsize='small', loc='right')

    axes[2].set_xlabel('path-position ($s$)')

    fig.suptitle(f'{path.name} - Components')
    plt.tight_layout()


def plot_components_ds(path):
    fig, axes = plt.subplots(3, sharex=True)

    s_values = np.linspace(0, 1, num=1000, endpoint=True)
    axes[0].plot(s_values, path.x_ds(s_values))
    axes[0].set_title("$x'(s)$", fontsize='small', loc='right')

    axes[1].plot(s_values, path.y_ds(s_values))
    axes[1].set_title("$y'(s)$", fontsize='small', loc='right')

    axes[2].plot(s_values, path.theta_ds(s_values))
    axes[2].set_title("$\\theta'(s)$", fontsize='small', loc='right')

    axes[2].set_xlabel('path-position ($s$)')

    fig.suptitle(f'{path.name} - Components ($d \\tau / d s$)')
    plt.tight_layout()


def plot_components_dds(path):
    fig, axes = plt.subplots(3, sharex=True)

    s_values = np.linspace(0, 1, num=1000, endpoint=True)
    axes[0].plot(s_values, path.x_dds(s_values))
    axes[0].set_title("$x''(s)$", fontsize='small', loc='right')

    axes[1].plot(s_values, path.y_dds(s_values))
    axes[1].set_title("$y''(s)$", fontsize='small', loc='right')

    axes[2].plot(s_values, path.theta_dds(s_values))
    axes[2].set_title("$\\theta''(s)$", fontsize='small', loc='right')

    axes[2].set_xlabel('path-position ($s$)')

    fig.suptitle(f'{path.name} - Components ($d^2 \\tau / d s^2)$')
    plt.tight_layout()


def plot_trace(path):
    fig, axis = plt.subplots()

    s_values = np.linspace(0, 1, num=1000, endpoint=True)
    axis.plot(path.x(s_values), path.y(s_values))
    axis.plot(path.x(0), path.y(0), 'o', color='green', label='start')
    axis.plot(path.x(1), path.y(1), 'o', color='red', label='end')

    axis.legend()
    axis.set_ylabel('$y$-axis ($y$)')
    axis.set_xlabel('$x$-axis ($x$)')
    axis.set_aspect('equal')

    fig.suptitle(f'{path.name} - Trace')
