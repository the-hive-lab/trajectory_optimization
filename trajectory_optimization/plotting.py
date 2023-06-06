import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def plot_nu(axis, colloc_points, s_knots, zeta_knots, bounds=(), block=False):
    axis.plot(s_knots, zeta_knots, 'o')

    zeta_func = sp.interpolate.interp1d(s_knots, zeta_knots, kind='previous',
                                        fill_value='extrapolate')
    s_samples = np.linspace(0, 1, num=1000)
    axis.plot(s_samples, zeta_func(s_samples))
    for point in colloc_points:
        axis.axvline(point, linestyle='dashed', color='grey', linewidth=0.5)

    for bound in bounds:
        axis.axhline(bound, linestyle='solid', color='black', linewidth=0.5)

    axis.set_title('Interpolated $\zeta = \ddot{s}$', loc='left', fontsize='small')
    axis.set_xlabel('Path position ($s$)')
    axis.set_ylabel('$\hat{\zeta}$')
    plt.show(block=block)


def plot_z(axis, colloc_points, s_knots, z_knots, bounds=(), block=False):
    axis.plot(s_knots, z_knots, 'o')

    z_func = sp.interpolate.interp1d(s_knots, z_knots, kind='linear')
    s_samples = np.linspace(0, 1, num=1000)
    axis.plot(s_samples, z_func(s_samples))
    for point in colloc_points:
        axis.axvline(point, linestyle='dashed', color='grey', linewidth=0.5)

    for bound in bounds:
        axis.axhline(bound, linestyle='solid', color='black', linewidth=0.5)

    axis.set_title('Interpolated $z = \dot{s}^2$', loc='left', fontsize='small')
    axis.set_xlabel('Path position ($s$)')
    axis.set_ylabel('$\hat{z}$')
    plt.show(block=block)


def plot_u1(axis, colloc_points, s_knots, u1_knots, bounds=(), block=False):
    axis.plot(s_knots, u1_knots, 'o')

    u1_func = sp.interpolate.interp1d(s_knots, u1_knots, kind='cubic',
                                      fill_value='extrapolate')
    s_samples = np.linspace(0, 1, num=1000)
    axis.plot(s_samples, u1_func(s_samples))
    for point in colloc_points:
        axis.axvline(point, linestyle='dashed', color='grey', linewidth=0.5)

    for bound in bounds:
        axis.axhline(bound, linestyle='solid', color='black', linewidth=0.5)

    axis.set_title('Interpolated $u_1$', loc='left', fontsize='small')
    axis.set_xlabel('Path position ($s$)')
    axis.set_ylabel('$\hat{u}_1$')
    plt.show(block=block)


def plot_u2(axis, colloc_points, s_knots, u2_knots, bounds=(), block=False):
    axis.plot(s_knots, u2_knots, 'o')

    u2_func = sp.interpolate.interp1d(s_knots, u2_knots, kind='cubic',
                                      fill_value='extrapolate')
    s_samples = np.linspace(0, 1, num=1000)
    axis.plot(s_samples, u2_func(s_samples))
    for point in colloc_points:
        axis.axvline(point, linestyle='dashed', color='grey', linewidth=0.5)

    for bound in bounds:
        axis.axhline(bound, linestyle='solid', color='black', linewidth=0.5)

    axis.set_title('Interpolated $u_2$', loc='left', fontsize='small')
    axis.set_xlabel('Path position ($s$)')
    axis.set_ylabel('$\hat{u}_2$')
    plt.show(block=block)


def plot_dec_var_interp(axis, knot_pts, var_vals, interp_func, var_name, bounds=()):
    if knot_pts.shape == var_vals.shape:
        axis.plot(knot_pts, var_vals, 'o', label='knot pts.')
    else:
        axis.plot(knot_pts[:-1], var_vals, 'o', label='knot pts.')

    s_vals = np.linspace(0, 1, num=1000, endpoint=True)
    axis.plot(s_vals, interp_func(s_vals), label='interp.')
    for pt in knot_pts:
        axis.axvline(pt, linestyle='dashed', color='gray', linewidth=0.5)

    for bound in bounds:
        axis.axhline(bound, linestyle='solid', color='black', linewidth=0.5)

    axis.grid(True, linestyle='dotted')
    axis.set_title(f'Interpolated {var_name}', loc='right', fontsize='small')
    axis.set_xlabel('Path-position ($s$)')
    axis.legend(loc='best', fontsize='x-small')


def plot_dec_var(axis, knot_pts, var_vals, var_name, bounds=()):
    axis.plot(knot_pts, var_vals, 'o', label='knot pts.')

    s_vals = np.linspace(0, 1, num=1000, endpoint=True)
    for pt in knot_pts:
        axis.axvline(pt, linestyle='dashed', color='gray', linewidth=0.5)

    for bound in bounds:
        axis.axhline(bound, linestyle='solid', color='black', linewidth=0.5)

    axis.grid(True, linestyle='dotted')
    axis.set_title(var_name, loc='right', fontsize='small')
    axis.set_xlabel('Path-position ($s$)')
    axis.set_ylabel(var_name)
    axis.legend(loc='best', fontsize='x-small')
