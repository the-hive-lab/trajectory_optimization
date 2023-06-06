import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def unicycle_second_order(state, control):
    _, _, theta, v, omega = state
    accel_lin, accel_ang = control

    d_x = v * np.cos(theta)
    d_y = v * np.sin(theta)
    d_theta = omega
    d_v = accel_lin
    d_omega = accel_ang

    return np.array([d_x, d_y, d_theta, d_v, d_omega])


def integrate_rk4(model, state, control, period):
    k1 = model(state, control)
    k2 = model(state + period * k1 / 2, control)
    k3 = model(state + period * k2 / 2, control)
    k4 = model(state + period * k3, control)

    return state + period / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_system(model, state_init, controls, periods):
    state_current = state_init
    trajectory = list()

    for index in range(controls.shape[1]-1):
        control = controls[:, index]
        state_next = integrate_rk4(model, state_current, control, periods[index])
        trajectory.append(state_current)
        state_current = state_next

    trajectory.append(state_current)

    return trajectory


def simulate_system2(model, state_init, controls, periods):
    state_current = state_init
    trajectory = list()

    for index in range(controls.shape[1]):
        control = controls[:, index]
        state_next = integrate_rk4(model, state_current, control, periods[index])
        trajectory.append(state_current)
        state_current = state_next

    trajectory.append(state_current)

    return trajectory


def plot_system(trajectory, controls, reference_path):
    s = np.linspace(0, 1, num=50)
    x_ref, y_ref, theta_ref = reference_path(s)

    x_ref_samp = np.squeeze(x_ref[:-1:5])
    y_ref_samp = np.squeeze(y_ref[:-1:5])
    theta_ref_samp = np.squeeze(theta_ref[:-1:5])

    fig = plt.figure(constrained_layout=True)
    fig.suptitle('Simulation Results')
    fig.patch.set_facecolor('lightgrey')
    gs = mpl.gridspec.GridSpec(4, 2, figure=fig)
    s_traj = np.linspace(0, 1, num=trajectory.shape[1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(s_traj, trajectory[0, :], 'o-')
    ax0.plot(s, np.squeeze(x_ref))
    ax0.grid(True, linestyle='dotted')
    ax0.set_title('$x$', loc='left', fontsize='small')

    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax1.plot(s_traj, trajectory[1, :], 'o-')
    ax1.plot(s, np.squeeze(y_ref))
    ax1.grid(True, linestyle='dotted')
    ax1.set_title('$y$', loc='left', fontsize='small')

    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax2.plot(s_traj, trajectory[2, :], 'o-')
    ax2.plot(s, np.squeeze(theta_ref))
    ax2.grid(True, linestyle='dotted')
    ax2.set_title('$\\theta$', loc='left', fontsize='small')

    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
    ax3.plot(s_traj, trajectory[3, :], 'o-')
    ax3.grid(True, linestyle='dotted')
    ax3.set_title('$v$', loc='left', fontsize='small')

    ax4 = fig.add_subplot(gs[0, 1])
    ax4.plot(s_traj, trajectory[4, :], 'o-')
    ax4.grid(True, linestyle='dotted')
    ax4.set_title('$\\omega$', loc='left', fontsize='small')

    ax5 = fig.add_subplot(gs[1, 1], sharex=ax4)
    ax5.step(s_traj[:-1], controls[0, :], 'o-', where='post')
    ax5.grid(True, linestyle='dotted')
    ax5.set_title('$u_a$', loc='left', fontsize='small')

    ax6 = fig.add_subplot(gs[2, 1], sharex=ax4)
    ax6.step(s_traj[:-1], controls[1, :], 'o-', where='post')
    ax6.grid(True, linestyle='dotted')
    ax6.set_title('$u_\\alpha$', loc='left', fontsize='small')

    plt.show(block=False)

    fig, ax = plt.subplots()
    fig.suptitle('Simulation Results')
    fig.patch.set_facecolor('lightgrey')
    ax.plot(np.squeeze(x_ref), np.squeeze(y_ref),
             color='tab:orange', label='reference path')
    ax.quiver(x_ref_samp, y_ref_samp, np.cos(theta_ref_samp),
               np.sin(theta_ref_samp), color='tab:orange')

    ax.plot(trajectory[0, :], trajectory[1, :], 'o-', color='tab:blue',
             label='system path')
    ax.quiver(trajectory[0, :], trajectory[1, :], np.cos(trajectory[2, :]),
               np.sin(trajectory[2, :]), color='tab:blue')

    ax.set_title('path trace', loc='left', fontsize='small')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, linestyle='dotted')
    plt.show(block=False)


def plot_system_states(fig, trajectory, controls, reference_path):
    s = np.linspace(0, 1, num=50)
    x_ref, y_ref, theta_ref = reference_path(s)

    fig.suptitle('Simulation Results')
    fig.patch.set_facecolor('lightgrey')
    gs = mpl.gridspec.GridSpec(4, 2, figure=fig)
    s_traj = np.linspace(0, 1, num=trajectory.shape[1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(s_traj, trajectory[0, :], 'o-')
    ax0.plot(s, np.squeeze(x_ref))
    ax0.grid(True, linestyle='dotted')
    ax0.set_title('$x$', loc='left', fontsize='small')

    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax1.plot(s_traj, trajectory[1, :], 'o-')
    ax1.plot(s, np.squeeze(y_ref))
    ax1.grid(True, linestyle='dotted')
    ax1.set_title('$y$', loc='left', fontsize='small')

    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax2.plot(s_traj, trajectory[2, :], 'o-')
    ax2.plot(s, np.squeeze(theta_ref))
    ax2.grid(True, linestyle='dotted')
    ax2.set_title('$\\theta$', loc='left', fontsize='small')

    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
    ax3.plot(s_traj, trajectory[3, :], 'o-')
    ax3.grid(True, linestyle='dotted')
    ax3.set_title('$v$', loc='left', fontsize='small')

    ax4 = fig.add_subplot(gs[0, 1])
    ax4.plot(s_traj, trajectory[4, :], 'o-')
    ax4.grid(True, linestyle='dotted')
    ax4.set_title('$\\omega$', loc='left', fontsize='small')

    ax5 = fig.add_subplot(gs[1, 1], sharex=ax4)
    ax5.step(s_traj[:-1], controls[0, :], 'o-', where='post')
    ax5.grid(True, linestyle='dotted')
    ax5.set_title('$u_a$', loc='left', fontsize='small')

    ax6 = fig.add_subplot(gs[2, 1], sharex=ax4)
    ax6.step(s_traj[:-1], controls[1, :], 'o-', where='post')
    ax6.grid(True, linestyle='dotted')
    ax6.set_title('$u_\\alpha$', loc='left', fontsize='small')


def plot_system_trace(fig, ax, trajectory, controls, reference_path):
    s = np.linspace(0, 1, num=50)
    x_ref, y_ref, theta_ref = reference_path(s)

    x_ref_samp = np.squeeze(x_ref[:-1:5])
    y_ref_samp = np.squeeze(y_ref[:-1:5])
    theta_ref_samp = np.squeeze(theta_ref[:-1:5])

    fig, ax = plt.subplots()
    fig.suptitle('Simulation Results')
    fig.patch.set_facecolor('lightgrey')
    ax.plot(np.squeeze(x_ref), np.squeeze(y_ref),
             color='tab:orange', label='reference path')
    ax.quiver(x_ref_samp, y_ref_samp, np.cos(theta_ref_samp),
               np.sin(theta_ref_samp), color='tab:orange')

    ax.plot(trajectory[0, :], trajectory[1, :], 'o-', color='tab:blue',
             label='system path')
    ax.quiver(trajectory[0, :], trajectory[1, :], np.cos(trajectory[2, :]),
               np.sin(trajectory[2, :]), color='tab:blue')

    ax.set_title('path trace', loc='left', fontsize='small')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, linestyle='dotted')


def plot_state(axis, t_vals,  state_vals, state_name, block=False):
    axis.plot(t_vals, state_vals, label=state_name)

    axis.grid(True, linestyle='dotted')
    axis.xlabel('Time (t)')
    axis.ylabel(state_name)


def plot_trace(axis, x_refs, y_refs, theta_refs, x_vals, y_vals, theta_vals, block=False):
    axis.plot(x_refs, y_refs, color='tab:orange', label='ref. path')
    axis.quiver(x_refs, y_refs, np.cos(theta_refs), np.sin(theta_refs), color='tab:orange')

    axis.plot(x_vals, y_vals, color='tab:blue', label='traced path')
    axis.quiver(x_vals, y_vals, np.cos(theta_vals), np.sin(theta_vals), color='tab:blue')

    axis.set_title('path trace', loc='right', fontsize='small')
    axis.legend(fontsize='x-small')
    axis.grid(True, linestyle='dotted')
    axis.set_aspect('equal')
