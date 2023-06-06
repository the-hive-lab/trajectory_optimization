# Standard library imports
from datetime import datetime
from pathlib import Path
import sys

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

# Local application imports

# Adds this file's grandparent directory to the Python module search
# path so it can find the package. This allows user to run the script
# without having to install the package.
sys.path.append(str(Path(__file__).resolve().parents[1]))
from trajectory_optimization import paths, plotting, simulation, socp_problem


EXIT_SUCCESS = 0


def run_simulation(output_basedir, path, z_interp, u_lin_interp, u_ang_interp):
    s_sim = np.linspace(0, 1, num=50, endpoint=True)
    delta_s_sim = np.diff(s_sim)
    z_sim = z_interp(s_sim)
    time_periods = 2 * delta_s_sim / (np.sqrt(z_sim[1:]) + np.sqrt(z_sim[:-1]))
    u_sim = np.stack([u_lin_interp(s_sim)[:-1], u_ang_interp(s_sim)[:-1]])

    model = simulation.unicycle_second_order
    x_0, y_0, theta_0 = np.asarray(path.tau(0)).squeeze()
    x_ds_0, y_ds_0, theta_ds_0 = np.asarray(path.tau_ds(0)).squeeze() * np.sqrt(z_interp(0))
    v_0 = np.linalg.norm([x_ds_0, y_ds_0])
    omega_0 = theta_ds_0
    state_init = np.array([x_0, y_0, theta_0, v_0, omega_0])
    trajectory = simulation.simulate_system2(model, state_init, u_sim, periods=time_periods)
    trajectory = np.array(trajectory).transpose()

    fig = plt.figure(constrained_layout=True)
    simulation.plot_system_states(fig, trajectory, u_sim, path.tau)
    plt.savefig(output_basedir / 'sim_output-states.png')

    fig, ax = plt.subplots()
    simulation.plot_system_trace(fig, ax, trajectory, u_sim, path.tau)
    plt.savefig(output_basedir / 'sim_output-trace.png')

    _, axis = plt.subplots()
    x_refs, y_refs, theta_refs = path.tau(s_sim)
    x_refs = np.asarray(x_refs).squeeze()
    y_refs = np.asarray(y_refs).squeeze()
    theta_refs = np.asarray(theta_refs).squeeze()
    simulation.plot_trace(axis, x_refs, y_refs, theta_refs, trajectory[0, :], trajectory[1, :], trajectory[2, :])
    plt.savefig(output_basedir / 'sim_trace.png')

    ref_data = np.vstack([s_sim, x_refs, y_refs, theta_refs]).T
    ctrl_data = np.hstack([u_sim, np.asarray([[np.nan], [np.nan]])])
    results_df = pd.DataFrame(np.hstack([ref_data, trajectory.T, ctrl_data.T]),
                              columns=['S', 'Ref X', 'Ref Y', 'Ref Theta',
                                       'X', 'Y', 'Theta', 'V', 'Omega',
                                       'Ctrl Lin', 'Ctrl Ang'])
    results_df.to_csv(output_basedir / 'sim_output-trajectory.csv',
                      index=False)


def _main():
    path = paths.LeftTurnPath(radius=5)
    traversal_time = 7
    problem = socp_problem.SocpProblem(path, K=20)

    solver_opts = {
        'max_iters': 300,
        'feastol': 1e-6  # Default: 1e-8
    }

    results = problem.solve(traversal_time, verbose=False, **solver_opts)
    if results is None:
        print('Problem could not be solved.')
        exit()

    # Interpolate functions
    nu_interp = sp.interpolate.interp1d(results.s[:-1], results.nu, kind='linear', fill_value='extrapolate')
    z_interp = sp.interpolate.interp1d(results.s, results.z, kind='cubic')
    u_lin_interp = sp.interpolate.interp1d(results.s[:-1], results.u[0, :], kind='cubic', fill_value='extrapolate')
    u_ang_interp = sp.interpolate.interp1d(results.s[:-1], results.u[1, :], kind='cubic', fill_value='extrapolate')

    output_basedir = Path(f'run-{datetime.now()}')
    output_basedir.mkdir(parents=True)

    # Plot results
    fig, axes = plt.subplots(2, 2)
    fig.suptitle('Optimization Results')
    plotting.plot_dec_var_interp(axes[0, 0], results.s, results.nu, nu_interp, '$\\nu$')
    plotting.plot_dec_var_interp(axes[1, 0], results.s, results.z, z_interp, '$z$')
    plotting.plot_dec_var_interp(axes[0, 1], results.s, results.u[0, :], u_lin_interp, '$u_{\mathrm{lin}}$', bounds=(-2, 2))
    plotting.plot_dec_var_interp(axes[1, 1], results.s, results.u[1, :], u_ang_interp, '$u_{\mathrm{ang}}$', bounds=(-2, 2))
    plt.tight_layout()
    plt.savefig(output_basedir / 'opt_output-dae_vars.png')

    fig, axes = plt.subplots(3)
    fig.suptitle('Optimization Results')
    plotting.plot_dec_var(axes[0], results.s, results.a, '$a$')
    plotting.plot_dec_var(axes[1], results.s[:-1], results.b, '$b$')
    plotting.plot_dec_var(axes[2], results.s[:-1], results.c, '$c$')
    plt.tight_layout()
    plt.savefig(output_basedir / 'opt_output-socp_vars.png')

    run_simulation(output_basedir, path, z_interp, u_lin_interp, u_ang_interp)

    return EXIT_SUCCESS


if __name__ == '__main__':
    sys.exit(_main())
