# Standard library imports
from pathlib import Path
import sys
import time

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

# Adds this file's grandparent directory to the Python module search
# path so it can find the package. This allows user to run the script
# without having to install the package.
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Local application imports
from trajectory_optimization import paths, simulation, socp_problem


def _main():
    test_paths = {
        'left_turn': paths.LeftTurnPath,
        'right_turn': paths.RightTurnPath,
        'straight': paths.StraightLinePath
    }

    traversal_times = list()
    solve_times = list()
    for path_name, path_cls in test_paths.items():
        for path_param in range(5, 15 + 1):
            path = path_cls(path_param)
            problem = socp_problem.SocpProblem(path, K=20)

            solver_opts = {
                'max_iters': 300,
                'feastol': 1e-6  # Default: 1e-8
            }

            for time_goal in range(5, 25 + 1):
                time_start = time.process_time_ns()
                results = problem.solve(time_goal, verbose=False, **solver_opts)
                time_stop = time.process_time_ns()
                solve_time = time_stop - time_start

                if results is None:
                    traversal_times.append((path_name, path_param, time_goal,
                                            np.nan, np.nan))
                    solve_times.append((path_name, path_param, time_goal, np.nan))
                    continue

                rel_err = abs(1 - results.trav_time / time_goal)
                traversal_times.append((path_name, path_param, time_goal,
                                        results.trav_time, rel_err))

                solve_times.append((path_name, path_param, time_goal, solve_time))

    traversal_times_df = pd.DataFrame(traversal_times,
                                      columns=['Path', 'Param', 'Time Goal',
                                               'Time', 'Rel Err'])
    print(traversal_times_df)
    traversal_times_df.to_csv('output.csv')

    print(f"mean error (rel err): {traversal_times_df['Rel Err'].mean():>.4e}")
    print(f"std dev. (rel err):   {traversal_times_df['Rel Err'].std():>.4e}")

    solve_times_df = pd.DataFrame(solve_times, columns=['Path', 'Param',
                                                        'Time Goal',
                                                        'Solve Time'])
    print(solve_times_df)
    solve_times_df.to_csv('solver_times.csv')

    print(f"mean solver time (ms): {solve_times_df['Solve Time'].mean() / 1e6:>.4}")
    print(f"std dev. (ms):         {solve_times_df['Solve Time'].std() / 1e6:>.4}")

    return 0


if __name__ == '__main__':
    sys.exit(_main())
