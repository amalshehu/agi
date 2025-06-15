# arc_prize_solvers.py

import numpy as np

# Import your ensemble solver entry point
from advanced_arc_solver import solve_with_advanced_methods

def solve(challenge: dict, debug: bool = False) -> np.ndarray:
    """
    Solve an ARC task via the advanced ensemble methods only.
    """
    # This will raise if the advanced solver isn’t present,
    # so make sure advanced_arc_solver.py is on your PYTHONPATH.
    return solve_with_advanced_methods(challenge, debug=debug)
