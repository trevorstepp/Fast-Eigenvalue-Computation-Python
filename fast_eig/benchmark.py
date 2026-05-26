import timeit
import numpy as np

from fast_eig.algorithm import eig_KxK_diagblocks

NUM_SAMPLES = 5

def median_time(func) -> float:
    """
    Compute the median execution time of a function using NUM_SAMPLES timing samples.

    Parameters
    ----
    func : callable
        Function to be timed. Must take no arguments (wrap with lambda if needed).

    Returns
    ----
    time : float
        Median execution time per function call in seconds.
    """
    timer = timeit.Timer(func)
    # automatically determine number of runs per sample
    loops, _ = timer.autorange()
    runs = timer.repeat(repeat=NUM_SAMPLES, number=loops)

    return np.median(runs) / loops