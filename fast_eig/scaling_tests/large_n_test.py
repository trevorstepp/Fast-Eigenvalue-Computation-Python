import pandas as pd
import numpy as np

from fast_eig.benchmark import median_time
from fast_eig.build import build_block_matrix
from fast_eig.algorithm import eig_KxK_diagblocks

N_VALUES = [100, 250, 500, 750, 1000, 1500, 2000, 3000, 4500, 6000, 10000]
K_VAL = 3

def measure_runtime_large_n() -> list[float]:
    """
    Docstring for measure_runtime.
    
    Parameters
    ----

    Returns
    ----
    """
    # lists to hold runtimes for block method and NumPy's eig method and max residuals for block method
    block_time = []

    for n in N_VALUES:
        print(f"\nRunning n = {n}")
        # create the block-diagonal matrix
        M = build_block_matrix(K=K_VAL, n=n, seed=0)

        # runtimes
        block_t = median_time(lambda: eig_KxK_diagblocks(K_VAL, n, M))
        block_time.append(block_t)
    
    return block_time

def main():
    """
    Docstring for main.
    """
    block_time = measure_runtime_large_n()
    # save times in CSV file for future reference
    df = pd.DataFrame({
        "K": K_VAL,
        "n": N_VALUES,
        "block_time": block_time,
    })
    df.to_csv("python_large_n_timings.csv", index=False)

if __name__ == '__main__':
    main()