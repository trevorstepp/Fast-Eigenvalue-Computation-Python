import pandas as pd
import numpy as np

from fast_eig.benchmark import median_time
from fast_eig.verify import verify_results
from fast_eig.build import build_block_matrix
from fast_eig.algorithm import eig_KxK_diagblocks

N_VALUES = [100, 250, 500, 750, 1000, 1500, 2000]  # adjust depending on patience
K_VAL = 3

def measure_runtime_and_verify() -> tuple[list[float], list[float], list[float]]:
    """
    Docstring for measure_runtime.
    
    Parameters
    ----

    Returns
    ----
    """
    # lists to hold runtimes for block method and NumPy's eig method and max residuals for block method
    block_time = []
    eig_time = []
    block_max_residuals = []

    for n in N_VALUES:
        print(f"\nRunning n = {n}")
        # create the block-diagonal matrix
        M = build_block_matrix(K=K_VAL, n=n, seed=0)

        # runtimes
        block_t = median_time(lambda: eig_KxK_diagblocks(K_VAL, n, M))
        eig_t = median_time(lambda: np.linalg.eig(M))

        block_time.append(block_t)
        eig_time.append(eig_t)

        # need to call functions again to get eigenvalues and eigenvectors for correctness check
        block_eigs, block_vecs = eig_KxK_diagblocks(K_VAL, n, M)
        numpy_eigs, numpy_vecs = np.linalg.eig(M)

        correct_check = verify_results(matrix=M, alg_eigs=block_eigs, alg_vecs=block_vecs, reg_eigs=numpy_eigs, 
                                       reg_vecs=numpy_vecs)
        print(f"Eigenvalues match (top {correct_check.num_eigenvalues_compared}): {correct_check.eigenvalues_match}")
        print(f"Maximum residual: {correct_check.max_residual:.2e}")
        print(f"Mean residual: {correct_check.mean_residual:.2e}")

        block_max_residuals.append(correct_check.max_residual)
    
    return block_time, eig_time, block_max_residuals

def store_data(csv_name: str) -> None:
    """
    Store runtimes for both methods and the maximum residuals for the block-diagonal algorithm in a .csv file.

    Parameters
    ----
    csv_name : str
        Name of the .csv file to store the data in.
    """
    block_time, eig_time, block_residuals = measure_runtime_and_verify()
    # save times in CSV file for future reference
    df = pd.DataFrame({
        "K": K_VAL,
        "n": N_VALUES,
        "block_time": block_time,
        "dense_time": eig_time,
        "max_residual": block_residuals
    })
    df.to_csv(csv_name, index=False)

if __name__ == '__main__':
    store_data("python_timings.csv")