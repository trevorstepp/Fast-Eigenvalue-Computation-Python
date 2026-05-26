import numpy as np
import numpy.typing as npt

def build_block_matrix(K: int, n: int, seed: int | None = None) -> npt.NDArray:
    """
    Build a (Kn x Kn) block matrix with K x K blocks, where each block is n x n and diagonal,
    sharing a common diagonal structure across blocks.
    
    Parameters
    ----
    K : int
        Number of block rows/columns.
    n : int
        Size of each diagonal block.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    ----
    M : ndarray
        Block matrix.
    """
    if seed is not None:
        np.random.seed(seed)
    
    M = np.zeros(shape=(K * n, K * n), dtype=complex)

    # loop n times, once for each diagonal index in the block matrices
    for l in range(n):
        # Construct the K×K matrix B_l = M_l (algorithm.py),
        # formed by collecting the l-th diagonal entries of each n×n block.
        B_l = np.random.randn(K, K)
        for i in range(K):
            for j in range(K):
                M[i * n + l, j * n + l] = B_l[i, j]

    return M

if __name__ == '__main__':
    print(f"{build_block_matrix(K=2, n=2, seed=0)}")