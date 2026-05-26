import numpy as np
import numpy.typing as npt

def eig_KxK_diagblocks(K: int, n: int, matrix: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Compute the eigenvalues and eigenvectors of a block-diagonal matrix.
    
    Parameters
    ----
    K : int
        Number of block matrices per row/col (K^2 total blocks).
    n : int
        Number of rows/cols in each block matrix (each block matrix is n x n).
    matrix : ndarray
        The full matrix containing all block matrices (Kn x Kn).

    Returns
    ----
    eigs : ndarray
        Eigenvalues of the full matrix.
    vecs : ndarray
        Corresponding eigenvectors of the full matrix.
    """
    # hold eigenvalues
    eigs = np.zeros(K * n, dtype=complex)
    # hold eigenvectors
    vecs = np.zeros(shape=(K * n, K * n), dtype=complex)

    M_l = np.zeros(shape=(K, K), dtype=complex)

    # column counter
    col = 0

    for l in range(n):
        # fill M_l
        for i in range(K):
            for j in range(K):
                M_l[i, j] = matrix[i * n + l, j * n + l]
        # get eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eig(M_l)

        eigs[col:col + K] = eigvals
        
        for j in range(K):  # col
            for i in range(K):  # row
                vecs[i * n + l, col + j] = eigvecs[i, j]
        
        col += K
    return eigs, vecs