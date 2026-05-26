import numpy as np
import numpy.typing as npt
from typing import NamedTuple

from fast_eig.algorithm import eig_KxK_diagblocks

K_LARGEST = 10  # number of largest eigenvalues to compare

class VerificationResult(NamedTuple):
    num_eigenvalues_compared: int
    eigenvalues_match: bool
    max_residual: float
    mean_residual: float

def verify_results(matrix: npt.NDArray, alg_eigs: npt.NDArray, alg_vecs: npt.NDArray, 
                   reg_eigs: npt.NDArray, reg_vecs: npt.NDArray, atol: float=1e-8) -> VerificationResult:
    """
    Verify correctness of the block-diagonal eigendecomposition algorithm.

    Compare eigenvalues produced by the K x K block-diagonal algorithm against Numpy's eig
    function and evaluate the accuracy of the computed eigenvectors using residual norms.
    
    Parameters
    ----
    matrix : (Kn, Kn) ndarray
        Full block-structured matrix whose eigenpairs are being verified.
    alg_eigs : (Kn,) ndarray
        Eigenvalues of `matrix` computed using the block-diagonal algorithm.
        The ordering is arbitrary and corresponds column-wise to `alg_vecs`.
    alg_vecs : (Kn, Kn) ndarray
        Eigenvectors of `matrix` computed using the block-diagonal algorithm.
        Each column ``alg_vecs[:, i]`` is the eigenvector associated with ``alg_eigs[i]``.
    reg_eigs: (Kn,) ndarray
        Reference eigenvalues of `matrix` computed using NumPy's dense eigensolver.
        The ordering may differ from `alg_eigs`.
    reg_vecs: (Kn, Kn) ndarray
        Reference eigenvectors of `matrix` computed using NumPy's dense eigensolver.
        Each column ``reg_vecs[:, i]`` corresponds to ``reg_eigs[i]``.
    atol : float, optional
        Absolute tolerance used when comparing the sorted eigenvalues from the 
        block algorithm and the NumPy reference solution.
    
    Returns
    ---- 
    A named tuple with the following attributes:

    eigenvalues_match : bool 
        Indicates whether the sorted eigenvalues match NumPy's eig results.
    max_residual : float 
        Maximum eigenpair residual ``||matrix @ v - lambda * v||``.
    mean_residual : float 
        Mean eigenpair residual over all eigenvectors.
    """
    # compare k largest eigenvalues
    alg_dom = dominant_eigs(alg_eigs, K_LARGEST)
    reg_dom = dominant_eigs(reg_eigs, K_LARGEST)

    eig_check = np.allclose(alg_dom, reg_dom, atol=atol, rtol=0)

    # eigenvector residual check
    max_res = 0.0
    res_sum = 0.0
    for i in range(len(alg_eigs)):
        eigval = alg_eigs[i]
        eigvec = alg_vecs[:, i]
        res = np.linalg.norm(matrix @ eigvec - eigval * eigvec)
        res_sum += res
        max_res = max(max_res, res)

    mean_res = res_sum / len(alg_eigs)

    return VerificationResult(
        num_eigenvalues_compared=K_LARGEST,
        eigenvalues_match=eig_check,
        max_residual=max_res,
        mean_residual=mean_res
    )

def dominant_eigs(eigs: npt.NDArray, k: int) -> npt.NDArray:
    """
    Return the `k` eigenvalues with the largest magnitudes from an array of eigenvalues.
    
    Parameters
    ----
    eigs : ndarray
        Array of eigenvalues (real or complex) whose magnitudes will be used to 
        determine dominance.
    k : int
        Number of dominant eigenvalues to return. Must be less than or equal to
        `len(eigs)`.
    Returns
    ----
    ndarray
        Array of length `k` containing the eigenvalues with the largest magnitudes,
        in descending order of magnitude.
    """
    index_arr = np.argsort(np.abs(eigs))[::-1]
    return eigs[index_arr][:k]