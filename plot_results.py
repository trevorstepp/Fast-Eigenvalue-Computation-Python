import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_runtime_comparison() -> None:
    """
    Docstring for plot_runtime_comparison.
    """
    # python csv
    python_df = pd.read_csv("python_timings.csv")
    K = python_df['K'][0]

    curr_dir = Path(__file__).parent
    base_dir = curr_dir.parent

    # julia csv
    julia_csv = base_dir / "Fast-Eigenvalue-Computation-Julia" / "julia_timings.csv"
    julia_df = pd.read_csv(julia_csv)

    # matlab csv
    matlab_csv = base_dir / "Fast-Eigenvalue-Computation-MATLAB" / "matlab_timings.csv"
    matlab_df = pd.read_csv(matlab_csv)

    plt.figure(figsize=(8,6))

    # python
    plt.semilogy(python_df.n, python_df.block_time, 'o-', label='Python Block')
    plt.semilogy(python_df.n, python_df.dense_time, 'o--', label='Python NumPy eig')

    # julia
    plt.semilogy(julia_df.n, julia_df.block_time, 's-', label='Julia Block')
    plt.semilogy(julia_df.n, julia_df.dense_time, 's--', label='Julia eigen')

    # matlab
    plt.semilogy(matlab_df.n, matlab_df.block_time, '^-', label='MATLAB Block')
    plt.semilogy(matlab_df.n, matlab_df.dense_time, '^--', label='MATLAB eig')

    plt.xlabel('discretization size (n)')
    plt.ylabel('run time (s)')
    plt.title(f'All eigenpairs runtime (K={K})')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"K={K}.png")
    plt.show()

if __name__ == '__main__':
    plot_runtime_comparison()