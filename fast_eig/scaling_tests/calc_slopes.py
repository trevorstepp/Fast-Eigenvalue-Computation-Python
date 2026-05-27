import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from numpy.polynomial.polynomial import polyfit

def calc_slopes(data_sources: dict) -> None:
    """
    Plots.

    Parameters
    ----
    data_sources : dict
        A dictionary containing language runtimes where the keys are the
        languages and the values are .csv files containing runtimes.
    """
    num_plots = len(data_sources)
    fig, axes = plt.subplots(1, num_plots, figsize=(3.5 * num_plots, 5))

    # must handle case of only one plot
    if num_plots == 1:
        axes = [axes]

    # one log-log subplot for each language's times
    for ax, (language, df_name) in zip(axes, data_sources.items()):
        df = pd.read_csv(df_name)
        #df_large_n = df[df["n"] >= 500]
        df_large_n = df[df["n"] >= 500].sort_values("n")

        n = df_large_n["n"]
        block = df_large_n["block_time"]
        log_n = np.log(n)
        log_block = np.log(block)

        slope, intercept = np.polyfit(log_n, log_block, 1)
        print(f"{language} block slope: {slope:.2f}")

        block_fit = np.exp(intercept) * (n ** slope)
        # we expect the fit to have a slope around one (O(n))
        # c sets the vertical position of the reference line (O(n))
        c = block.iloc[0] / n.iloc[0]
        theoretical = c * n

        ax.scatter(n, block, label="data")
        ax.plot(n, block_fit, linestyle="--", label=f"fit ({slope:.2f})")
        ax.plot(n, theoretical, linestyle=":", label="O(n)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(language)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()
    
    fig.suptitle("Block-Diagonal Method Runtimes for Large $n$")
    fig.supxlabel("Problem size $n$")
    fig.supylabel("Runtime (seconds)")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    base_dir = Path(__file__).parent.parent.parent.parent
    scaling_dir = Path(__file__).parent
    python_csv = scaling_dir / "python_large_n_timings.csv"
    julia_csv = base_dir / "Fast-Eigenvalue-Computation-Julia" / "julia_large_n_timings.csv"
    matlab_csv = base_dir / "Fast-Eigenvalue-Computation-MATLAB" / "matlab_large_n_timings.csv"

    data = {
        "Python": python_csv,
        "Julia": julia_csv,
        "MATLAB": matlab_csv
    }
    calc_slopes(data)