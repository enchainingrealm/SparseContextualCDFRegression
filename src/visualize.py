import os

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

from root import root


def main() -> None:
    os.makedirs(root("visualizations"), exist_ok=True)
    matplotlib.use("TkAgg")
    matplotlib.rcParams.update({
        "figure.dpi": 227,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}"
    })

    visualize_n_trend()
    visualize_n_lambda_ridge_trend()
    visualize_s_trend()
    visualize_d_trend()


def visualize_n_trend() -> None:
    """
    Produce PNG image plot of experimental results for Figure 1(a).
    """
    df = pd.read_csv(root("results/n_trend.csv"), sep=",", usecols=lambda x: x != "run")
    g = df.groupby(by="sweep_value", sort=True).agg(["mean", "std"]).reset_index()

    x = g["sweep_value"]
    for metric, color, label in [("error_1", "blue", "Ridge"), ("error_2", "red", "Lasso")]:
        mean = g[metric]["mean"]
        std = g[metric]["std"]
        plt.fill_between(x, mean - std, y2=mean + std, color=color, alpha=0.3)
        plt.plot(x, mean, color=color, label=label)

    plt.xlabel(r"$n$")
    plt.ylabel(r"$\left\lVert \hat{\theta}_\lambda - \theta_* \right\rVert$")
    plt.legend()
    plt.xlim([1e4, 1e6])
    plt.ylim([1e-2, 1e0])
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()

    plt.tight_layout()
    plt.savefig(root("visualizations/n_trend.png"))
    plt.close()


def visualize_n_lambda_ridge_trend() -> None:
    """
    Produce PNG image plot of experimental results for Figure 1(b).
    """
    df = pd.read_csv(root("results/n_lambda_ridge_trend.csv"), sep=",", usecols=lambda x: x != "run")
    g = df.groupby(by="sweep_value", sort=True).agg(["mean", "std"]).reset_index()

    x = g["sweep_value"]
    for metric, color, label in [
        ("error_1", "orange", r"$\lambda = 10^{-3}$"),
        ("error_2", "green", r"$\lambda = 10^{-2}$"),
        ("error_3", "purple", r"$\lambda = 10^{-1}$")
    ]:
        mean = g[metric]["mean"]
        std = g[metric]["std"]
        plt.fill_between(x, mean - std, y2=mean + std, color=color, alpha=0.3)
        plt.plot(x, mean, color=color, label=label)

    plt.xlabel(r"$n$")
    plt.ylabel(r"$\left\lVert \hat{\theta}_\lambda - \theta_* \right\rVert$")
    plt.legend()
    plt.xlim([1e4, 1e6])
    plt.ylim([1e-2, 1e0])
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()

    plt.tight_layout()
    plt.savefig(root("visualizations/n_lambda_ridge_trend.png"))
    plt.close()


def visualize_s_trend() -> None:
    """
    Produce PNG image plot of experimental results for Figure 1(c).
    """
    df = pd.read_csv(root("results/s_trend.csv"), sep=",", usecols=lambda x: x != "run")
    g = df.groupby(by="sweep_value", sort=True).agg(["mean", "std"]).reset_index()

    x = g["sweep_value"]
    for metric, color, label in [("error_1", "blue", "Ridge"), ("error_2", "red", "Lasso")]:
        mean = g[metric]["mean"]
        std = g[metric]["std"]
        plt.fill_between(x, mean - std, y2=mean + std, color=color, alpha=0.3)
        plt.plot(x, mean, color=color, label=label)

    plt.xlabel(r"$s$")
    plt.ylabel(r"$\left\lVert \hat{\theta}_\lambda - \theta_* \right\rVert$")
    plt.legend()
    plt.xlim([1, 30])
    plt.grid()

    plt.tight_layout()
    plt.savefig(root("visualizations/s_trend.png"))
    plt.close()


def visualize_d_trend() -> None:
    """
    Produce PNG image plot of experimental results for Figure 1(d).
    """
    df = pd.read_csv(root("results/d_trend.csv"), sep=",", usecols=lambda x: x != "run")
    g = df.groupby(by="sweep_value", sort=True).agg(["mean", "std"]).reset_index()

    x = g["sweep_value"]
    for metric, color, label in [("error_1", "blue", "Ridge"), ("error_2", "red", "Lasso")]:
        mean = g[metric]["mean"]
        std = g[metric]["std"]
        plt.fill_between(x, mean - std, y2=mean + std, color=color, alpha=0.3)
        plt.plot(x, mean, color=color, label=label)

    plt.xlabel(r"$d$")
    plt.ylabel(r"$\left\lVert \hat{\theta}_\lambda - \theta_* \right\rVert$")
    plt.legend()
    plt.xlim([10, 100])
    plt.grid()

    plt.tight_layout()
    plt.savefig(root("visualizations/d_trend.png"))
    plt.close()


if __name__ == "__main__":
    main()
