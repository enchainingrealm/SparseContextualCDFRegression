import os

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from root import root


def main() -> None:
    os.makedirs(root("visualizations"), exist_ok=True)
    matplotlib.use("TkAgg")
    matplotlib.rcParams.update({
        "font.size": 14,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}"
    })

    visualize_n_fixed()
    visualize_lambda_ridge_fixed()
    visualize_s_fixed()
    visualize_d_fixed()
    visualize_lambda2_elasticnet_fixed()
    visualize_n_random()
    visualize_s_random()
    visualize_d_random()
    visualize_lambda2_elasticnet_random()


def visualize_n_fixed() -> None:
    """
    Produce PDF image plot of experimental results for Figure 1(a).
    """
    df = pd.read_csv(root("results/n_fixed.csv"), sep=",", usecols=lambda x: x != "run")
    g = df.groupby(by="sweep_value", sort=True).agg(["mean", "std"]).reset_index()

    x = g["sweep_value"]
    for i, (color, label) in enumerate([
        ("blue", "Ridge"), ("red", "Lasso"), ("green", r"Elastic Net ($\lambda_2 = 5 \times 10^{-3}$)")
    ], 1):
        mean = g[f"error_{i}"]["mean"]
        std = g[f"error_{i}"]["std"]
        plt.fill_between(x, mean - std, y2=mean + std, color=color, alpha=0.3)
        plt.plot(x, mean, color=color, label=label)

    y = 10 / np.sqrt(x)
    plt.plot(x, y, color="grey", linestyle="dashed")
    plt.text(1e5, 2.5e-2, r"slope = $-1/2$", color="grey", rotation=-21, ha="center", va="center")

    plt.xlabel(r"$n$")
    plt.ylabel(r"$\left\lVert \hat{\theta}_{\lambda(_1, \lambda_2)} - \theta_* \right\rVert$")
    plt.legend().set_zorder(0)
    plt.xlim([1e4, 1e6])
    plt.ylim([1e-2, 1e0])
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()

    plt.tight_layout()
    plt.savefig(root("visualizations/n_fixed.pdf"))
    plt.close()


def visualize_lambda_ridge_fixed() -> None:
    """
    Produce PDF image plot of experimental results for Figure 1(b).
    """
    df = pd.read_csv(root("results/lambda_ridge_fixed.csv"), sep=",", usecols=lambda x: x != "run")
    g = df.groupby(by="sweep_value", sort=True).agg(["mean", "std"]).reset_index()

    x = g["sweep_value"]
    for i, (color, label) in enumerate([
        ("orange", r"$\lambda = 10^{-3}$"), ("navy", r"$\lambda = 10^{-2}$"), ("purple", r"$\lambda = 10^{-1}$")
    ], 1):
        mean = g[f"error_{i}"]["mean"]
        std = g[f"error_{i}"]["std"]
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
    plt.savefig(root("visualizations/lambda_ridge_fixed.pdf"))
    plt.close()


def visualize_s_fixed() -> None:
    """
    Produce PDF image plot of experimental results for Figure 1(c).
    """
    df = pd.read_csv(root("results/s_fixed.csv"), sep=",", usecols=lambda x: x != "run")
    g = df.groupby(by="sweep_value", sort=True).agg(["mean", "std"]).reset_index()

    x = g["sweep_value"]
    for i, (color, label) in enumerate([
        ("blue", "Ridge"), ("red", "Lasso"), ("green", r"Elastic Net ($\lambda_2 = 10^{-3}$)")
    ], 1):
        mean = g[f"error_{i}"]["mean"]
        std = g[f"error_{i}"]["std"]
        plt.fill_between(x, mean - std, y2=mean + std, color=color, alpha=0.3)
        plt.plot(x, mean, color=color, label=label)

    plt.xlabel(r"$s$")
    plt.ylabel(r"$\left\lVert \hat{\theta}_{\lambda(_1, \lambda_2)} - \theta_* \right\rVert$")
    plt.legend()
    plt.xlim([1, 30])
    plt.grid()

    plt.tight_layout()
    plt.savefig(root("visualizations/s_fixed.pdf"))
    plt.close()


def visualize_d_fixed() -> None:
    """
    Produce PDF image plot of experimental results for Figure 1(d).
    """
    df = pd.read_csv(root("results/d_fixed.csv"), sep=",", usecols=lambda x: x != "run")
    g = df.groupby(by="sweep_value", sort=True).agg(["mean", "std"]).reset_index()

    x = g["sweep_value"]
    for i, (color, label) in enumerate([
        ("blue", "Ridge"), ("red", "Lasso"), ("green", r"Elastic Net ($\lambda_2 = 5 \times 10^{-3}$)")
    ], 1):
        mean = g[f"error_{i}"]["mean"]
        std = g[f"error_{i}"]["std"]
        plt.fill_between(x, mean - std, y2=mean + std, color=color, alpha=0.3)
        plt.plot(x, mean, color=color, label=label)

    plt.xlabel(r"$d$")
    plt.ylabel(r"$\left\lVert \hat{\theta}_{\lambda(_1, \lambda_2)} - \theta_* \right\rVert$")
    plt.legend()
    plt.xlim([10, 100])
    plt.grid()

    plt.tight_layout()
    plt.savefig(root("visualizations/d_fixed.pdf"))
    plt.close()


def visualize_lambda2_elasticnet_fixed() -> None:
    """
    Produce PDF image plot of experimental results for Figure 1(e).
    """
    df = pd.read_csv(root("results/lambda2_elasticnet_fixed.csv"), sep=",", usecols=lambda x: x != "run")
    g = df.groupby(by="sweep_value", sort=True).agg(["mean", "std"]).reset_index()

    x = g["sweep_value"]
    for i, (color, label) in enumerate([
        ("orange", r"$\lambda_2 = 10^{-4}$"), ("navy", r"$\lambda_2 = 10^{-3.5}$"),
        ("green", r"$\lambda_2 = 10^{-3}$"), ("purple", r"$\lambda_2 = 10^{-2.5}$")
    ], 1):
        mean = g[f"error_{i}"]["mean"]
        std = g[f"error_{i}"]["std"]
        plt.fill_between(x, mean - std, y2=mean + std, color=color, alpha=0.3)
        plt.plot(x, mean, color=color, label=label)

    plt.xlabel(r"$s$")
    plt.ylabel(r"$\left\lVert \hat{\theta}_{\lambda_1, \lambda_2} - \theta_* \right\rVert$")
    plt.legend()
    plt.xlim([1, 30])
    plt.grid()

    plt.tight_layout()
    plt.savefig(root("visualizations/lambda2_elasticnet_fixed.pdf"))
    plt.close()


def visualize_n_random() -> None:
    """
    Produce PDF image plot of experimental results for Figure 2(a).
    """
    df = pd.read_csv(root("results/n_random.csv"), sep=",", usecols=lambda x: x != "run")
    g = df.groupby(by="sweep_value", sort=True).agg(["mean", "std"]).reset_index()

    x = g["sweep_value"]
    for i, (color, label) in enumerate([
        ("blue", "Ridge"), ("red", "Lasso"), ("green", r"Elastic Net ($\lambda_2 = 10^{-2}$)")
    ], 1):
        mean = g[f"error_{i}"]["mean"]
        std = g[f"error_{i}"]["std"]
        plt.fill_between(x, mean - std, y2=mean + std, color=color, alpha=0.3)
        plt.plot(x, mean, color=color, label=label)

    y = 10 / np.sqrt(x)
    plt.plot(x, y, color="grey", linestyle="dashed")
    plt.text(1e5, 2.5e-2, r"slope = $-1/2$", color="grey", rotation=-21, ha="center", va="center")

    plt.xlabel(r"$n$")
    plt.ylabel(r"$\left\lVert \hat{\theta}_{\lambda(_1, \lambda_2)} - \theta_* \right\rVert$")
    plt.legend().set_zorder(0)
    plt.xlim([1e4, 1e6])
    plt.ylim([1e-2, 1e0])
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()

    plt.tight_layout()
    plt.savefig(root("visualizations/n_random.pdf"))
    plt.close()


def visualize_s_random() -> None:
    """
    Produce PDF image plot of experimental results for Figure 2(b).
    """
    df = pd.read_csv(root("results/s_random.csv"), sep=",", usecols=lambda x: x != "run")
    g = df.groupby(by="sweep_value", sort=True).agg(["mean", "std"]).reset_index()

    x = g["sweep_value"]
    for i, (color, label) in enumerate([
        ("blue", "Ridge"), ("red", "Lasso"), ("green", r"Elastic Net ($\lambda_2 = 10^{-2}$)")
    ], 1):
        mean = g[f"error_{i}"]["mean"]
        std = g[f"error_{i}"]["std"]
        plt.fill_between(x, mean - std, y2=mean + std, color=color, alpha=0.3)
        plt.plot(x, mean, color=color, label=label)

    plt.xlabel(r"$s$")
    plt.ylabel(r"$\left\lVert \hat{\theta}_{\lambda(_1, \lambda_2)} - \theta_* \right\rVert$")
    plt.legend()
    plt.xlim([1, 30])
    plt.grid()

    plt.tight_layout()
    plt.savefig(root("visualizations/s_random.pdf"))
    plt.close()


def visualize_d_random() -> None:
    """
    Produce PDF image plot of experimental results for Figure 2(c).
    """
    df = pd.read_csv(root("results/d_random.csv"), sep=",", usecols=lambda x: x != "run")
    g = df.groupby(by="sweep_value", sort=True).agg(["mean", "std"]).reset_index()

    x = g["sweep_value"]
    for i, (color, label) in enumerate([
        ("blue", "Ridge"), ("red", "Lasso"), ("green", r"Elastic Net ($\lambda_2 = 10^{-2}$)")
    ], 1):
        mean = g[f"error_{i}"]["mean"]
        std = g[f"error_{i}"]["std"]
        plt.fill_between(x, mean - std, y2=mean + std, color=color, alpha=0.3)
        plt.plot(x, mean, color=color, label=label)

    plt.xlabel(r"$d$")
    plt.ylabel(r"$\left\lVert \hat{\theta}_{\lambda(_1, \lambda_2)} - \theta_* \right\rVert$")
    plt.legend()
    plt.xlim([10, 100])
    plt.grid()

    plt.tight_layout()
    plt.savefig(root("visualizations/d_random.pdf"))
    plt.close()


def visualize_lambda2_elasticnet_random() -> None:
    """
    Produce PDF image plot of experimental results for Figure 2(d).
    """
    df = pd.read_csv(root("results/lambda2_elasticnet_random.csv"), sep=",", usecols=lambda x: x != "run")
    g = df.groupby(by="sweep_value", sort=True).agg(["mean", "std"]).reset_index()

    x = g["sweep_value"]
    for i, (color, label) in enumerate([
        ("orange", r"$\lambda_2 = 10^{-2.5}$"), ("green", r"$\lambda_2 = 10^{-2}$"),
        ("navy", r"$\lambda_2 = 10^{-1.5}$"), ("purple", r"$\lambda_2 = 10^{-1}$")
    ], 1):
        mean = g[f"error_{i}"]["mean"]
        std = g[f"error_{i}"]["std"]
        plt.fill_between(x, mean - std, y2=mean + std, color=color, alpha=0.3)
        plt.plot(x, mean, color=color, label=label)

    plt.xlabel(r"$d$")
    plt.ylabel(r"$\left\lVert \hat{\theta}_{\lambda_1, \lambda_2} - \theta_* \right\rVert$")
    plt.legend()
    plt.xlim([10, 100])
    plt.grid()

    plt.tight_layout()
    plt.savefig(root("visualizations/lambda2_elasticnet_random.pdf"))
    plt.close()


if __name__ == "__main__":
    main()
