import csv
import os
import random
from math import sqrt, log
from typing import List, Dict, Any, Iterable, Callable

import numpy as np
from sklearn.linear_model import Lasso, Ridge
# noinspection PyProtectedMember
from sklearn.linear_model._base import LinearModel

from root import root

# Construct scikit-learn model object given basis dimension d, sample size n, sparsity s, and failure probability delta.
ModelConstructor = Callable[[int, int, int, float], LinearModel]


def main() -> None:
    os.makedirs(root("results"), exist_ok=True)

    # Run simulations and produce CSV results file for Figure 1(a).
    run_experiment("n_trend", {
        "d": 10, "s": 5, "delta": 0.001, "create_models": [optimal_ridge, optimal_lasso]
    }, "n", np.logspace(4, 6, num=100, dtype=int))

    # Run simulations and produce CSV results file for Figure 1(b).
    run_experiment("n_lambda_ridge_trend", {
        "d": 10, "s": 5, "delta": 0.001, "create_models": [
            lambda d, n, s, delta, lam=lam: Ridge(alpha=n * lam, fit_intercept=False, max_iter=1000000, random_state=1731)
            for lam in np.logspace(-3, -1, num=3)
        ]
    }, "n", np.logspace(4, 6, num=100, dtype=int))

    # Run simulations and produce CSV results file for Figure 1(c).
    run_experiment("s_trend", {
        "d": 100, "n": 100000, "delta": 0.001, "create_models": [optimal_ridge, optimal_lasso]
    }, "s", range(1, 31))

    # Run simulations and produce CSV results file for Figure 1(d).
    run_experiment("d_trend", {
        "n": 100000, "s": 10, "delta": 0.001, "create_models": [optimal_ridge, optimal_lasso]
    }, "d", range(10, 101))


def run_experiment(name: str, cfg: Dict[str, Any], sweep_key: str, sweep_values: Iterable[Any]) -> None:
    """
    Simulate the data generation and CDF regression processes according to the given hyperparameters.
    :param name: filename for CSV output file
    :param cfg: names and values of hyperparameters fixed throughout the experiment
    :param sweep_key: hyperparameter to vary throughout the experiment
    :param sweep_values: values to use for the varying hyperparameter
    """
    random.seed(1731)   # reset random seed before each experiment so experiments can be independently reproduced
    np.random.seed(1731)

    # Open file in append mode to protect against accidentally overwriting previous results
    with open(root(f"results/{name}.csv"), "a") as csv_file:
        csv_writer = csv.writer(csv_file)

        n_models = len(cfg["create_models"])
        csv_writer.writerow(["sweep_value", "run"] + [f"error_{k}" for k in range(1, n_models + 1)])
        csv_file.flush()

        for sweep_value in sweep_values:
            cfg[sweep_key] = sweep_value
            n_runs = cfg.pop("n_runs", 30)   # https://stats.stackexchange.com/a/2542
            for run in range(1, n_runs + 1):
                errors = run_trial(**cfg)
                csv_writer.writerow([sweep_value, run] + errors)
                csv_file.flush()


def run_trial(d: int, n: int, s: int, delta: float, create_models: List[ModelConstructor]) -> List[float]:
    """
    Construct context vectors, sample response variables, perform CDF regression, and compute estimation errors based on
    the given hyperparameters as defined in Theorem 1.
    :param d: dimension of the CDF basis
    :param n: sample size
    :param s: sparsity (l0-norm) of the true parameter
    :param delta: failure probability
    :param create_models: constructors for models to compute estimation errors of
    """
    theta_true = random_sparse_pmf(d, s)
    x_list = construct_x_j_list(n, d)
    y_list = [sample_y_j(theta_true, x_j) for x_j in x_list]

    A = np.stack(x_list)
    b = np.array(y_list)

    errors = []
    for create_model in create_models:
        model = create_model(d, n, s, delta)
        model.fit(A, b)
        errors.append(np.linalg.norm(model.coef_ - theta_true, ord=2))
    return errors


def random_sparse_pmf(d: int, s: int) -> np.ndarray:
    """
    Uniformly sample a random PMF with sparsity (l0-norm) s from the (d-1)-dimensional probability simplex.
    """
    theta = np.zeros(d)
    support = np.random.choice(d, size=s, replace=False)
    theta[support] = np.random.dirichlet([1] * s)
    return theta


def construct_x_j_list(n: int, d: int) -> List[np.ndarray]:
    """
    Construct context vectors according to the procedure described in Section 8.
    :param n: sample size
    :param d: dimension of the CDF basis
    """
    x_list = []
    M_prev = None
    outer_sum_prev = np.zeros((d, d))
    denominator = None

    for j in range(1, n + 1):
        if j <= d:
            x_val = 1 / 2
        else:
            mu_min = np.linalg.eigvalsh(M_prev)[0]
            while mu_min / denominator > 0.5:
                denominator *= 2
            x_val = mu_min / denominator

        x_j = np.full(d, 1 - x_val)
        x_j[(j - 1) % d] -= x_val
        x_list.append(x_j)

        outer = np.outer(1 - x_j, 1 - x_j)
        if j >= d:
            M_prev = outer + outer_sum_prev / n
        if j == d:
            denominator = np.linalg.eigvalsh(M_prev)[0] / 2
        outer_sum_prev += outer
    return x_list


def sample_y_j(theta_true: np.ndarray, x_j: np.ndarray) -> int:
    """
    Sample a response variable according to the Bernoulli CDF basis described in Section 8.
    :param theta_true: the true parameter of convex coefficients
    :param x_j: the context vector
    """
    return np.random.binomial(1, np.dot(theta_true, x_j))


# noinspection PyUnusedLocal
def optimal_ridge(d: int, n: int, s: int, delta: float) -> Ridge:
    """
    Construct a ridge regression object (using the loss formulation in Equation 1) with regularization hyperparameter
    lambda = 4 sqrt(2/n log(2d/delta)).
    """
    lam = 4 * sqrt(2 / n * log(2 * d / delta))
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
    return Ridge(alpha=n * lam, fit_intercept=False, max_iter=1000000, random_state=1731)


# noinspection PyUnusedLocal
def optimal_lasso(d: int, n: int, s: int, delta: float) -> Lasso:
    """
    Construct a lasso regression object (using the loss formulation in Equation 1) with regularization hyperparameter
    lambda = 4 sqrt(2/n log(2d/delta)), as stated in Theorem 1.
    """
    lam = 4 * sqrt(2 / n * log(2 * d / delta))
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    return Lasso(alpha=lam / 2, fit_intercept=False, max_iter=1000000, random_state=1731)


if __name__ == "__main__":
    main()
