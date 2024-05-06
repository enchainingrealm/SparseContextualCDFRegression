import csv
import functools
import os
import random
from math import sqrt, log
from typing import List, Dict, Any, Iterable, Callable

import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
# noinspection PyProtectedMember
from sklearn.linear_model._base import LinearModel

from root import root

# Construct scikit-learn model object given basis dimension d, sample size n, sparsity s, and failure probability delta.
ModelConstructor = Callable[[int, int, int, float], LinearModel]


def main() -> None:
    os.makedirs(root("results"), exist_ok=True)

    # Run simulations and produce CSV results file for Figure 1(a).
    run_experiment("n_fixed", {
        "d": 10, "s": 5, "delta": 0.001, "design": "fixed",
        "create_models": [optimal_ridge, optimal_lasso, elastic_net(5e-3)]
    }, "n", np.logspace(4, 6, num=100, dtype=int))

    # Run simulations and produce CSV results file for Figure 1(b).
    run_experiment("lambda_ridge_fixed", {
        "d": 10, "s": 5, "delta": None, "design": "fixed",
        "create_models": [ridge(lam) for lam in np.logspace(-3, -1, num=3)]
    }, "n", np.logspace(4, 6, num=100, dtype=int))

    # Run simulations and produce CSV results file for Figure 1(c).
    run_experiment("s_fixed", {
        "d": 100, "n": 100000, "delta": 0.001, "design": "fixed",
        "create_models": [optimal_ridge, optimal_lasso, elastic_net(1e-3)]
    }, "s", range(1, 31))

    # Run simulations and produce CSV results file for Figure 1(d).
    run_experiment("lambda2_elasticnet_fixed", {
        "d": 100, "n": 100000, "delta": 0.001, "design": "fixed",
        "create_models": [elastic_net(lam_2) for lam_2 in np.logspace(-4.0, -2.5, num=4)]
    }, "s", range(1, 31))

    # Run simulations and produce CSV results file for Figure 1(e).
    run_experiment("d_fixed", {
        "n": 100000, "s": 10, "delta": 0.001, "design": "fixed",
        "create_models": [optimal_ridge, optimal_lasso, elastic_net(5e-3)]
    }, "d", range(10, 101))

    # Run simulations and produce CSV results file for Figure 2(a).
    run_experiment("n_random", {
        "d": 10, "s": 5, "delta": 0.001, "design": "random",
        "create_models": [optimal_ridge, optimal_lasso, elastic_net(1e-2)]
    }, "n", np.logspace(4, 6, num=100, dtype=int))

    # Run simulations and produce CSV results file for Figure 2(b).
    run_experiment("s_random", {
        "d": 100, "n": 100000, "delta": 0.001, "design": "random",
        "create_models": [optimal_ridge, optimal_lasso, elastic_net(1e-2)]
    }, "s", range(1, 31))

    # Run simulations and produce CSV results file for Figure 2(c).
    run_experiment("d_random", {
        "n": 100000, "s": 10, "delta": 0.001, "design": "random",
        "create_models": [optimal_ridge, optimal_lasso, elastic_net(1e-2)]
    }, "d", range(10, 101))

    # Run simulations and produce CSV results file for Figure 2(d).
    run_experiment("lambda2_elasticnet_random", {
        "n": 100000, "s": 10, "delta": 0.001, "design": "random",
        "create_models": [elastic_net(lam_2) for lam_2 in np.logspace(-2.5, -1.0, num=4)]
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


def run_trial(d: int, n: int, s: int, delta: float, design: str, create_models: List[ModelConstructor]) -> List[float]:
    """
    Construct context vectors, sample response variables, perform CDF regression, and compute estimation errors based on
    the given hyperparameters as defined in Theorem 1.
    :param d: dimension of the CDF basis
    :param n: sample size
    :param s: sparsity (l0-norm) of the true parameter
    :param delta: failure probability
    :param design: data generation process for context vectors; "fixed" or "random"
    :param create_models: constructors for models to compute estimation errors of
    """
    assert design in {"fixed", "random"}

    theta_true = random_sparse_pmf(d, s)
    x_list = fixed_x_j_list(n, d) if design == "fixed" else random_x_j_list(n, d)
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


@functools.lru_cache(maxsize=1)   # don't recompute deterministic context vectors for successive trials
def fixed_x_j_list(n: int, d: int) -> List[np.ndarray]:
    """
    Construct context vectors according to the fixed design procedure described in Section 5.
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


def random_x_j_list(n: int, d: int) -> List[np.ndarray]:
    """
    Sample n random context vectors, each containing d Bernoulli parameters.
    :param n: sample size
    :param d: dimension of the CDF basis
    """
    return list(np.random.rand(n, d))


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


def ridge(lam: float) -> ModelConstructor:
    """
    Return a constructor for a ridge regression object (using the loss formulation in Equation 1) with the given
    regularization hyperparameter.
    :param lam: the regularization hyperparameter
    """
    # noinspection PyUnusedLocal
    def constructor(d: int, n: int, s: int, delta: float) -> Ridge:
        return Ridge(alpha=n * lam, fit_intercept=False, max_iter=1000000, random_state=1731)
    return constructor


# noinspection PyUnusedLocal
def optimal_lasso(d: int, n: int, s: int, delta: float) -> Lasso:
    """
    Construct a lasso regression object (using the loss formulation in Equation 1) with regularization hyperparameter
    lambda = 4 sqrt(2/n log(2d/delta)), as stated in Theorem 1.
    """
    lam = 4 * sqrt(2 / n * log(2 * d / delta))
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    return Lasso(alpha=lam / 2, fit_intercept=False, max_iter=1000000, random_state=1731)


def elastic_net(lam_2: float) -> ModelConstructor:
    """
    Return a constructor for an elastic net regression object (using the loss formulation in Equation 2) with
    l1-regularization hyperparameter lambda_1 = 4 sqrt(2/n log(2d/delta)), as stated in Theorem 3, and the given
    l2-regularization hyperparameter.
    :param lam_2: the l2-regularization hyperparameter
    """
    # noinspection PyUnusedLocal
    def constructor(d: int, n: int, s: int, delta: float) -> ElasticNet:
        lam_1 = 4 * sqrt(2 / n * log(2 * d / delta))
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
        return ElasticNet(
            alpha=lam_1 / 2 + lam_2, l1_ratio=lam_1 / (lam_1 + 2 * lam_2), fit_intercept=False, max_iter=1000000,
            random_state=1731
        )
    return constructor


if __name__ == "__main__":
    main()
