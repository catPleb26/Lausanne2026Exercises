"""Supporting functions for Lab 2.

The modelling implementation is kept outside the notebook so participants see
their experiment settings next to the resulting comparison.
"""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


MONETARY_SCALE = 1e9
RANDOM_SEED = 42
REQUIRED_FILES = ("train_X.csv", "train_Z.csv", "test_X.csv", "test_Y.csv")


@dataclass
class LabData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    mc_se_train: pd.Series
    n_inner_train: int


@dataclass
class Experiment:
    results: pd.DataFrame
    predictions: dict[str, np.ndarray]
    y_test: pd.Series
    details: dict[str, str | int]


def _find_data_folder() -> Path:
    for folder in (Path("data"), Path(".")):
        if all((folder / name).exists() for name in REQUIRED_FILES):
            return folder
    raise FileNotFoundError(
        "Could not find the four CSV files. Keep the supplied data folder next "
        "to this notebook, or place the CSV files directly next to it."
    )


def load_data() -> LabData:
    """Load, validate, and aggregate the supplied nested-simulation data."""
    folder = _find_data_folder()
    train_X = pd.read_csv(folder / "train_X.csv")
    train_Z = pd.read_csv(folder / "train_Z.csv")
    test_X = pd.read_csv(folder / "test_X.csv")
    test_Y = pd.read_csv(folder / "test_Y.csv")

    if len(train_X) != len(train_Z):
        raise ValueError("train_X and train_Z must have the same number of rows.")
    if list(train_X.columns) != list(test_X.columns):
        raise ValueError("The risk-factor columns in train_X and test_X differ.")
    if not {"mean", "sd"}.issubset(test_Y.columns):
        raise ValueError("test_Y must contain the columns 'mean' and 'sd'.")
    if len(test_X) != len(test_Y):
        raise ValueError("test_X and test_Y must have the same number of rows.")
    if train_Z.shape[1] != 54:
        raise ValueError("The training data should contain 54 inner outcomes per state.")
    if any(frame.isna().any().any() for frame in (train_X, train_Z, test_X, test_Y)):
        raise ValueError("The supplied data contain missing values.")

    n_inner_train = train_Z.shape[1]
    return LabData(
        X_train=train_X,
        y_train=train_Z.mean(axis=1) / MONETARY_SCALE,
        X_test=test_X,
        y_test=test_Y["mean"] / MONETARY_SCALE,
        mc_se_train=(
            train_Z.std(axis=1, ddof=1) / MONETARY_SCALE / np.sqrt(n_inner_train)
        ),
        n_inner_train=n_inner_train,
    )


def data_summary(data: LabData) -> pd.DataFrame:
    """Return the central dimensions of the fitting and benchmark data."""
    return pd.DataFrame(
        {
            "quantity": [
                "outer training states",
                "risk factors",
                "inner outcomes per training state",
                "outer test states",
                "inner outcomes per test benchmark",
            ],
            "value": [
                len(data.X_train),
                data.X_train.shape[1],
                data.n_inner_train,
                len(data.X_test),
                10_000,
            ],
        }
    )


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(y_pred)
    truth = np.asarray(y_true)
    return {
        "RMSE (€bn)": np.sqrt(mean_squared_error(truth, prediction)),
        "MAE (€bn)": mean_absolute_error(truth, prediction),
        "bias (€bn)": np.mean(prediction - truth),
        "R²": r2_score(truth, prediction),
    }


def run_experiment(
    data: LabData,
    nn_hidden_layers: tuple[int, ...],
    nn_activation: str,
    nn_regularization: float,
    gp_training_size: int,
    gp_optimize_kernel: bool,
) -> Experiment:
    """Fit polynomial, neural-network, and Gaussian-process proxy models."""
    if nn_activation not in {"tanh", "relu"}:
        raise ValueError('NN_ACTIVATION must be either "tanh" or "relu".')
    if not nn_hidden_layers or any(width < 1 for width in nn_hidden_layers):
        raise ValueError("NN_HIDDEN_LAYERS must contain positive layer widths.")
    if not 1 <= gp_training_size <= len(data.X_train):
        raise ValueError(
            f"GP_TRAINING_SIZE must be between 1 and {len(data.X_train)}."
        )
    if gp_training_size > 2_500:
        print("Warning: a standard GP above 2,500 states may be slow and memory-intensive.")

    predictions: dict[str, np.ndarray] = {}
    fit_times: dict[str, float] = {}
    details: dict[str, str | int] = {}

    polynomial_name = "Polynomial LSMC (degree 2)"
    polynomial_model = Pipeline(
        steps=[
            ("scale inputs", StandardScaler()),
            ("polynomial basis", PolynomialFeatures(degree=2, include_bias=False)),
            ("least squares", LinearRegression()),
        ]
    )
    started = perf_counter()
    polynomial_model.fit(data.X_train, data.y_train)
    fit_times[polynomial_name] = perf_counter() - started
    predictions[polynomial_name] = polynomial_model.predict(data.X_test)
    details["polynomial basis terms"] = polynomial_model.named_steps[
        "polynomial basis"
    ].n_output_features_

    neural_network_name = f"Neural network {nn_hidden_layers}"
    neural_network = Pipeline(
        steps=[
            ("scale inputs", StandardScaler()),
            (
                "neural network",
                MLPRegressor(
                    hidden_layer_sizes=nn_hidden_layers,
                    activation=nn_activation,
                    solver="adam",
                    alpha=nn_regularization,
                    learning_rate_init=5e-4,
                    max_iter=1_500,
                    early_stopping=True,
                    validation_fraction=0.20,
                    n_iter_no_change=50,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    started = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        neural_network.fit(data.X_train, data.y_train)
    fit_times[neural_network_name] = perf_counter() - started
    predictions[neural_network_name] = neural_network.predict(data.X_test)
    details["neural-network epochs"] = neural_network.named_steps[
        "neural network"
    ].n_iter_

    gp_input_scaler = StandardScaler().fit(data.X_train)
    X_train_gp = gp_input_scaler.transform(data.X_train)
    X_test_gp = gp_input_scaler.transform(data.X_test)

    y_center = data.y_train.mean()
    y_scale = data.y_train.std(ddof=0)
    y_train_gp = (data.y_train.to_numpy() - y_center) / y_scale
    mc_noise_variance_gp = (data.mc_se_train.to_numpy() / y_scale) ** 2

    rng = np.random.default_rng(RANDOM_SEED)
    gp_indices = rng.permutation(len(data.X_train))[:gp_training_size]
    kernel = ConstantKernel(
        constant_value=1.0,
        constant_value_bounds=(1e-2, 1e2),
    ) * RBF(
        length_scale=1.0,
        length_scale_bounds=(1e-2, 1e2),
    )
    gp_model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=mc_noise_variance_gp[gp_indices] + 1e-8,
        optimizer="fmin_l_bfgs_b" if gp_optimize_kernel else None,
        n_restarts_optimizer=0,
        normalize_y=False,
        random_state=RANDOM_SEED,
    )

    gp_name = f"Gaussian process (n={gp_training_size})"
    started = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp_model.fit(X_train_gp[gp_indices], y_train_gp[gp_indices])
    fit_times[gp_name] = perf_counter() - started
    predictions[gp_name] = y_center + y_scale * gp_model.predict(X_test_gp)
    details["fitted GP kernel"] = str(gp_model.kernel_)

    results = pd.DataFrame(
        [
            {
                "model": name,
                "fit time (s)": fit_times[name],
                **_metrics(data.y_test, prediction),
            }
            for name, prediction in predictions.items()
        ]
    ).set_index("model").sort_values("RMSE (€bn)")

    return Experiment(
        results=results,
        predictions=predictions,
        y_test=data.y_test,
        details=details,
    )


def experiment_details(experiment: Experiment) -> pd.DataFrame:
    """Show small pieces of fitted-model information without implementation noise."""
    return pd.DataFrame(
        {"quantity": list(experiment.details), "value": list(experiment.details.values())}
    )


def plot_comparison(experiment: Experiment) -> None:
    """Plot benchmark accuracy and model-fitting time."""
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = ["#2F6B8A", "#6C9F73", "#D88C36"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    experiment.results.sort_values("RMSE (€bn)")["RMSE (€bn)"].plot.barh(
        ax=axes[0], color=colors
    )
    axes[0].set_xlabel("Test RMSE (€bn; lower is better)")
    axes[0].set_ylabel("")
    axes[0].set_title("Accuracy against the 10,000-path benchmark")

    experiment.results.sort_values("fit time (s)")["fit time (s)"].plot.barh(
        ax=axes[1], color=colors
    )
    axes[1].set_xlabel("Fit time (seconds; machine-dependent)")
    axes[1].set_ylabel("")
    axes[1].set_title("Computational cost")

    plt.tight_layout()
    plt.show()


def plot_predictions(experiment: Experiment) -> None:
    """Plot model predictions against benchmark conditional expectations."""
    lower = min(
        experiment.y_test.min(),
        *(np.min(value) for value in experiment.predictions.values()),
    )
    upper = max(
        experiment.y_test.max(),
        *(np.max(value) for value in experiment.predictions.values()),
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True, sharey=True)
    colors = ("#2F6B8A", "#6C9F73", "#D88C36")

    for ax, (name, prediction), color in zip(
        axes, experiment.predictions.items(), colors
    ):
        ax.scatter(
            experiment.y_test,
            prediction,
            alpha=0.55,
            color=color,
            edgecolor="white",
            s=28,
        )
        ax.plot([lower, upper], [lower, upper], "--", color="black", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("Benchmark (€bn)")

    axes[0].set_ylabel("Proxy prediction (€bn)")
    fig.suptitle("Predicted versus benchmark conditional expectations", y=1.02)
    plt.tight_layout()
    plt.show()
