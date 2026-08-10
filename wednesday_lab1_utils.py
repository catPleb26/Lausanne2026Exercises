"""Supporting functions for Lab 1.

The notebook deliberately exposes only the experiment choices.  Data checks,
basis construction, model fitting, evaluation, and plotting live here so that
participants can concentrate on the modelling decisions.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

MONETARY_SCALE = 1e9
N_INNER_TEST = 10_000
REQUIRED_FILES = ("train_X.csv", "train_Z.csv", "test_X.csv", "test_Y.csv")


@dataclass
class LabData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    mc_se_train: pd.Series
    benchmark_mc_se: pd.Series
    n_inner_train: int


@dataclass
class Experiment:
    results: pd.DataFrame
    predictions: dict[str, np.ndarray]
    y_test: pd.Series


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
        raise ValueError(
            "The training data should contain 54 inner outcomes per state."
        )
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
        benchmark_mc_se=(test_Y["sd"] / MONETARY_SCALE / np.sqrt(N_INNER_TEST)),
        n_inner_train=n_inner_train,
    )


def data_summary(data: LabData) -> pd.DataFrame:
    """Return the quantities needed to interpret the regression experiment."""
    return pd.DataFrame(
        {
            "quantity": [
                "outer training states",
                "risk factors",
                "inner outcomes per training state",
                "outer test states",
                "inner outcomes per test benchmark",
                "mean training-target MC standard error (€bn)",
                "mean benchmark MC standard error (€bn)",
            ],
            "value": [
                len(data.X_train),
                data.X_train.shape[1],
                data.n_inner_train,
                len(data.X_test),
                N_INNER_TEST,
                data.mc_se_train.mean(),
                data.benchmark_mc_se.mean(),
            ],
        }
    )


def plot_training_targets(data: LabData) -> None:
    """Plot target dispersion and the remaining simulation noise."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    axes[0].hist(data.y_train, bins=40, color="#2F6B8A", alpha=0.85)
    axes[0].set_title("Training targets")
    axes[0].set_xlabel("Mean available capital (€bn)")
    axes[0].set_ylabel("Outer states")

    axes[1].hist(data.mc_se_train, bins=40, color="#D88C36", alpha=0.85)
    axes[1].axvline(
        data.mc_se_train.mean(), color="black", linestyle="--", label="mean"
    )
    axes[1].set_title("Monte Carlo noise in each training mean")
    axes[1].set_xlabel("Standard error (€bn)")
    axes[1].set_ylabel("Outer states")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(y_pred)
    truth = np.asarray(y_true)
    return {
        "RMSE (€bn)": np.sqrt(mean_squared_error(truth, prediction)),
        "MAE (€bn)": mean_absolute_error(truth, prediction),
        "bias (€bn)": np.mean(prediction - truth),
        "R²": r2_score(truth, prediction),
    }


def _selected_basis(
    X_scaled: pd.DataFrame,
    main_degree: int,
    interactions: list[tuple[str, ...]],
) -> pd.DataFrame:
    if main_degree < 1:
        raise ValueError("MAIN_EFFECT_DEGREE must be at least 1.")

    feature_names = list(X_scaled.columns)
    valid_names = set(feature_names)
    basis: dict[str, np.ndarray] = {}

    for name in feature_names:
        values = X_scaled[name].to_numpy()
        for power in range(1, main_degree + 1):
            label = name if power == 1 else f"{name}^{power}"
            basis[label] = values**power

    seen_interactions: set[tuple[int, ...]] = set()
    for factors in interactions:
        if isinstance(factors, str) or len(factors) < 2:
            raise ValueError(
                f"Each interaction must be a tuple of at least two names: {factors!r}"
            )
        unknown = set(factors) - valid_names
        if unknown:
            raise ValueError(
                f"Unknown risk factor(s) {sorted(unknown)}. Choose from {feature_names}."
            )
        if len(set(factors)) < 2:
            raise ValueError(
                f"An interaction must involve at least two risk factors: {factors!r}"
            )

        powers = Counter(factors)
        signature = tuple(powers.get(name, 0) for name in feature_names)
        if signature in seen_interactions:
            continue
        seen_interactions.add(signature)

        pieces = [
            name if powers[name] == 1 else f"{name}^{powers[name]}"
            for name in feature_names
            if powers[name] > 0
        ]
        values = np.ones(len(X_scaled))
        for name, power in powers.items():
            values *= X_scaled[name].to_numpy() ** power
        basis[" × ".join(pieces)] = values

    return pd.DataFrame(basis, index=X_scaled.index)


def run_experiment(
    data: LabData,
    main_effect_degree: int,
    selected_interactions: list[tuple[str, ...]],
    all_terms_degrees: tuple[int, ...],
) -> Experiment:
    """Fit the compact bases and exhaustive polynomial reference models."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(data.X_train),
        columns=data.X_train.columns,
        index=data.X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(data.X_test),
        columns=data.X_test.columns,
        index=data.X_test.index,
    )

    predictions: dict[str, np.ndarray] = {}
    rows: list[dict[str, float | int | str]] = []

    constant_prediction = np.full(len(data.y_test), data.y_train.mean())
    rows.append(
        {
            "model": "constant baseline",
            "basis terms": 1,
            "basis build (s)": 0.0,
            "fit time (s)": 0.0,
            "total time (s)": 0.0,
            **_regression_metrics(data.y_test, constant_prediction),
        }
    )

    def fit_selected(name: str, degree: int, interactions: list[tuple[str, ...]]):
        started = perf_counter()
        train_basis = _selected_basis(X_train_scaled, degree, interactions)
        test_basis = _selected_basis(X_test_scaled, degree, interactions)
        basis_time = perf_counter() - started

        model = LinearRegression()
        started = perf_counter()
        model.fit(train_basis, data.y_train)
        fit_time = perf_counter() - started
        prediction = model.predict(test_basis)

        predictions[name] = prediction
        rows.append(
            {
                "model": name,
                "basis terms": train_basis.shape[1],
                "basis build (s)": basis_time,
                "fit time (s)": fit_time,
                "total time (s)": basis_time + fit_time,
                **_regression_metrics(data.y_test, prediction),
            }
        )

    def fit_exhaustive(degree: int):
        if degree < 1:
            raise ValueError("Every exhaustive polynomial degree must be at least 1.")
        started = perf_counter()
        transformer = PolynomialFeatures(degree=degree, include_bias=False)
        train_basis = transformer.fit_transform(X_train_scaled)
        test_basis = transformer.transform(X_test_scaled)
        basis_time = perf_counter() - started

        model = LinearRegression()
        started = perf_counter()
        model.fit(train_basis, data.y_train)
        fit_time = perf_counter() - started
        prediction = model.predict(test_basis)
        name = f"all terms to degree {degree}"

        predictions[name] = prediction
        rows.append(
            {
                "model": name,
                "basis terms": transformer.n_output_features_,
                "basis build (s)": basis_time,
                "fit time (s)": fit_time,
                "total time (s)": basis_time + fit_time,
                **_regression_metrics(data.y_test, prediction),
            }
        )

    fit_selected("linear main effects", 1, [])
    if main_effect_degree > 1:
        fit_selected(
            f"main effects to degree {main_effect_degree}", main_effect_degree, []
        )
    if selected_interactions:
        fit_selected(
            f"degree {main_effect_degree} + {len(selected_interactions)} selected interactions",
            main_effect_degree,
            selected_interactions,
        )
    for degree in dict.fromkeys(all_terms_degrees):
        fit_exhaustive(degree)

    results = pd.DataFrame(rows).set_index("model")
    return Experiment(results=results, predictions=predictions, y_test=data.y_test)


def plot_model_comparison(experiment: Experiment) -> None:
    """Compare benchmark error with basis size and measured runtime."""
    plt.style.use("seaborn-v0_8-whitegrid")
    model_results = experiment.results.drop(index="constant baseline")
    colors = plt.cm.Blues(np.linspace(0.40, 0.90, len(model_results)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    model_results.sort_values("RMSE (€bn)")["RMSE (€bn)"].plot.barh(
        ax=axes[0], color=colors
    )
    axes[0].axvline(
        experiment.results.loc["constant baseline", "RMSE (€bn)"],
        color="gray",
        linestyle="--",
        label="constant baseline",
    )
    axes[0].set_xlabel("Test RMSE (€bn; lower is better)")
    axes[0].set_ylabel("")
    axes[0].set_title("Benchmark accuracy")
    axes[0].legend()

    axes[1].scatter(
        model_results["basis terms"],
        model_results["total time (s)"],
        s=75,
        c=colors,
        edgecolor="white",
    )
    for name, row in model_results.iterrows():
        axes[1].annotate(
            name, (row["basis terms"] + 30, row["total time (s)"]), fontsize=8
        )
    axes[1].set_xlabel("Number of basis terms")
    axes[1].set_ylabel("Basis build + fit time (seconds)")
    axes[1].set_title("Computational cost")

    plt.tight_layout()
    plt.show()


def plot_best_prediction(experiment: Experiment) -> None:
    """Plot the best tested proxy against the 10,000-path benchmark."""
    model_results = experiment.results.drop(index="constant baseline")
    best_name = model_results["RMSE (€bn)"].idxmin()
    prediction = experiment.predictions[best_name]
    lower = min(experiment.y_test.min(), prediction.min())
    upper = max(experiment.y_test.max(), prediction.max())

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(5.5, 5.0))
    plt.scatter(
        experiment.y_test,
        prediction,
        alpha=0.65,
        color="#2F6B8A",
        edgecolor="white",
    )
    plt.plot(
        [lower, upper], [lower, upper], "--", color="black", label="perfect prediction"
    )
    plt.xlabel("10,000-path benchmark (€bn)")
    plt.ylabel("Proxy prediction (€bn)")
    plt.title(f"Best tested model: {best_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(
        f"{best_name}: RMSE = {experiment.results.loc[best_name, 'RMSE (€bn)']:.3f} €bn, "
        f"R² = {experiment.results.loc[best_name, 'R²']:.3f}."
    )
