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
import ipywidgets as widgets
from IPython.display import clear_output, display
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

MONETARY_SCALE = 1e9
N_INNER_TEST = 10_000
REQUIRED_FILES = ("train_X.csv", "train_Z.csv", "test_X.csv", "test_Y.csv")
MAIN_EFFECT_DEGREE = 3

# A deliberately compact menu: participants select individual cross-factor
# monomials, while exhaustive polynomial bases remain a separate benchmark.
INTERACTION_CANDIDATES = (
    (
        "Degree 2",
        (
            ("x × y", ("initial_x", "initial_y")),
            (
                "PC1 × PC2",
                ("yield_curve_pc1_risk_factor", "yield_curve_pc2_risk_factor"),
            ),
            (
                "PC1 × PC3",
                ("yield_curve_pc1_risk_factor", "yield_curve_pc3_risk_factor"),
            ),
            (
                "PC2 × PC3",
                ("yield_curve_pc2_risk_factor", "yield_curve_pc3_risk_factor"),
            ),
            ("x × PC1", ("initial_x", "yield_curve_pc1_risk_factor")),
            ("y × PC1", ("initial_y", "yield_curve_pc1_risk_factor")),
            (
                "PC1 × stock volatility",
                ("yield_curve_pc1_risk_factor", "stock_vola_risk_factor"),
            ),
            (
                "stock volatility × credit default",
                ("stock_vola_risk_factor", "credit_default_rf"),
            ),
            (
                "mortality × base lapse",
                ("mortality_risk_factor", "base_lapse_risk_factor"),
            ),
            (
                "base lapse × mass lapse",
                ("base_lapse_risk_factor", "mass_lapse_risk_factor"),
            ),
        ),
    ),
    (
        "Degree 3",
        (
            (
                "PC1 × PC2 × PC3",
                (
                    "yield_curve_pc1_risk_factor",
                    "yield_curve_pc2_risk_factor",
                    "yield_curve_pc3_risk_factor",
                ),
            ),
            (
                "x × y × PC1",
                ("initial_x", "initial_y", "yield_curve_pc1_risk_factor"),
            ),
            (
                "PC1² × PC2",
                (
                    "yield_curve_pc1_risk_factor",
                    "yield_curve_pc1_risk_factor",
                    "yield_curve_pc2_risk_factor",
                ),
            ),
            (
                "PC1 × PC2²",
                (
                    "yield_curve_pc1_risk_factor",
                    "yield_curve_pc2_risk_factor",
                    "yield_curve_pc2_risk_factor",
                ),
            ),
            (
                "PC1² × stock volatility",
                (
                    "yield_curve_pc1_risk_factor",
                    "yield_curve_pc1_risk_factor",
                    "stock_vola_risk_factor",
                ),
            ),
            (
                "stock volatility² × credit default",
                (
                    "stock_vola_risk_factor",
                    "stock_vola_risk_factor",
                    "credit_default_rf",
                ),
            ),
        ),
    ),
)


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


@dataclass(frozen=True)
class ManualSpecification:
    """One evaluated hand-selected interaction specification."""

    main_effect_degree: int
    interactions: tuple[tuple[str, ...], ...]
    r2: float
    rmse: float
    delta_scr: float
    fit_time: float
    bic: float
    basis_terms: int


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
        "ΔSCR (€bn)": np.quantile(prediction, 0.995)
        - np.quantile(truth, 0.995),
    }


def _bic_score(
    y_train: pd.Series | np.ndarray,
    fitted_values: np.ndarray,
    n_parameters: int,
) -> float:
    """Gaussian OLS BIC; lower values indicate a preferred specification."""
    residuals = np.asarray(y_train) - np.asarray(fitted_values)
    residual_sum_squares = max(
        float(residuals @ residuals), np.finfo(float).tiny
    )
    n_observations = len(residuals)
    return (
        n_observations * np.log(residual_sum_squares / n_observations)
        + n_parameters * np.log(n_observations)
    )


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

    constant_training_fit = np.full(len(data.y_train), data.y_train.mean())
    constant_prediction = np.full(len(data.y_test), data.y_train.mean())
    rows.append(
        {
            "model": "constant baseline",
            "basis terms": 1,
            "basis build (s)": 0.0,
            "fit time (s)": 0.0,
            "total time (s)": 0.0,
            "BIC (train)": _bic_score(
                data.y_train, constant_training_fit, n_parameters=1
            ),
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
        training_fit = model.predict(train_basis)
        prediction = model.predict(test_basis)

        predictions[name] = prediction
        rows.append(
            {
                "model": name,
                "basis terms": train_basis.shape[1],
                "basis build (s)": basis_time,
                "fit time (s)": fit_time,
                "total time (s)": basis_time + fit_time,
                "BIC (train)": _bic_score(
                    data.y_train,
                    training_fit,
                    n_parameters=train_basis.shape[1] + 1,
                ),
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
        training_fit = model.predict(train_basis)
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
                "BIC (train)": _bic_score(
                    data.y_train,
                    training_fit,
                    n_parameters=transformer.n_output_features_ + 1,
                ),
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


def plot_distribution_diagnostics(
    truth: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
    title: str,
) -> None:
    """Compare the predictive and benchmark distributions by Q-Q plot and ECDF."""
    truth_values = np.asarray(truth)
    prediction_values = np.asarray(prediction)
    probability_grid = np.linspace(0.01, 0.99, 99)
    truth_quantiles = np.quantile(truth_values, probability_grid)
    prediction_quantiles = np.quantile(prediction_values, probability_grid)
    lower = min(truth_values.min(), prediction_values.min())
    upper = max(truth_values.max(), prediction_values.max())

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    axes[0].scatter(
        truth_quantiles,
        prediction_quantiles,
        s=24,
        alpha=0.75,
        color="#2F6B8A",
        edgecolor="white",
    )
    axes[0].plot(
        [lower, upper],
        [lower, upper],
        "--",
        color="black",
        label="matching quantiles",
    )
    axes[0].set_xlabel("Benchmark quantiles (€bn)")
    axes[0].set_ylabel("Prediction quantiles (€bn)")
    axes[0].set_title("Q–Q comparison")
    axes[0].legend()

    for values, label, color in (
        (truth_values, "10,000-path benchmark", "#222222"),
        (prediction_values, "proxy prediction", "#2F6B8A"),
    ):
        sorted_values = np.sort(values)
        cumulative_probability = np.arange(1, len(values) + 1) / len(values)
        axes[1].step(
            sorted_values,
            cumulative_probability,
            where="post",
            linewidth=2,
            label=label,
            color=color,
        )
    axes[1].set_xlabel("Available capital (€bn)")
    axes[1].set_ylabel("Empirical cumulative probability")
    axes[1].set_title("Empirical CDF")
    axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


class ManualInteractionExplorer:
    """Small widget UI for the hand-selection challenge."""

    def __init__(self, data: LabData):
        self.data = data
        self.history: list[ManualSpecification] = []
        self.best_specification: ManualSpecification | None = None
        self.best_bic_specification: ManualSpecification | None = None
        self._checkboxes: dict[tuple[str, ...], widgets.Checkbox] = {}
        self._labels: dict[tuple[str, ...], str] = {}

        columns = []
        for heading, candidates in INTERACTION_CANDIDATES:
            checkboxes = []
            for label, factors in candidates:
                checkbox = widgets.Checkbox(
                    value=False,
                    description=label,
                    indent=False,
                    style={"description_width": "initial"},
                    layout=widgets.Layout(width="auto"),
                )
                self._checkboxes[factors] = checkbox
                self._labels[factors] = label
                checkboxes.append(checkbox)
            columns.append(
                widgets.VBox(
                    [widgets.HTML(f"<b>{heading} interactions</b>"), *checkboxes],
                    layout=widgets.Layout(
                        width="49%", border="1px solid #dddddd", padding="8px"
                    ),
                )
            )

        fit_button = widgets.Button(
            description="Fit selected model",
            button_style="primary",
            icon="play",
        )
        clear_button = widgets.Button(
            description="Reset to no interactions", icon="eraser"
        )
        fit_button.on_click(self._evaluate)
        clear_button.on_click(self._clear)

        self.output = widgets.Output()
        self.widget = widgets.VBox(
            [
                widgets.HTML(
                    "Every manual model contains all univariate powers through "
                    "degree 3. The checkboxes add only the selected cross-factor terms."
                ),
                widgets.HBox(columns, layout=widgets.Layout(align_items="flex-start")),
                widgets.HBox([fit_button, clear_button]),
                self.output,
            ]
        )

        display(self.widget)
        self._evaluate()

    def _selected_interactions(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            factors
            for factors, checkbox in self._checkboxes.items()
            if checkbox.value
        )

    def _clear(self, _button=None) -> None:
        for checkbox in self._checkboxes.values():
            checkbox.value = False
        self._evaluate()

    def _evaluate(self, _button=None) -> None:
        selected = self._selected_interactions()
        experiment = run_experiment(
            self.data,
            main_effect_degree=MAIN_EFFECT_DEGREE,
            selected_interactions=list(selected),
            all_terms_degrees=(),
        )
        if selected:
            model_name = (
                f"degree {MAIN_EFFECT_DEGREE} + {len(selected)} selected interactions"
            )
        else:
            model_name = f"main effects to degree {MAIN_EFFECT_DEGREE}"

        row = experiment.results.loc[model_name]
        specification = ManualSpecification(
            main_effect_degree=MAIN_EFFECT_DEGREE,
            interactions=selected,
            r2=float(row["R²"]),
            rmse=float(row["RMSE (€bn)"]),
            delta_scr=float(row["ΔSCR (€bn)"]),
            fit_time=float(row["fit time (s)"]),
            bic=float(row["BIC (train)"]),
            basis_terms=int(row["basis terms"]),
        )
        self.history.append(specification)
        if (
            self.best_specification is None
            or specification.r2 > self.best_specification.r2
        ):
            self.best_specification = specification
        if (
            self.best_bic_specification is None
            or specification.bic < self.best_bic_specification.bic
        ):
            self.best_bic_specification = specification

        selected_labels = [self._labels[factors] for factors in selected]
        best_r2_labels = [
            self._labels[factors]
            for factors in self.best_specification.interactions
        ]
        best_bic_labels = [
            self._labels[factors]
            for factors in self.best_bic_specification.interactions
        ]
        with self.output:
            clear_output(wait=True)
            print(
                f"Current R²: {specification.r2:.4f}    "
                f"Best R² so far: {self.best_specification.r2:.4f}    "
                f"Lowest BIC so far: {self.best_bic_specification.bic:.1f}"
            )
            display(
                pd.DataFrame(
                    {
                        "selected interactions": [len(selected)],
                        "basis terms": [specification.basis_terms],
                        "fit time (s)": [specification.fit_time],
                        "RMSE (€bn)": [specification.rmse],
                        "R²": [specification.r2],
                        "ΔSCR (€bn)": [specification.delta_scr],
                        "BIC (train; lower is better)": [specification.bic],
                    },
                    index=["current model"],
                ).round(4)
            )
            if selected_labels:
                print("Selected: " + ", ".join(selected_labels))
            else:
                print("Selected: none (univariate powers only)")
            if best_r2_labels:
                print("Best R² selection: " + ", ".join(best_r2_labels))
            else:
                print("Best R² selection: none (univariate powers only)")
            if best_bic_labels:
                print("Lowest-BIC selection: " + ", ".join(best_bic_labels))
            else:
                print("Lowest-BIC selection: none (univariate powers only)")
            plot_distribution_diagnostics(
                experiment.y_test,
                experiment.predictions[model_name],
                "Current hand-selected model",
            )


def interaction_selector(data: LabData) -> ManualInteractionExplorer:
    """Display and return the interaction-selection challenge."""
    return ManualInteractionExplorer(data)


def run_full_degree_comparison(
    data: LabData,
    manual_explorer: ManualInteractionExplorer,
    degrees: tuple[int, ...] = (2, 3),
) -> Experiment:
    """Compare the best manual specification with exhaustive polynomial bases."""
    if not degrees or any(degree not in (2, 3) for degree in degrees):
        raise ValueError(
            "This exercise compares exhaustive polynomial degrees 2 and 3 only."
        )
    specification = manual_explorer.best_specification
    if specification is None:
        raise RuntimeError("Fit at least one hand-selected model first.")

    experiment = run_experiment(
        data,
        main_effect_degree=specification.main_effect_degree,
        selected_interactions=list(specification.interactions),
        all_terms_degrees=degrees,
    )
    if not specification.interactions:
        manual_name = f"main effects to degree {specification.main_effect_degree}"
    else:
        manual_name = (
            f"degree {specification.main_effect_degree} + "
            f"{len(specification.interactions)} selected interactions"
        )

    if specification.interactions:
        display_name = (
            f"best-R² hand-selected model ({len(specification.interactions)} interactions)"
        )
    else:
        display_name = "best-R² manual model (no interactions)"
    results = experiment.results.rename(index={manual_name: display_name})
    predictions = {
        (display_name if name == manual_name else name): prediction
        for name, prediction in experiment.predictions.items()
    }
    return Experiment(results=results, predictions=predictions, y_test=experiment.y_test)


def comparison_table(experiment: Experiment) -> pd.DataFrame:
    """Return the workshop metrics and identify the BIC-selected model."""
    table = experiment.results[
        [
            "basis terms",
            "fit time (s)",
            "RMSE (€bn)",
            "R²",
            "ΔSCR (€bn)",
            "BIC (train)",
        ]
    ].copy()
    table["ΔBIC"] = table["BIC (train)"] - table["BIC (train)"].min()
    selected_name = table["BIC (train)"].idxmin()
    print(
        f'BIC selects "{selected_name}"; lower BIC and ΔBIC are better.'
    )
    return table.round(4)


def plot_model_comparison(experiment: Experiment) -> None:
    """Compare benchmark error, BIC, basis size, and measured runtime."""
    plt.style.use("seaborn-v0_8-whitegrid")
    model_results = experiment.results.drop(index="constant baseline")
    colors = plt.cm.Blues(np.linspace(0.45, 0.90, len(model_results)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    accuracy = model_results.sort_values("RMSE (€bn)")["RMSE (€bn)"]
    accuracy.plot.barh(
        ax=axes[0], color=plt.cm.Blues(np.linspace(0.45, 0.90, len(accuracy)))
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

    delta_bic = model_results["BIC (train)"] - model_results["BIC (train)"].min()
    delta_bic = delta_bic.sort_values(ascending=False)
    delta_bic.plot.barh(
        ax=axes[1], color=plt.cm.Oranges(np.linspace(0.45, 0.85, len(delta_bic)))
    )
    axes[1].set_xlabel("ΔBIC from selected model (lower is better)")
    axes[1].set_ylabel("")
    axes[1].set_title("Complexity-penalized fit")

    axes[2].scatter(
        model_results["basis terms"],
        model_results["total time (s)"],
        s=75,
        c=colors,
        edgecolor="white",
    )
    for name, row in model_results.iterrows():
        axes[2].annotate(
            name,
            (row["basis terms"], row["total time (s)"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axes[2].set_xlabel("Number of basis terms")
    axes[2].set_ylabel("Basis build + fit time (seconds)")
    axes[2].set_title("Computational cost")

    plt.tight_layout()
    plt.show()


def plot_bic_selected_distribution(experiment: Experiment) -> None:
    """Show Q-Q and ECDF diagnostics for the model selected by training BIC."""
    model_results = experiment.results.drop(index="constant baseline")
    selected_name = model_results["BIC (train)"].idxmin()
    selected_row = experiment.results.loc[selected_name]
    plot_distribution_diagnostics(
        experiment.y_test,
        experiment.predictions[selected_name],
        f"BIC-selected model: {selected_name}",
    )
    print(
        f"{selected_name}: R² = {selected_row['R²']:.3f}, "
        f"ΔSCR = {selected_row['ΔSCR (€bn)']:.3f} €bn, "
        f"fit time = {selected_row['fit time (s)']:.4f} s, "
        f"BIC = {selected_row['BIC (train)']:.1f}."
    )


def plot_best_distribution(experiment: Experiment) -> None:
    """Show Q-Q and ECDF diagnostics for the best tested non-constant model."""
    model_results = experiment.results.drop(index="constant baseline")
    best_name = model_results["R²"].idxmax()
    plot_distribution_diagnostics(
        experiment.y_test,
        experiment.predictions[best_name],
        f"Best tested model: {best_name}",
    )
    print(
        f"{best_name}: RMSE = {experiment.results.loc[best_name, 'RMSE (€bn)']:.3f} €bn, "
        f"R² = {experiment.results.loc[best_name, 'R²']:.3f}."
    )


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
