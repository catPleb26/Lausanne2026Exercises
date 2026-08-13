import numpy as np
from dataclasses import dataclass
from enum import Enum

#########################
### CLASS DEFINITIONS ###
#########################
@dataclass(frozen=True)
class FinancialMarket:
    """
    A dataclass representing a financial market in the Black-Scholes model.

    Attributes:
    -----------
        risk_free_rate (float): The risk-free interest rate (r). [Default: 0.025]
        risk_premium (float): The risk premium (mu - r). [Default: 0.05]
        volatility (float): The volatility of the underlying asset (sigma). [Default: 0.25]

    Properties:
        drift (float): The drift of the underlying asset, calculated as 'risk_free_rate' + 'risk_premium'.
    """
    risk_free_rate: float = 0.025
    risk_premium: float   = 0.05
    volatility: float     = 0.25

    def __post_init__(self):
        # Parameter validation
        if self.volatility < 0:
            raise ValueError("Volatility must be non-negative.")

    @property
    def drift(self):
        """
        Calculate the drift of the underlying asset.

        Returns:
            float: The drift, calculated as 'risk_free_rate' + 'risk_premium'.
        """
        return self.risk_free_rate + self.risk_premium


@dataclass(frozen=True)
class SDESimulationParameters:
    """
    A dataclass representing the parameters for simulating a stochastic differential equation (SDE).

    Attributes:
    -----------
        time_horizon (float): The total time horizon for the simulation. [Default: 1.0]
        time_steps (int): The number of discrete time steps in the simulation. [Default: 250]
        num_paths (int): The number of simulation paths to generate. [Default: 1_000]
        seed (int): The random seed for reproducibility. [Default: 42]

    Properties:
    -----------
        dt (float): The time increment (dt) calculated as 'time_horizon' / 'time_steps'
    """
    time_horizon: float = 1.0
    time_steps: int     = 250
    num_paths: int      = 1_000
    seed: int           = 42

    def __post_init__(self):
        # Parameter validation
        if self.time_horizon <= 0:
            raise ValueError("Time horizon must be positive.")
        if self.time_steps <= 0:
            raise ValueError("Number of time steps must be positive.")
        if self.num_paths <= 0:
            raise ValueError("Number of paths must be positive.")

    @property
    def dt(self):
        """
        Calculate the time increment (dt) based on the time horizon and number of time steps.

        Returns:
            float: The time increment (dt).
        """
        return self.time_horizon / self.time_steps


@dataclass(frozen=True)
class SDEOutput:
    """
    A dataclass representing the output of a stochastic differential equation (SDE) simulation.

    Attributes:
    -----------
        time_grid (np.ndarray): The time grid used in the simulation.
        paths (dict[str, np.ndarray]): The simulated paths of the SDE. Each key in the dictionary corresponds to a different component of the trajectory, and the values are numpy arrays representing the simulated paths for that component.

    Properties:
    -----------
        quadratic_variation (np.ndarray): The quadratic variation of the simulated paths.
    """
    time_grid: np.ndarray
    paths: dict[str, np.ndarray]

    @property
    def quadratic_variation(self):
        """
        Calculate the quadratic variation of the simulated paths.

        Returns:
            np.ndarray: The quadratic variation of the paths.
        """
        return {key: np.cumsum(np.diff(value, axis=0) ** 2, axis=0) for key, value in self.paths.items()}


@dataclass(frozen=True)
class TestStatistics:
    """
    A dataclass representing the statistics of a test.

    Attributes:
    -----------
        mean (np.ndarray): The mean of the test results.
        std_dev (np.ndarray): The standard deviation of the test results.
        min_value (np.ndarray): The minimum value of the test results.
        max_value (np.ndarray): The maximum value of the test results.
        drange (np.ndarray): The range of the test results, calculated as 'max_value' - 'min_value'.
        quantiles (dict): A dictionary containing the quantiles of the test results. (5%, 25%, 50%, 75%, 95%)
    """
    mean: np.ndarray
    std_dev: np.ndarray
    min_value: np.ndarray
    max_value: np.ndarray
    drange: np.ndarray
    quantiles: dict

    def __str__(self):
        """
        Return a string representation of the TestStatistics object.
        Only implemented for scalar statistics (i.e., when the attributes are not arrays).

        Returns:
            str: A formatted string containing the statistics.
        """
        if not all(isinstance(attr, (int, float, np.float64)) for attr in [self.mean, self.std_dev, self.min_value, self.max_value, self.drange]):
            raise ValueError("String representation is only implemented for scalar statistics.")

        quantiles_str = '|'.join([f"{k}: {v:.6f}" for k, v in self.quantiles.items()])
        return (f"Mean ± StdDev: {self.mean:.6f} ± {self.std_dev:.6f},\n"
                f"(Min) {self.min_value:.6f} <-- {self.drange:.6f} --> {self.max_value:.6f} (Max)\n"
                f"Quantiles: \n|{quantiles_str}|")


class InvestmentType(Enum):
    ConstantFraction = 1
    ConstantNominal = 2
        
        
############################
### FUNCTION DEFINITIONS ###
############################
def check_value(name: str, value: float, expected: float, rel_tol: float = 1e-5, abs_tol: float = 1e-8, exit_on_fail: bool = False):
    """
    Check if a value is approximately equal to a reference value within specified tolerances.

    Args:
        name (str): The name of the test.
        value (float): The value to check.
        expected (float): The expected value.
        rel_tol (float): The relative tolerance. [Default: 1e-5]
        abs_tol (float): The absolute tolerance. [Default: 1e-8]
        exit_on_fail (bool): Whether to raise an AssertionError on failure. [Default: False]

    Raises:
        AssertionError: If the value is not approximately equal to the reference value. (Only raised if exit_on_fail is True)
    """

    bool_str = lambda x: "✅" if x else "❌"

    if abs(value - expected) <= max(rel_tol * abs(expected), abs_tol):
        print(f"{bool_str(True)} {name}: {value:.6f} ≈ {expected:.6f}")
    else:
        print(f"{bool_str(False)} {name}: {value:.6f} ≠ {expected:.6f}")
        if exit_on_fail:
            raise AssertionError(f"{name}: {value:.6f} ≠ {expected:.6f}")


def check_condition(name: str, condition: bool, exit_on_fail: bool = False):
    """
    Check if a condition is True and print the result.

    Args:
        name (str): The name of the test.
        condition (bool): The condition to check.
        exit_on_fail (bool): Whether to raise an AssertionError on failure. [Default: False]

    Raises:
        AssertionError: If the condition is False. (Only raised if exit_on_fail is True)
    """
    
    bool_str = lambda x: "✅" if x else "❌"

    if condition:
        print(f"{bool_str(True)} {name}: Condition is True")
    else:
        print(f"{bool_str(False)} {name}: Condition is False")
        if exit_on_fail:
            raise AssertionError(f"{name}: Condition is False")


def compute_test_statistics(data: np.ndarray, axis: int = 0) -> TestStatistics:
    """
    Compute basic statistics for a given dataset.

    Args:
        data (np.ndarray): A numpy array of numerical values.
        axis (int): The axis along which to compute the statistics. [Default: 0]

    Returns:
        TestStatistics: A dataclass containing the computed statistics.
    """

    mean = np.mean(data, axis=axis)
    std_dev = np.std(data, axis=axis)
    min_value = np.min(data, axis=axis)
    max_value = np.max(data, axis=axis)
    drange = max_value - min_value
    quantiles = {
        "5%": np.percentile(data, 5, axis=axis),
        "25%": np.percentile(data, 25, axis=axis),
        "50%": np.percentile(data, 50, axis=axis),
        "75%": np.percentile(data, 75, axis=axis),
        "95%": np.percentile(data, 95, axis=axis),
    }

    return TestStatistics(
        mean=mean, 
        std_dev=std_dev,
        min_value=min_value, 
        max_value=max_value, 
        drange=drange, 
        quantiles=quantiles
    )


def lognormal_pdf(x, mu, sigma):
    """
    Compute the probability density function (PDF) of a lognormal distribution.
    """
    density = np.zeros_like(x, dtype=float)
    positive = x > 0

    density[positive] = (
        1 / (x[positive] * sigma * np.sqrt(2 * np.pi))
        * np.exp(-0.5*((np.log(x[positive]) - mu) / sigma) ** 2)
    )

    return density


def exponential_utility(x: float, abs_risk_aversion: float):
    """
    Compute the exponential utility of a given value.
    """
    return 1 - np.exp(-abs_risk_aversion * x)


def expected_utility_exact(x:float, T: float, xi: float, abs_risk_aversion: float, market_params: FinancialMarket):
    """
    Compute the expected utility of a given value using the exact formula.

    NOTE: This formula assumes that the risk-free rate is zero (r = 0). If the risk-free rate is not zero, a ValueError will be raised.
    """

    if not np.isclose(market_params.risk_free_rate, 0):
        raise ValueError("This formula assumes r = 0.")
    
    return 1 - np.exp(
        -abs_risk_aversion * x
        -abs_risk_aversion * xi * market_params.drift * T
        +0.5 * abs_risk_aversion**2 * xi**2 * market_params.volatility**2 * T
    )


def generate_strategies(simulate_wealth_handle, xi_sub: float, abs_risk_aversion: float, market_params: FinancialMarket, sde_params: SDESimulationParameters) -> list[tuple]:
    """
    Generate a list of tuples containing information about optimal and suboptimal investment strategies.
    """

    # WLOG x = 1
    x = 1

    # Optimal and suboptimal constant monetary investments
    xi_star = market_params.drift / (abs_risk_aversion * market_params.volatility**2)
  
    # The same seed gives both strategies the same Brownian shocks
    out_optimal = simulate_wealth_handle(initial_wealth=x, constant_investment=xi_star, investment_strategy=InvestmentType.ConstantNominal, market_params=market_params, sde_params=sde_params)
    out_suboptimal = simulate_wealth_handle(initial_wealth=x, constant_investment=xi_sub, investment_strategy=InvestmentType.ConstantNominal, market_params=market_params, sde_params=sde_params)

    t     = out_optimal.time_grid
    X_opt = out_optimal.paths["X"]
    X_sub = out_suboptimal.paths["X"]

    # Utility along all paths
    U_opt = exponential_utility(X_opt, abs_risk_aversion=abs_risk_aversion)
    U_sub = exponential_utility(X_sub, abs_risk_aversion=abs_risk_aversion)

    # Mean utility and standard deviation at every time point
    stats_opt = compute_test_statistics(U_opt, axis=1)
    stats_sub = compute_test_statistics(U_sub, axis=1)

    margin_opt = 1.96 * stats_opt.std_dev / np.sqrt(sde_params.num_paths)
    margin_sub = 1.96 * stats_sub.std_dev / np.sqrt(sde_params.num_paths)

    mean_U_opt  = stats_opt.mean
    lower_U_opt = mean_U_opt - margin_opt
    upper_U_opt = mean_U_opt + margin_opt

    mean_U_sub  = stats_sub.mean
    lower_U_sub = mean_U_sub - margin_sub
    upper_U_sub = mean_U_sub + margin_sub

    # Analytical expected utility
    analytical_U_opt = expected_utility_exact(
        x, t, xi_star, abs_risk_aversion, market_params
    )

    analytical_U_sub = expected_utility_exact(
        x, t, xi_sub, abs_risk_aversion, market_params
    )

    # Return the strategies as a list of tuples
    strategies = [
        (
            "Optimal strategy",
            xi_star,
            t,
            X_opt,
            mean_U_opt,
            lower_U_opt,
            upper_U_opt,
            analytical_U_opt,
            "b"
        ),
        (
            "Suboptimal strategy",
            xi_sub,
            t,
            X_sub,
            mean_U_sub,
            lower_U_sub,
            upper_U_sub,
            analytical_U_sub,
            "orange"
        )
    ]

    return strategies

def cumulative_trapezoid(values: np.ndarray, dt: float) -> np.ndarray:
    '''Integrate pathwise rates without requiring an additional SciPy helper.'''

    cumulative = np.zeros_like(values)
    cumulative[1:, :] = np.cumsum(
        0.5 * (values[:-1, :] + values[1:, :]) * dt,
        axis=0,
    )
    return cumulative

def exponential_integral(rate: float, maturity: float) -> float:
    """
    Compute the exponential integral of a given rate over a specified maturity.

    Args:
        rate (float): The rate for which to compute the exponential integral.
        maturity (float): The maturity time over which to compute the integral.

    Returns:
        float: The computed exponential integral.
    """
    if np.isclose(rate, 0.0):
        return maturity
    return np.expm1(rate * maturity) / rate


###
###
###
### DAY 4
###
###
###

IPCC_AR6_SCENARIO_CENTRAL = {
    "SSP1-1.9": np.array([1.5, 1.6, 1.4]),
    "SSP1-2.6": np.array([1.5, 1.7, 1.8]),
    "SSP2-4.5": np.array([1.5, 2.0, 2.7]),
    "SSP5-8.5": np.array([1.6, 2.4, 4.4]),
}


def rebase_temperature_anomaly(years: np.ndarray, anomaly: np.ndarray, baseline_start: int = 1850, baseline_end: int = 1900) -> np.ndarray:
    """Rebase a temperature-anomaly series to a selected reference period."""

    years   = np.asarray(years)
    anomaly = np.asarray(anomaly, dtype=float)

    if years.shape != anomaly.shape:
        raise ValueError("Years and anomalies must have the same shape.")

    baseline_mask = (years >= baseline_start) & (years <= baseline_end)
    if not np.any(baseline_mask):
        raise ValueError("The selected baseline period is not contained in the data.")

    return anomaly - np.mean(anomaly[baseline_mask])


def estimate_arithmetic_brownian_parameters(years: np.ndarray, anomaly: np.ndarray) -> tuple[float, float]:
    """Estimate annual drift and volatility from consecutive annual changes."""

    years   = np.asarray(years)
    anomaly = np.asarray(anomaly, dtype=float)

    if years.shape != anomaly.shape:
        raise ValueError("Years and anomalies must have the same shape.")
    if len(years) < 3:
        raise ValueError("At least three annual observations are required.")
    if not np.allclose(np.diff(years), 1):
        raise ValueError("The observations must be consecutive annual values.")

    annual_changes = np.diff(anomaly)
    drift          = np.mean(annual_changes)
    volatility     = np.std(annual_changes - drift, ddof=1)

    return drift, volatility


def build_climate_scenario_mean(forecast_years: np.ndarray, initial_anomaly: float, scenario_values: np.ndarray) -> np.ndarray:
    """Interpolate an illustrative annual mean path through the IPCC period midpoints."""

    forecast_years = np.asarray(forecast_years)
    scenario_values = np.asarray(scenario_values, dtype=float)

    if scenario_values.shape != (3,):
        raise ValueError("Scenario values must contain the near-, mid-, and long-term central estimates.")
    if forecast_years[0] != 2020:
        raise ValueError("The forecast must start in 2020.")

    anchor_years  = np.array([2020, 2030, 2050, 2090, 2100])
    anchor_values = np.array([initial_anomaly, *scenario_values, scenario_values[-1]])

    return np.interp(forecast_years, anchor_years, anchor_values)

###
###
###
### DAY 4
###
###
###

IPCC_AR6_SCENARIO_CENTRAL = {
    "SSP1-1.9": np.array([1.5, 1.6, 1.4]),
    "SSP1-2.6": np.array([1.5, 1.7, 1.8]),
    "SSP2-4.5": np.array([1.5, 2.0, 2.7]),
    "SSP5-8.5": np.array([1.6, 2.4, 4.4]),
}


def rebase_temperature_anomaly(years: np.ndarray, anomaly: np.ndarray, baseline_start: int = 1850, baseline_end: int = 1900) -> np.ndarray:
    """Rebase a temperature-anomaly series to a selected reference period."""

    years   = np.asarray(years)
    anomaly = np.asarray(anomaly, dtype=float)

    if years.shape != anomaly.shape:
        raise ValueError("Years and anomalies must have the same shape.")

    baseline_mask = (years >= baseline_start) & (years <= baseline_end)
    if not np.any(baseline_mask):
        raise ValueError("The selected baseline period is not contained in the data.")

    return anomaly - np.mean(anomaly[baseline_mask])


def estimate_arithmetic_brownian_parameters(years: np.ndarray, anomaly: np.ndarray) -> tuple[float, float]:
    """Estimate annual drift and volatility from consecutive annual changes."""

    years   = np.asarray(years)
    anomaly = np.asarray(anomaly, dtype=float)

    if years.shape != anomaly.shape:
        raise ValueError("Years and anomalies must have the same shape.")
    if len(years) < 3:
        raise ValueError("At least three annual observations are required.")
    if not np.allclose(np.diff(years), 1):
        raise ValueError("The observations must be consecutive annual values.")

    annual_changes = np.diff(anomaly)
    drift          = np.mean(annual_changes)
    volatility     = np.std(annual_changes - drift, ddof=1)

    return drift, volatility


def build_climate_scenario_mean(forecast_years: np.ndarray, initial_anomaly: float, scenario_values: np.ndarray) -> np.ndarray:
    """Interpolate an illustrative annual mean path through the IPCC period midpoints."""

    forecast_years = np.asarray(forecast_years)
    scenario_values = np.asarray(scenario_values, dtype=float)

    if scenario_values.shape != (3,):
        raise ValueError("Scenario values must contain the near-, mid-, and long-term central estimates.")
    if forecast_years[0] != 2020:
        raise ValueError("The forecast must start in 2020.")

    anchor_years  = np.array([2020, 2030, 2050, 2090, 2100])
    anchor_values = np.array([initial_anomaly, *scenario_values, scenario_values[-1]])

    return np.interp(forecast_years, anchor_years, anchor_values)
