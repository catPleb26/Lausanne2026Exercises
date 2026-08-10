import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def generate_strategies(simulate_wealth_handle, abs_risk_aversion: float, market_params: FinancialMarket, sde_params: SDESimulationParameters) -> list[tuple]:
    """
    Generate a list of tuples containing information about optimal and suboptimal investment strategies.
    """

    # WLOG x = 1
    x = 1

    # Optimal and suboptimal constant monetary investments
    xi_star = market_params.drift / (abs_risk_aversion * market_params.volatility**2)
    xi_sub  = 2 * xi_star

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


def plot_SDE_paths(simulated: SDEOutput, plot_component: str = 'W', num_plot_paths: int = 250, title: str = "Simulated SDE Paths", xlabel: str = "Time", ylabel: str = "Value", theoretical_distribution: callable = None, theoretical_mean: float = None):
    """
    Plot the simulated paths of a stochastic differential equation (SDE).

    Args:
        simulated (SDEOutput): An instance of SDEOutput containing the time grid and simulated paths.
        plot_component (str): The component of the SDE to plot. [Default: 'W']
        num_plot_paths (int): The number of paths to plot. [Default: 250]
        title (str): The title of the plot. [Default: "Simulated SDE Paths"]
        xlabel (str): The label for the x-axis. [Default: "Time"]
        ylabel (str): The label for the y-axis. [Default: "Value"]
        theoretical_distribution (callable): A function representing the theoretical distribution to overlay on the histogram. [Default: None]
        theoretical_mean (float): The theoretical mean to overlay on the histogram. [Default: None]
    """

    if theoretical_distribution is not None and theoretical_mean is None:
        raise ValueError("If a theoretical distribution is provided, the theoretical mean must also be provided.")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, (1, 3))

    num_paths_to_plot = min(num_plot_paths, simulated.paths[plot_component].shape[1])

    for i in range(num_paths_to_plot):
        plt.plot(simulated.time_grid, simulated.paths[plot_component][:, i], lw=0.5, alpha=0.3, c='k')
    plt.plot(simulated.time_grid, np.mean(simulated.paths[plot_component], axis=1), lw=2, c='r', label='Mean Path')

    plt.xlabel(xlabel)
    plt.xlim(0, simulated.time_grid[-1])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.legend()

    ylims = plt.ylim()

    query_points = np.linspace(ylims[0], ylims[1], 100)

    plt.subplot(1, 4, 4)
    plt.hist(simulated.paths[plot_component][-1, :], bins=30, density=True, alpha=0.3, color='k', edgecolor='black', orientation='horizontal', label='Simulated')
    if theoretical_distribution:
        plt.plot(theoretical_distribution(query_points),
                query_points,
                lw=2, c='b', label='Theoretical')

        plt.axhline(y=theoretical_mean, color='r', linestyle='--', lw=2, label='Mean')
        plt.xlim(0, 1.5*theoretical_distribution(theoretical_mean))

    plt.ylabel(ylabel)
    plt.ylim(ylims)
    plt.gca().yaxis.set_visible(False)
    plt.title("Distribution of\nTerminal Values")
    plt.grid()
    plt.legend()

    plt.show()


def compare_paths(simulated: list[SDEOutput], plot_component: str = 'W', num_plot_paths: int = 250, title: str = "Simulated SDE Paths", xlabel: str = "Time", ylabel: str = "Value", colororder: list[str] = None, labels: list[str] = None):
    """
    Compare multiple simulated paths of an SDE and plot them.

    Parameters:
    -----------
        simulated (list[SDEOutput]): A list of SDEOutput objects to compare.
        plot_component (str): The component of the SDE to plot (default is 'W').
        num_plot_paths (int): The number of paths to plot (default is 250).
        title (str): The title of the plot (default is "Simulated SDE Paths").
        xlabel (str): The label for the x-axis (default is "Time").
        ylabel (str): The label for the y-axis (default is "Value").
        colororder (list[str]): A list of colors to use for the plots. If None, default colors are used.
        labels (list[str]): A list of labels for the plots. If None, default labels are used.
    """
    plt.figure(figsize=(12, 6))
    
    if colororder is None:
        colororder = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, sim in enumerate(simulated):
        if i >= len(colororder):
            break  # Avoid index error if there are more simulations than colors
        plt.plot(sim.time_grid, sim.paths[plot_component][:,:num_plot_paths], color=colororder[i], alpha=0.7)

    # create phantom lines for legend
    for i, sim in enumerate(simulated):
        if labels is None:
            plt.plot([], [], color=colororder[i], label=f"Simulation {i+1}")
        else:
            plt.plot([], [], color=colororder[i], label=labels[i])
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()


def visualize_degenerate_paths(simulate_market_handle, M: int = 1000, M_inner: int = 100, drift_range: tuple = (-1, 1), vola_range: tuple = (0, 1), dt_range: tuple = (1e-3, 1)):
    """
    Visualize degenerate paths in a 3D scatter plot based on randomly sampled (drift, volatility, dt) values.
    A data point is created in the scatter plot if, at any time, the stock price falls below 0.
    """
    degenerate_points = []
    percentage_degenerate = []

    np.random.seed(42)  # For reproducibility
    drift_vals = np.random.uniform(*drift_range, size=M)
    vola_vals = np.random.uniform(*vola_range, size=M)
    dt_vals = np.random.uniform(*dt_range, size=M)

    for i in range(M):
        # Randomly sample drift, volatility, and dt values
        drift = drift_vals[i]
        vola = vola_vals[i]
        dt = dt_vals[i]

        print(f"Simulating for Drift: {drift:.4f}, Volatility: {vola:.4f}, dt: {dt:.6f} (Simulation {i+1}/{M})\r", end="")

        # Simulate M_inner paths for the sampled parameters
        market_params = FinancialMarket(risk_free_rate=0, risk_premium=drift, volatility=vola)
        sde_params = SDESimulationParameters(time_horizon=1, time_steps=int(1/dt), num_paths=M_inner)
        out = simulate_market_handle(initial_stock_price=1, market_params=market_params, sde_params=sde_params)

        # Check if any path falls below 0
        if (out.paths['stock'] < 0).any():
            degenerate_points.append((drift, vola, dt))

            p = np.mean(np.any(out.paths['stock'] < 0, axis=0))
            percentage_degenerate.append(p)

    degenerate_points = np.array(degenerate_points)
    sizes = np.array(percentage_degenerate) * 100  # Scale for better visualization
    colors = np.array(percentage_degenerate)

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    CLIM_UPPER = 0.5

    # plot projections onto the three planes
    ax.scatter(degenerate_points[:, 0], degenerate_points[:, 1], dt_range[0]*np.ones_like(degenerate_points[:, 2]), c=colors, marker='.', alpha=0.3, vmin=0, vmax=CLIM_UPPER, label='Projection on Drift-Volatility Plane')
    ax.scatter(degenerate_points[:, 0], vola_range[1]*np.ones_like(degenerate_points[:, 1]), degenerate_points[:, 2], c=colors, marker='.', alpha=0.3, vmin=0, vmax=CLIM_UPPER, label='Projection on Drift-dt Plane')
    ax.scatter(drift_range[0]*np.ones_like(degenerate_points[:, 0]), degenerate_points[:, 1], degenerate_points[:, 2], c=colors, marker='.', alpha=0.3, vmin=0, vmax=CLIM_UPPER, label='Projection on Volatility-dt Plane')

    scatter = ax.scatter(degenerate_points[:, 0], degenerate_points[:, 1], degenerate_points[:, 2], c=colors, s=sizes, cmap='viridis', alpha=0.6, vmin=0, vmax=CLIM_UPPER)
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Percentage of Degenerate Paths')
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{val:.0%}"))

    ax.set_xlabel('Drift')
    ax.set_ylabel('Volatility')
    ax.set_zlabel('dt')
    ax.set_title('Degenerate Paths in Stock Price Dynamics')

    plt.show()  


def plot_expected_utility_mc(abs_risk_aversion: float, runs: int, market_params: FinancialMarket):
    """
    Plot the expected utility of terminal wealth for different constant investment strategies using Monte Carlo simulation.
    """

    # WLOG T = 1 and 
    T = 1.0
    x = 1.0

    xi_star = market_params.drift / (abs_risk_aversion * market_params.volatility**2)
    xi_sub  = 2 * xi_star

    # Values of xi to compare
    xi_width  = max(2.5 * abs(xi_star), 2.0)
    xi_values = np.linspace(xi_star - xi_width, xi_star + xi_width, 51)

    # Compute the terminal Brownian motion values for all 
    np.random.seed(42)  # For reproducibility

    W_T = np.random.normal(loc=0, scale=np.sqrt(T), size=runs)

    # Terminal wealth and utility for every xi using exact solution for vectorized computation
    X_T = x + xi_values[:, np.newaxis] * (market_params.drift * T + market_params.volatility * W_T[np.newaxis, :])

    U_T = exponential_utility(X_T, abs_risk_aversion)

    # Monte Carlo statistics
    utility_statistics = compute_test_statistics(U_T, axis=1)

    mc_mean = utility_statistics.mean
    margin  = 1.96* utility_statistics.std_dev/np.sqrt(runs)

    xi_fine = np.linspace(xi_values.min(), xi_values.max(), 500)

    plt.figure(figsize=(12, 7))

    plt.plot(xi_fine, expected_utility_exact(x, T, xi_fine, abs_risk_aversion, market_params), color="b", lw=2.5, label="Analytical expected utility")
    plt.errorbar(xi_values, mc_mean, yerr=margin, fmt="o", markersize=3,color="k", ecolor="k", elinewidth=0.8, capsize=2, alpha=0.8, label="Monte Carlo mean with 95% CI")
    plt.axvline(xi_star, color="b", linestyle="--", lw=2, label=rf"Optimal $\xi^\ast={xi_star:.2f}$")
    plt.axvline(xi_sub, color="r", linestyle=":", lw=2, label=rf"Suboptimal $\xi={xi_sub:.2f}$")
    plt.scatter(xi_star, expected_utility_exact(x, T, xi_star, abs_risk_aversion, market_params), color="b", s=60, zorder=5)
    plt.annotate("Maximum", xy=(xi_star, expected_utility_exact(x, T, xi_star, abs_risk_aversion, market_params)), xytext=(10, 12), textcoords="offset points", color="b")

    plt.xlabel(r"Constant amount invested in the stock, $\xi$")
    plt.ylabel(r"$\mathbb{E}[1-\exp(-\lambda X_T^\xi)]$")
    plt.title("Expected terminal utility for different constant strategies")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.show()


def plot_strategies(strategies: list[tuple]):
    """
    Plot the wealth paths and mean utility for optimal and suboptimal investment strategies as computed by the `generate_strategies` function.
    """

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

    # Common right-axis scale
    lower_U = np.array([strategy[5] for strategy in strategies])
    upper_U = np.array([strategy[6] for strategy in strategies])

    utility_min = lower_U.min()
    utility_max = upper_U.max()
    utility_padding = max(0.05 * (utility_max - utility_min), 1e-3)

    num_paths_to_plot = 25

    for ax, strategy in zip(axes, strategies):
        (name, xi, t, wealth, mean_utility, lower_utility, upper_utility, analytical_mean, color) = strategy

        # Left axis: wealth paths
        ax.plot(t, wealth[:, :num_paths_to_plot], color=color, lw=0.8, alpha=0.18)

        ax.plot(t, np.mean(wealth, axis=1), color=color, lw=2.5, label="Mean wealth")

        ax.axhline(0, color="grey", lw=0.8, alpha=0.5)
        ax.set_xlabel("Time")
        ax.set_ylabel("Wealth paths")
        ax.set_title(rf"{name}: $\xi={xi:.2f}$")
        ax.grid(alpha=0.2)

        # Right axis: mean utility
        utility_axis = ax.twinx()

        utility_axis.fill_between(t, lower_utility, upper_utility, color="mediumpurple", alpha=0.15, label="95% CI")
        utility_axis.plot(t, mean_utility, color="mediumpurple", lw=2.5, label="Monte Carlo mean utility")
        utility_axis.plot(t, analytical_mean, color="black", linestyle="--", lw=1.5, label="Analytical mean utility")

        utility_axis.set_ylabel("Mean utility")
        utility_axis.set_ylim(utility_min - utility_padding, utility_max + utility_padding)

        left_lines, left_labels = ax.get_legend_handles_labels()
        right_lines, right_labels = utility_axis.get_legend_handles_labels()

        utility_axis.legend(left_lines + right_lines, left_labels + right_labels, loc="upper left", fontsize=9)

    fig.suptitle("Optimal versus suboptimal constant investment strategy", fontsize=15)

    plt.tight_layout()
    plt.show()


def plot_wealth_consumption_policy(generate_deflator_H_handle, market_params: FinancialMarket, sde_params: SDESimulationParameters):
    """
    Plot optimal wealth, consumption, and policy rules.
    """

    # WLOG x = 1
    x = 1

    out = generate_deflator_H_handle(market_params, sde_params)
    t = out.time_grid
    H = out.paths['H']

    # Optimal wealth and consumption
    X_star = x * (2 - t[:, np.newaxis]) / (2 * H)
    c_star = x / (2 * H)

    consumption_wealth_rate = 1 / (2 - t)

    # Optimal policy rules
    pi_star                 = market_params.risk_premium / market_params.volatility**2

    print(f"Optimal stock fraction: {pi_star:.2%}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    num_paths_to_plot = 25

    # 1. Optimal wealth
    axes[0].plot(t, X_star[:, :num_paths_to_plot], color="b", alpha=0.18, lw=0.8)
    axes[0].plot(t, np.mean(X_star, axis=1), color="b", lw=2.5, label="Monte Carlo mean")

    axes[0].set_title("Optimal wealth")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel(r"$X_t^\ast$")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    # 2. Optimal consumption
    axes[1].plot(t, c_star[:, :num_paths_to_plot], color="orange", alpha=0.18, lw=0.8)

    axes[1].plot(t, np.mean(c_star, axis=1), color="orange", lw=2.5, label="Monte Carlo mean")

    axes[1].set_title("Optimal consumption")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel(r"$c_t^\ast$")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    # 3. Optimal policy rules
    policy_axis = axes[2]

    policy_axis.plot(t, consumption_wealth_rate, color="g", lw=2.5, label=r"$c_t^\ast/X_t^\ast$")

    policy_axis.scatter([0, 1], [consumption_wealth_rate[0], consumption_wealth_rate[-1]], color="g", zorder=5)

    policy_axis.set_xlabel("Time")
    policy_axis.set_ylabel("Consumption-to-wealth rate", color="g")
    policy_axis.tick_params(axis="y", labelcolor="g")
    policy_axis.set_title("Optimal policy rules")
    policy_axis.grid(alpha=0.2)

    portfolio_axis = policy_axis.twinx()

    portfolio_axis.axhline(pi_star, color="r", linestyle="--", lw=2.5, label=rf"$\pi^\ast={100*pi_star:.2f}\%$")

    portfolio_axis.set_ylabel("Fraction invested in stock", color="r")
    portfolio_axis.tick_params(axis="y", labelcolor="r")

    # Combined legend for the third panel
    left_lines, left_labels = policy_axis.get_legend_handles_labels()
    right_lines, right_labels = portfolio_axis.get_legend_handles_labels()

    policy_axis.legend(left_lines + right_lines, left_labels + right_labels, loc="upper left")

    fig.suptitle("Log-optimal wealth, consumption, and portfolio strategy", fontsize=15)

    plt.tight_layout()
    plt.show()