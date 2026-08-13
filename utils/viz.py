from .common import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import ipywidgets as widgets
from IPython.display import display


###
###
###
### Very general plotting functions
###
###
###

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

###
###
###
### DAY 1
###
###
###

def visually_explore_brownian_motion(generate_brownian_path_handle):
    mu_slider = widgets.FloatSlider(
        value=0.2,
        min=-1.0,
        max=1.0,
        step=0.05,
        description=r"$\mu$:",
        continuous_update=True,
    )

    sigma_slider = widgets.FloatSlider(
        value=0.3,
        min=0.05,          # Avoid division by zero
        max=1.0,
        step=0.05,
        description=r"$\sigma$:",
        continuous_update=True,
    )

    Msub_slider = widgets.IntSlider(
        value=25,
        min=1,    
        max=500,
        step=1,
        description=r"#Paths2Plot:",
        continuous_update=True,
    )

    def update_plot(mu, sigma, Msub):
        # Use M = 2500, T = 1
        M = 2_500
        T = 1

        # Regenerate the paths with the selected parameters
        test_params = SDESimulationParameters(time_horizon=T, time_steps=int(250*T), num_paths=M)
        out = generate_brownian_path_handle(mu=mu, sigma=sigma, params=test_params)

        # W_T ~ Normal(mu*T, sigma^2*T)
        theoretical_distribution = lambda x: (
            1 / (sigma * np.sqrt(2 * np.pi * T))
            * np.exp(-0.5 * ((x - mu * T) / (sigma * np.sqrt(T))) ** 2)
        )

        theoretical_mean = mu * T

        plot_SDE_paths(
            out,
            plot_component="W",
            num_plot_paths=Msub,
            title=rf"Brownian Motion: $\mu={mu:.2f}$, $\sigma={sigma:.2f}$",
            xlabel="Time",
            ylabel="Value",
            theoretical_distribution=theoretical_distribution,
            theoretical_mean=theoretical_mean,
        )

    interactive_plot = widgets.interactive_output(update_plot, {"mu": mu_slider, "sigma": sigma_slider, "Msub": Msub_slider})

    display(widgets.HBox([mu_slider, sigma_slider, Msub_slider]), interactive_plot)


def visually_explore_stock_prices(simulate_market_handle):
    b_slider = widgets.FloatSlider(
        value=0.08,
        min=-1.0,
        max=1.0,
        step=0.05,
        description=r"$b$:",
        continuous_update=True,
    )

    sigma_slider = widgets.FloatSlider(
        value=0.25,
        min=0.05,          # Avoid division by zero
        max=1.0,
        step=0.05,
        description=r"$\sigma$:",
        continuous_update=True,
    )

    Msub_slider = widgets.IntSlider(
        value=25,
        min=1,    
        max=500,
        step=1,
        description=r"#Paths2Plot:",
        continuous_update=True,
    )

    def update_plot(b, sigma, Msub):
        # Use M = 5000, T = 5, r = 0, S0 = 100
        M  = 2_500 
        T  = 5
        r  = 0
        S0 = 100

        # S_T ~ LogNormal(ln(S0) + (mu - o^2/2)*T, sigma^2*T)
        theoretical_mean         = S0 * np.exp(b*T)
        theoretical_distribution = lambda x: (
            lognormal_pdf(x, mu=np.log(S0) + (b - 0.5*sigma**2)*T, sigma=sigma*np.sqrt(T))
        )

        market_params = FinancialMarket(risk_free_rate=r, risk_premium=b - r, volatility=sigma)
        sde_params = SDESimulationParameters(time_horizon=T, time_steps=int(250*T), num_paths=M)
        out = simulate_market_handle(initial_stock_price=S0, market_params=market_params, sde_params=sde_params)

        plot_SDE_paths(
            out, 
            plot_component='stock', 
            num_plot_paths=Msub, 
            title="Simulated Stock Price Paths", 
            xlabel="Time", 
            ylabel="Value", 
            theoretical_distribution=theoretical_distribution, 
            theoretical_mean=theoretical_mean
        )

    box = widgets.HBox([b_slider, sigma_slider, Msub_slider])
    interactive_plot = widgets.interactive_output(update_plot, {"b": b_slider, "sigma": sigma_slider, "Msub": Msub_slider})

    display(box, interactive_plot)

###
###
###

def show_euler_maruyama_breakdown(simulate_market_handle, simulate_market_dynamics_exact_handle):
    b_slider = widgets.FloatSlider(
        value=-1,
        min=-1.0,
        max=1.0,
        step=0.05,
        description=r"$b$:",
        continuous_update=True,
    )

    sigma_slider = widgets.FloatSlider(
        value=0.75,
        min=0.05,          # Avoid division by zero
        max=1.0,
        step=0.05,
        description=r"$\sigma$:",
        continuous_update=True,
    )

    nsteps_slider = widgets.IntSlider(
        value=3,
        min=1,
        max=50,
        step=1,
        description=r"#TimeSteps:",
        continuous_update=True,
    )

    def update_plot(b, sigma, nSteps):
        # Use M = 50, T = 1, r = 0, S0 = 100
        M  = 50
        T  = 1
        r  = 0
        S0 = 100

        market_params = FinancialMarket(risk_free_rate=0, risk_premium=b, volatility=sigma)
        sde_params    = SDESimulationParameters(time_horizon=T, time_steps=nSteps, num_paths=M)
        
        probability_negative_step = norm.cdf(- (1 + b * T/nSteps) / (sigma * np.sqrt(T/nSteps)))
        
        title = rf"""Comparison of EM Scheme vs Exact Solution for Stock Prices
        Probability of negative stock price for b={b:.2f}, $\sigma$={sigma:.2f}, dt={sde_params.dt:.3f}: {100 * probability_negative_step: .2f}%"""
        
        out_em    = simulate_market_handle(initial_stock_price=S0, market_params=market_params, sde_params=sde_params)
        out_exact = simulate_market_dynamics_exact_handle(initial_stock_price=S0, market_params=market_params, sde_params=sde_params)
        
        compare_paths(
            [out_em, out_exact], 
            plot_component='stock', 
            num_plot_paths=M, 
            title=title, 
            xlabel="Time", 
            ylabel="Stock Price", 
            colororder=['blue', 'orange'], 
            labels=['EM Scheme', 'Exact Solution']
        )

    box = widgets.HBox([b_slider, sigma_slider, nsteps_slider])
    interactive_plot = widgets.interactive_output(update_plot, {"b": b_slider, "sigma": sigma_slider, "nSteps": nsteps_slider})

    display(box, interactive_plot)

###
###
###

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

###
###
###

def compare_investment_strategies(simulate_wealth_handle):
    r_slider = widgets.FloatSlider(
        value=0.025,
        min=-0.1,
        max=0.1,
        step=0.005,
        description=r"$r$:",
        continuous_update=True,
    )

    b_slider = widgets.FloatSlider(
        value=0.08,
        min=-0.2,
        max=0.2,
        step=0.005,
        description=r"$b$:",
        continuous_update=True,
    )

    sigma_slider = widgets.FloatSlider(
        value=0.25,
        min=0.05,          # Avoid division by zero
        max=0.5,
        step=0.005,
        description=r"$\sigma$:",
        continuous_update=True,
    )

    merton_check = widgets.Checkbox(
        value=True,
        description='Use growth optimal $\\pi$ (gof)',
    )

    def update_plot(r, b, sigma, use_merton_fraction):
        # Use M = 10, T = 25, X0 = 100
        M  = 10
        T  = 25
        X0 = 100

        market_params = FinancialMarket(risk_free_rate=r, risk_premium=b - r, volatility=sigma)
        sde_params = SDESimulationParameters(time_horizon=T, time_steps=250*T, num_paths=M)
        
        if use_merton_fraction:
            pi = (b - r) / (sigma**2)
        else:
            pi = 0.67

        title = rf"""Comparison of Constant Fraction vs Constant Nominal Investment Strategies
        r = {r:.3f} b = {b:.3f} $\sigma$ = {sigma:.3f}
        Using $\pi = {pi:.3f}${" (gof)" if use_merton_fraction else ""} and $\xi = {pi*X0:.2f}$"""
        
        out_constant_fraction = simulate_wealth_handle(initial_wealth=X0, constant_investment=pi, investment_strategy=InvestmentType.ConstantFraction, market_params=market_params, sde_params=sde_params)
        out_constant_nominal  = simulate_wealth_handle(initial_wealth=X0, constant_investment=pi*X0, investment_strategy=InvestmentType.ConstantNominal, market_params=market_params, sde_params=sde_params)
        
        compare_paths(
            [out_constant_fraction, out_constant_nominal],
            plot_component='X',
            num_plot_paths=M,
            title=title,
            xlabel="Time",
            ylabel="Wealth",
            colororder=['blue', 'orange'],
            labels=['Constant Fraction', 'Constant Nominal']
        )

    box = widgets.VBox([widgets.HBox([r_slider, b_slider, sigma_slider]), merton_check])
    interactive_plot = widgets.interactive_output(update_plot, {"r": r_slider, "b": b_slider, "sigma": sigma_slider, "use_merton_fraction": merton_check})

    display(box, interactive_plot)

###
###
###

def explore_parameter_impact_on_stock_prices(simulate_market_handle):
    b_1_slider = widgets.FloatSlider(
        value=-0.1,
        min=-0.2,
        max=0.2,
        step=0.005,
        description=r"$b_1$:",
        continuous_update=True,
    )

    b_2_slider = widgets.FloatSlider(
        value=0.1,
        min=-0.2,
        max=0.2,
        step=0.005,
        description=r"$b_2$:",
        continuous_update=True,
    )

    sigma_1_slider = widgets.FloatSlider(
        value=0.2,
        min=0.05,
        max=0.5,
        step=0.005,
        description=r"$\sigma_1$:",
        continuous_update=True,
    )

    sigma_2_slider = widgets.FloatSlider(
        value=0.4,
        min=0.05,
        max=0.5,
        step=0.005,
        description=r"$\sigma_2$:",
        continuous_update=True,
    )

    def update_plot(b_1, b_2, sigma_1, sigma_2):
        # Use M = 1, T = 10, r = 0, S0 = 100
        T  = 10
        r  = 0
        S0 = 100

        # compare paths for different pairs of (drift, volatility) values
        params_list = [
            (b_1, sigma_1),
            (b_2, sigma_1),
            (b_1, sigma_2),
            (b_2, sigma_2)
        ]

        out_list = []
        for drift, vola in params_list:
            market_params = FinancialMarket(risk_free_rate=r, risk_premium=drift, volatility=vola)
            sde_params = SDESimulationParameters(time_horizon=T, time_steps=int(T*250), num_paths=1)
            out = simulate_market_handle(initial_stock_price=S0, market_params=market_params, sde_params=sde_params)
            out_list.append(out)

        compare_paths(
            out_list, 
            plot_component='stock',
            num_plot_paths=1,
            title="Comparison of Stock Price Dynamics for Different (Drift, Volatility) Pairs",
            xlabel="Time",
            ylabel="Stock Price",
            colororder=['blue', 'orange', 'green', 'red'],
            labels=[f"Drift: {drift:.3f}, Volatility: {vola:.3f}" for drift, vola in params_list]
        )

    box = widgets.VBox([widgets.HBox([b_1_slider, b_2_slider]), widgets.HBox([sigma_1_slider, sigma_2_slider])])
    interactive_plot = widgets.interactive_output(update_plot, {"b_1": b_1_slider, "b_2": b_2_slider, "sigma_1": sigma_1_slider, "sigma_2": sigma_2_slider})

    display(box, interactive_plot)


def explore_investment_fraction_impact_on_wealth(simulate_wealth_handle):
    custom_fractions = widgets.FloatSlider(
        value=0.67,
        min=-1,
        max=2,
        step=0.01,
        description=r"$\pi$:",
        continuous_update=True
    )

    def update_plot(pi):
        # Use M = 1, T = 10, r = 0.025, b = 0.08, sigma = 0.25, S0 = 100
        T  = 10
        r  = 0.025
        b = 0.08
        sigma = 0.25
        S0 = 100

        # compare wealth paths for different values of pi
        values = np.array([0, 0.25, 0.5, 0.75, 1.0, pi])
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'k']

        idx = np.argsort(values)
        risky_fractions = values[idx]
        colors = [colors[i] for i in idx]

        out_list = []
        for pi in risky_fractions:
            market_params = FinancialMarket(risk_free_rate=r, risk_premium=b-r, volatility=sigma)
            sde_params = SDESimulationParameters(time_horizon=T, time_steps=int(T*250), num_paths=1)
            out = simulate_wealth_handle(initial_wealth=S0, constant_investment=pi, investment_strategy=InvestmentType.ConstantFraction, market_params=market_params, sde_params=sde_params)
            out_list.append(out)

        compare_paths(
            out_list,
            plot_component='X',
            num_plot_paths=1,
            title="Comparison of Wealth Dynamics for Different Investment Fractions",
            xlabel="Time",
            ylabel="Wealth",
            colororder=colors,
            labels=[f"Investment Fraction: {pi:.3f}" for pi in risky_fractions]
        )

    box = widgets.HBox([custom_fractions])
    interactive_plot = widgets.interactive_output(update_plot, {"pi": custom_fractions})
    display(box, interactive_plot)

###
###
###

def plot_expected_utility_mc(abs_risk_aversion: float, runs: int, market_params: FinancialMarket):
    """
    Plot the expected utility of terminal wealth for different constant investment strategies using Monte Carlo simulation.
    """

    # WLOG T = 1 and x = 1
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
        ax.set_ylabel("Wealth paths", color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.set_title(rf"{name}: $\xi={xi:.2f}$")
        ax.grid(alpha=0.2)

        # Right axis: mean utility
        utility_axis = ax.twinx()

        utility_axis.fill_between(t, lower_utility, upper_utility, color="mediumpurple", alpha=0.15, label="95% CI")
        utility_axis.plot(t, mean_utility, color="mediumpurple", lw=2.5, label="Monte Carlo mean utility")
        utility_axis.plot(t, analytical_mean, color="black", linestyle="--", lw=1.5, label="Analytical mean utility")

        utility_axis.set_ylabel("Mean utility", color="mediumpurple")
        utility_axis.tick_params(axis="y", labelcolor="mediumpurple")
        utility_axis.set_ylim(utility_min - utility_padding, utility_max + utility_padding)

        left_lines, left_labels = ax.get_legend_handles_labels()
        right_lines, right_labels = utility_axis.get_legend_handles_labels()

        utility_axis.legend(left_lines + right_lines, left_labels + right_labels, loc="upper left", fontsize=9)

    fig.suptitle("Optimal versus suboptimal constant investment strategy", fontsize=15)

    plt.tight_layout()
    plt.show()

def optimal_vs_suboptimal_plot(simulate_wealth_handle):
    xi_sub_slider = widgets.FloatSlider(
        value=3.2,
        min=-4,
        max=12,
        step=0.1,
        description=r"$\xi_{\mathrm{sub}}$:",
        continuous_update=True
    )

    def update_plot(xi_sub):
        # Use M = 5000, T = 10, r = 0, b = 0.05, sigma = 0.25, x = 1, abs_risk_aversion = 0.25
        M                 = 5_000
        T                 = 10
        r                 = 0.0
        b                 = 0.05
        sigma             = 0.25
        x                 = 1.0
        abs_risk_aversion = 0.25

        market_params = FinancialMarket(risk_free_rate=r, risk_premium=b-r, volatility=sigma)
        sde_params = SDESimulationParameters(time_horizon=T, time_steps=int(250*T), num_paths=M)

        strategies = generate_strategies(simulate_wealth_handle, xi_sub, abs_risk_aversion, market_params=market_params, sde_params=sde_params)

        plot_strategies(strategies)

    box = widgets.HBox([xi_sub_slider])
    interactive_plot = widgets.interactive_output(update_plot, {"xi_sub": xi_sub_slider})
    display(box, interactive_plot)

###
###
###

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

###
###
###
### DAY 2
###
###
###

def explore_continuous_vs_buy_and_hold(simulate_strategy_handle):
    """Interactively compare continuous rebalancing with buy-and-hold."""

    r_slider = widgets.FloatSlider(
        value=0.02,
        min=0.0,
        max=0.08,
        step=0.005,
        description=r"$r$:",
        continuous_update=True,
    )

    b_slider = widgets.FloatSlider(
        value=0.08,
        min=0.02,
        max=0.18,
        step=0.005,
        description=r"$b$:",
        continuous_update=True,
    )

    sigma_slider = widgets.FloatSlider(
        value=0.30,
        min=0.15,
        max=0.60,
        step=0.01,
        description=r"$\sigma$:",
        continuous_update=True,
    )

    path_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=49,
        step=1,
        description="Path:",
        continuous_update=True,
    )

    def update_plot(r, b, sigma, path_index):
        # Use X0 = 250, S0 = 100, and a small number of horizons for clarity
        X0 = 250
        S0 = 100
        horizons = np.array([0.1, 0.4, 1.0, 2.0, 5.0])

        T = horizons[-1]

        market  = FinancialMarket(risk_free_rate=r, risk_premium=b - r, volatility=sigma)
        pi_star = market.risk_premium / market.volatility**2

        if not 0 <= pi_star <= 1:
            print(
                rf"The selected parameters give $\pi^\ast={pi_star:.2f}$. "
                "For this visualization, choose parameters with "
                rf"$0\leq\pi^\ast\leq1$ so buy-and-hold wealth remains positive."
            )
            return

        params = SDESimulationParameters(
            time_horizon=T,
            time_steps=int(250 * T),
            num_paths=50,
        )

        out = simulate_strategy_handle(
            initial_wealth=X0,
            initial_stock_price=S0,
            market_params=market,
            sde_params=params,
        )

        t = out.time_grid
        X_continuous = out.paths["continuous_rebalancing"]
        X_buy_hold   = out.paths["buy_and_hold"]
        pi_buy_hold  = out.paths["buy_and_hold_fraction"]

        # A separate terminal-only simulation gives precise utility estimates
        # without storing a very large number of complete paths. We do not use
        # our previous simulation function for vectorized computation
        utility_runs = 100_000
        rng = np.random.default_rng(12345)
        horizon_increments = np.diff(np.concatenate(([0.0], horizons)))
        terminal_W = np.cumsum(
            rng.normal(scale=np.sqrt(horizon_increments)[:, np.newaxis], size=(len(horizons), utility_runs)),
            axis=0,
        )

        paired_means = []
        paired_margins = []
        for horizon, W_horizon in zip(horizons, terminal_W):
            continuous_terminal = X0 * np.exp(
                (r + pi_star * market.risk_premium - 0.5 * pi_star**2 * sigma**2) * horizon
                + pi_star * sigma * W_horizon
            )
            stock_ratio = np.exp(
                (b - 0.5 * sigma**2) * horizon + sigma * W_horizon
            )
            buy_hold_terminal = X0 * (
                (1 - pi_star) * np.exp(r * horizon)
                + pi_star * stock_ratio
            )
            paired_difference = (
                np.log(continuous_terminal) - np.log(buy_hold_terminal)
            )
            paired_means.append(np.mean(paired_difference))
            paired_margins.append(
                1.96 * np.std(paired_difference) / np.sqrt(utility_runs)
            )

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

        axes[0].plot(t, X_continuous[:, path_index], color="b", lw=2.2, label="Continuous rebalancing")
        axes[0].plot(t, X_buy_hold[:, path_index], color="orange", lw=2.2, label="Buy-and-hold")
        for horizon in horizons:
            axes[0].axvline(horizon, color="grey", lw=0.8, alpha=0.35)
        axes[0].set_title("Same market path")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Wealth")
        axes[0].grid(alpha=0.25)
        axes[0].legend()

        axes[1].plot(t, pi_buy_hold[:, path_index], color="orange", lw=2.2, label="Buy-and-hold fraction")
        axes[1].axhline(pi_star, color="b", linestyle="--", lw=2.2, label=rf"Target $\pi^\ast={100 * pi_star:.2f}\%$")
        axes[1].set_title("The buy-and-hold weight drifts")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Fraction invested in stock")
        axes[1].set_ylim(0, 1)
        axes[1].grid(alpha=0.25)
        axes[1].legend()

        axes[2].errorbar(
            np.arange(len(horizons)),
            paired_means,
            yerr=paired_margins,
            fmt="o",
            color="mediumpurple",
            ecolor="mediumpurple",
            capsize=5,
            markersize=7,
            label="Paired Monte Carlo mean with 95% CI",
        )
        axes[2].axhline(0, color="black", lw=1.0)
        axes[2].set_xticks(np.arange(len(horizons)))
        axes[2].set_xticklabels([str(horizon) for horizon in horizons])
        axes[2].set_title("Expected log-utility advantage")
        axes[2].set_xlabel("Maturity $T$")
        axes[2].set_ylabel(r"$\mathbb{E}[\log X_T^{\mathrm{cont}}-\log X_T^{\mathrm{BH}}]$")
        axes[2].grid(alpha=0.25)
        axes[2].legend()

        fig.suptitle(rf"Continuous rebalancing versus buy-and-hold: "
            rf"$r={r:.3f}$, $b={b:.3f}$, $\sigma={sigma:.2f}$",
            fontsize=15,
        )
        plt.tight_layout()
        plt.show()

    controls = widgets.VBox([widgets.HBox([r_slider, b_slider, sigma_slider]), path_slider])
    interactive_plot = widgets.interactive_output(
        update_plot,
        {
            "r": r_slider,
            "b": b_slider,
            "sigma": sigma_slider,
            "path_index": path_slider,
        },
    )

    display(controls, interactive_plot)


def explore_cppi_gap_risk(simulate_cppi_handle):
    """Interactively illustrate discrete-time CPPI gap risk."""

    multiplier_slider = widgets.FloatSlider(
        value=5.0,
        min=1.0,
        max=12.0,
        step=0.5,
        description=r"$M$:",
        continuous_update=True,
    )

    rebalancing_slider = widgets.SelectionSlider(
        options=[1, 2, 4, 12, 26, 52, 250],
        value=12,
        description="Trades/year:",
        continuous_update=True,
    )

    cap_checkbox = widgets.Checkbox(
        value=False,
        description="Cap stock position at wealth",
    )

    def update_plot(multiplier, rebalances, cap_at_wealth):
        # Use M = 5000, T = 1, X0 = 100, and guarantee = 75
        M      = 5_000
        T         = 1.0
        X0        = 100
        guarantee = 75

        market = FinancialMarket(risk_free_rate=0.02, risk_premium=0.06, volatility=0.25)
        params = SDESimulationParameters(time_horizon=T, time_steps=int(rebalances * T), num_paths=M)

        out = simulate_cppi_handle(
            initial_wealth=X0,
            guarantee=guarantee,
            multiplier=multiplier,
            cap_at_wealth=cap_at_wealth,
            market_params=market,
            sde_params=params,
        )

        t = out.time_grid
        floor        = out.paths["floor"]
        X_continuous = out.paths["continuous_cppi"]
        X_discrete   = out.paths["discrete_cppi"]
        cap_active   = out.paths["cap_active"]

        shortfall = X_discrete - floor
        gap_paths = np.any(shortfall < -1e-10, axis=0)
        gap_probability = np.mean(gap_paths)
        stressed_path = int(np.argmin(np.min(shortfall, axis=0)))

        frequencies = np.array([1, 2, 4, 12, 26, 52, 250])
        gap_probabilities = []
        for frequency in frequencies:
            frequency_params = SDESimulationParameters(time_horizon=T, time_steps=int(frequency * T), num_paths=M)
            frequency_out = simulate_cppi_handle(
                initial_wealth=X0,
                guarantee=guarantee,
                multiplier=multiplier,
                cap_at_wealth=cap_at_wealth,
                market_params=market,
                sde_params=frequency_params,
            )
            frequency_gap = np.any(
                frequency_out.paths["discrete_cppi"] < frequency_out.paths["floor"] - 1e-10,
                axis=0,
            )
            gap_probabilities.append(np.mean(frequency_gap))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

        marker_idx = cap_active[:, stressed_path].nonzero()[0]

        axes[0].plot(t, floor[:, 0], color="black", linestyle="--", lw=2.0, label="Floor")
        axes[0].plot(t, X_continuous[:, stressed_path], color="b", lw=2.2, label="Continuous-time CPPI")
        axes[0].plot(t, X_discrete[:, stressed_path], color="orange", marker="o", markersize=3, lw=2.0, label="Discrete CPPI")
        if cap_at_wealth and marker_idx.size > 0:
            axes[0].scatter(t[marker_idx], X_discrete[marker_idx, stressed_path], color="red", marker="x", s=100, zorder=5, label="Cap active")
        axes[0].set_title("Most stressed simulated path")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Wealth")
        axes[0].grid(alpha=0.25)
        axes[0].legend()

        terminal_values = np.concatenate([X_continuous[-1, :], X_discrete[-1, :]])
        histogram_range = tuple(np.percentile(terminal_values, [0.5, 99.5]))

        axes[1].hist(X_continuous[-1, :],bins=40,range=histogram_range, density=True, alpha=0.45, color="b", label="Continuous-time CPPI")
        axes[1].hist(X_discrete[-1, :],bins=40,range=histogram_range, density=True, alpha=0.45, color="orange", label="Discrete CPPI")
        axes[1].axvline(guarantee, color="black", linestyle="--", lw=2.0)
        axes[1].set_title("Terminal wealth distribution (central 99%)")
        axes[1].set_xlabel(r"$X_T$")
        axes[1].set_ylabel("Density")
        axes[1].grid(alpha=0.25)
        axes[1].legend()

        axes[2].plot(frequencies, gap_probabilities, color="crimson", marker="o", lw=2.2)
        axes[2].scatter(rebalances,gap_probability, color="black", s=60, zorder=5, label="Selected frequency")
        axes[2].set_xscale("log")
        axes[2].set_xticks(frequencies)
        axes[2].set_xticklabels([str(value) for value in frequencies])
        axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, position: f"{value:.1%}"))
        axes[2].set_title("Estimated probability of a floor breach")
        axes[2].set_xlabel("Rebalancings per year")
        axes[2].set_ylabel("Gap probability")
        axes[2].grid(alpha=0.25)
        axes[2].legend()

        cap_text = "with cap" if cap_at_wealth else "without cap"
        fig.suptitle(
            rf"CPPI gap risk: $M={multiplier:.1f}$, {cap_text}; "
            rf"selected gap probability = {gap_probability:.2%}",
            fontsize=15,
        )
        plt.tight_layout()
        plt.show()

    controls = widgets.VBox([widgets.HBox([multiplier_slider, rebalancing_slider]), cap_checkbox])
    interactive_plot = widgets.interactive_output(
        update_plot,
        {
            "multiplier": multiplier_slider,
            "rebalances": rebalancing_slider,
            "cap_at_wealth": cap_checkbox,
        },
    )

    display(controls, interactive_plot)


def explore_log_vs_value_preserving(simulate_strategy_handle):
    """Compare log-optimal and value-preserving wealth and consumption."""

    r_slider = widgets.FloatSlider(
        value=0.01,
        min=0.0,
        max=0.05,
        step=0.005,
        description=r"$r$:",
        continuous_update=True,
    )

    b_slider = widgets.FloatSlider(
        value=0.05,
        min=0.01,
        max=0.15,
        step=0.005,
        description=r"$b$:",
        continuous_update=True,
    )

    sigma_slider = widgets.FloatSlider(
        value=0.20,
        min=0.10,
        max=0.50,
        step=0.01,
        description=r"$\sigma$:",
        continuous_update=True,
    )

    maturity_slider = widgets.SelectionSlider(
        options=[1, 5, 10, 20],
        value=10,
        description=r"$T$:",
        continuous_update=True,
    )

    path_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=49,
        step=1,
        description="Path:",
        continuous_update=True,
    )

    def update_plot(r, b, sigma, maturity, path_index):
        # Use M = 2500, X0 = 100
        M = 2500
        X0 = 100.0
        T = float(maturity)

        market = FinancialMarket(risk_free_rate=r, risk_premium=b - r, volatility=sigma)
        params = SDESimulationParameters(time_horizon=T, time_steps=int(5*T), num_paths=M)

        out = simulate_strategy_handle(initial_wealth=X0, market_params=market, sde_params=params)

        t = out.time_grid
        X_log = out.paths["log_wealth"]
        C_log = out.paths["log_cumulative_consumption"]
        X_vps = out.paths["vps_wealth"]
        C_vps = out.paths["vps_cumulative_consumption"]

        # accumulate the negative consumption to get the total personal investment over time
        personal_investment = -np.minimum(0, np.diff(C_vps, axis=0, prepend=0))
        cumulative_personal_investment = np.cumsum(personal_investment, axis=0)
        
        theta       = market.risk_premium / market.volatility
        growth_rate = r + theta**2

        analytical_log = np.array([X0 * np.exp(growth_rate * T) / (T + 1), X0 * exponential_integral(growth_rate, T) / (T + 1)])
        analytical_vps = np.array([X0 * np.exp(r * T), X0 * theta**2 * exponential_integral(r, T)])

        monte_carlo_log = np.array([np.mean(X_log[-1, :]), np.mean(C_log[-1, :])])
        monte_carlo_vps = np.array([np.mean(X_vps[-1, :]), np.mean(C_vps[-1, :])])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

        axes[0].plot(t, X_log[:, path_index], color="b", lw=2.2, label="Log-optimal wealth")
        axes[0].plot(t, X_vps[:, path_index], color="orange", linestyle="--", lw=2.2, label="Value-preserving wealth")
        axes[0].set_title("Wealth on the same market path")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Wealth")
        axes[0].grid(alpha=0.25)
        axes[0].legend()

        axes[1].plot(t, C_log[:, path_index], color="b", lw=2.2, label="Log-optimal consumption")
        axes[1].plot(t, C_vps[:, path_index], color="orange", linestyle="--", lw=2.2, label="Value-preserving consumption")
        axes[1].axhline(0, color="black", lw=0.8)
        axes[1].set_title("Cumulative consumption")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Cumulative amount")
        axes[1].grid(alpha=0.25)
        axes[1].legend()

        ylims = axes[1].get_ylim()
        ylims = (ylims[0], max(ylims[1], cumulative_personal_investment[:, path_index].max() * 1.05))

        consumption_axis = axes[1].twinx()
        consumption_axis.plot(t, cumulative_personal_investment[:, path_index], color="red", lw=1.5, label="Cumulative personal investment")
        consumption_axis.set_ylabel("Cumulative personal investment (injections)", color="red")
        consumption_axis.tick_params(axis="y", labelcolor="red")
        consumption_axis.set_ylim(ylims)

        locations = np.arange(2)
        width = 0.34
        axes[2].bar(locations - width / 2, monte_carlo_log, width, color="b", alpha=0.70, label="Log-optimal MC mean")
        axes[2].bar(locations + width / 2, monte_carlo_vps, width, color="orange", alpha=0.70, label="Value-preserving MC mean")
        axes[2].scatter(locations - width / 2, analytical_log, marker="_", s=900, linewidths=3, color="black", label="Analytical mean")
        axes[2].scatter(locations + width / 2, analytical_vps, marker="_", s=900, linewidths=3, color="black")
        axes[2].set_xticks(locations)
        axes[2].set_xticklabels(["Terminal wealth", "Consumption"])
        axes[2].set_title("Expected allocation by maturity")
        axes[2].set_ylabel("Expected amount")
        axes[2].grid(axis="y", alpha=0.25)
        axes[2].legend(fontsize=9)

        pi_star = market.risk_premium / market.volatility**2
        fig.suptitle(
            rf"Log-optimal versus value-preserving: $T={T:g}$, "
            rf"$\pi^\ast={100 * pi_star:.2f}\%$",
            fontsize=15,
        )
        plt.tight_layout()
        plt.show()

    controls = widgets.VBox([widgets.HBox([r_slider, b_slider, sigma_slider]), widgets.HBox([maturity_slider, path_slider])])
    interactive_plot = widgets.interactive_output(
        update_plot,
        {
            "r": r_slider,
            "b": b_slider,
            "sigma": sigma_slider,
            "maturity": maturity_slider,
            "path_index": path_slider,
        },
    )

    display(controls, interactive_plot)

###
###
###
### DAY 4
###
###
###

def explore_payout_plans(simulate_payout_plans_handle):
    """Interactively compare four fund payout plans and an annuity reference."""

    lifetime_slider = widgets.IntSlider(
        value=21,
        min=10,
        max=30,
        step=1,
        description=r"$1/\lambda$:",
        continuous_update=True,
    )

    discount_slider = widgets.FloatSlider(
        value=0.03,
        min=0.005,
        max=0.08,
        step=0.005,
        description=r"$\delta$:",
        continuous_update=True,
    )

    tail_slider = widgets.SelectionSlider(
        options=[("10%", 0.10), ("5%", 0.05), ("1%", 0.01)],
        value=0.05,
        description=r"$z$:",
        continuous_update=True,
    )

    path_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=49,
        step=1,
        description="Path:",
        continuous_update=True,
    )

    def update_plot(expected_lifetime, discount_rate, tail_probability, path_index):
        # Use M = 2500, X0 = 100,000
        M  = 2_500
        X0 = 100_000.0

        mortality_rate = 1 / expected_lifetime
        quantile_horizon = -np.log(tail_probability) / mortality_rate
        simulation_horizon = quantile_horizon + 4 * expected_lifetime

        market = FinancialMarket(risk_free_rate=0.025, risk_premium=0.05, volatility=0.25)
        params = SDESimulationParameters(time_horizon=simulation_horizon, time_steps=int(12 * simulation_horizon), num_paths=M)

        annuity_rate = X0 * (market.risk_free_rate + mortality_rate)

        out = simulate_payout_plans_handle(
            initial_wealth=X0,
            expected_lifetime=expected_lifetime,
            discount_rate=discount_rate,
            tail_probability=tail_probability,
            market_params=market,
            sde_params=params,
        )

        t = out.time_grid
        lifetimes = out.paths["lifetime"]

        plans = [
            ("standard", "Standard infinity", "b"),
            ("mortality", "Exponential lifetime", "orange"),
            ("mean_horizon", r"Fixed $T=\mathbb{E}[\tau]$", "g"),
            ("quantile", r"Quantile horizon $T_z$", "mediumpurple"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

        plot_until = min(simulation_horizon, max(50, quantile_horizon + 5))
        plot_mask = t <= plot_until

        for key, label, color in plans:
            axes[0].plot(t[plot_mask], out.paths[f"{key}_wealth"][plot_mask, path_index], color=color, lw=2.2, label=label)
            axes[1].plot(t[plot_mask], out.paths[f"{key}_consumption"][plot_mask, path_index], color=color, lw=2.2, label=label)

        axes[0].axvline(expected_lifetime, color="grey", linestyle="--", lw=1.2, label=r"$\mathbb{E}[\tau]$")
        axes[0].axvline(quantile_horizon, color="black", linestyle=":", lw=1.5, label=r"$T_z$")
        axes[0].axvline(lifetimes[path_index], color="orange", linestyle="-.", lw=1.5, label="Actual lifetime")
        axes[0].set_title("Fund wealth on the same market path")
        axes[0].set_xlabel("Years")
        axes[0].set_ylabel("Wealth")
        axes[0].grid(alpha=0.25)
        # axes[0].legend(fontsize=8)

        axes[1].axvline(expected_lifetime, color="grey", linestyle="--", lw=1.2, label=r"$\mathbb{E}[\tau]$")
        axes[1].axvline(quantile_horizon, color="black", linestyle=":", lw=1.5, label=r"$T_z$")
        axes[1].axvline(lifetimes[path_index], color="orange", linestyle="-.", lw=1.5, label="Actual lifetime")
        axes[1].axhline(annuity_rate, color="crimson", linestyle="--", lw=1.5, label="Fair annuity")
        axes[1].set_title("Annual payout on the same market path")
        axes[1].set_xlabel("Years")
        axes[1].set_ylabel("Annual payout")
        axes[1].grid(alpha=0.25)
        axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)

        death_index = np.searchsorted(t, np.minimum(lifetimes, t[-1]), side="right") - 1
        path_index_array = np.arange(M)

        mean_lifetime_consumption = []
        depletion_probability = []
        metric_labels = []

        for key, label, color in plans:
            cumulative = out.paths[f"{key}_cumulative_consumption"]
            mean_lifetime_consumption.append(np.mean(cumulative[death_index, path_index_array]))

            if key == "mean_horizon":
                depletion_probability.append(np.mean(lifetimes > expected_lifetime))
            elif key == "quantile":
                depletion_probability.append(np.mean(lifetimes > quantile_horizon))
            else:
                depletion_probability.append(0.0)
            metric_labels.append(label)

        mean_lifetime_consumption.append(np.mean(annuity_rate * lifetimes))
        depletion_probability.append(0.0)
        metric_labels.append("Fair annuity")

        locations = np.arange(len(metric_labels))
        bars = axes[2].bar(locations, mean_lifetime_consumption, color=[plan[2] for plan in plans] + ["crimson"], alpha=0.70)
        axes[2].set_xticks(locations)
        axes[2].set_xticklabels(metric_labels, rotation=25, ha="right")
        axes[2].set_title("Lifetime payout and longevity risk")
        axes[2].set_ylabel("Mean payout received before death")
        axes[2].grid(axis="y", alpha=0.25)

        risk_axis = axes[2].twinx()
        risk_axis.plot(locations, depletion_probability, color="black", marker="o", linestyle="--", lw=1.5)
        risk_axis.set_ylabel("Probability plan ends before death")
        risk_axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, position: f"{value:.0%}"))
        risk_axis.set_ylim(0, max(0.5, 1.15 * max(depletion_probability)))

        fig.suptitle(
            rf"Payout plans: $\mathbb{{E}}[\tau]={expected_lifetime}$, "
            rf"$T_z={quantile_horizon:.1f}$, $\delta={discount_rate:.3f}$",
            fontsize=15,
        )
        plt.tight_layout()
        plt.show()

    controls = widgets.VBox([
        widgets.HBox([lifetime_slider, discount_slider, tail_slider]),
        path_slider,
    ])
    interactive_plot = widgets.interactive_output(
        update_plot,
        {
            "expected_lifetime": lifetime_slider,
            "discount_rate": discount_slider,
            "tail_probability": tail_slider,
            "path_index": path_slider,
        },
    )

    display(controls, interactive_plot)


def explore_climate_scenarios(history_years, history_anomaly, annual_volatility, simulate_climate_scenario_handle):
    """Interactively simulate arithmetic-Brownian temperature paths around IPCC scenario means."""

    scenario_dropdown = widgets.Dropdown(
        options=list(IPCC_AR6_SCENARIO_CENTRAL.keys()),
        value="SSP2-4.5",
        description="Scenario:",
    )

    threshold_slider = widgets.FloatSlider(
        value=1.5,
        min=1.0,
        max=4.0,
        step=0.1,
        description="Threshold:",
        continuous_update=True,
    )

    volatility_slider = widgets.FloatSlider(
        value=1.0,
        min=0,
        max=2,
        step=0.05,
        description=r"$\sigma$ mult.:",
        continuous_update=True,
    )

    path_slider = widgets.IntSlider(
        value=25,
        min=5,
        max=250,
        step=5,
        description="#Paths:",
        continuous_update=False,
    )

    def update_plot(scenario, threshold, volatility_multiplier, num_plot_paths):
        forecast_years = np.arange(2020, 2101)
        initial_anomaly = history_anomaly[-1]
        num_paths = 5_000

        scenario_means = {
            name: build_climate_scenario_mean(forecast_years, initial_anomaly, values)
            for name, values in IPCC_AR6_SCENARIO_CENTRAL.items()
        }

        out = simulate_climate_scenario_handle(
            initial_anomaly=initial_anomaly,
            forecast_years=forecast_years,
            scenario_mean=scenario_means[scenario],
            annual_volatility=volatility_multiplier * annual_volatility,
            num_paths=num_paths
        )

        K = out.paths["temperature_anomaly"]
        selected_mean = out.paths["scenario_mean"]

        colors = {
            "SSP1-1.9": "green",
            "SSP1-2.6": "teal",
            "SSP2-4.5": "orange",
            "SSP5-8.5": "crimson",
        }

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

        # Compute moving average for the history anomaly to smooth out short-term fluctuations
        window_size = 10
        history_anomaly_smoothed = np.convolve(history_anomaly, np.ones(window_size)/window_size, mode='valid')
        history_anomaly_smoothed = np.append(history_anomaly_smoothed, scenario_means[scenario][0])
        smoothed_years = np.arange(history_years[window_size - 2], history_years[-1] + 1)

        axes[0].plot(history_years, history_anomaly, color="black", lw=1.8, label="HadCRUT5 history")
        axes[0].plot(smoothed_years, history_anomaly_smoothed, color="blue", lw=2.0, linestyle="-", label="Smoothed history")
        for name, mean_path in scenario_means.items():
            width = 3.0 if name == scenario else 1.5
            alpha = 1.0 if name == scenario else 0.55
            axes[0].plot(forecast_years, mean_path, color=colors[name], lw=width, alpha=alpha, label=name)
        axes[0].axvline(2020, color="grey", linestyle="--", lw=1.3)
        axes[0].set_title("History and conditional scenario means")
        axes[0].set_xlabel("Year")
        axes[0].set_ylabel(r"Warming relative to 1850--1900 ($^\circ$C)")
        axes[0].grid(alpha=0.25)
        axes[0].legend(fontsize=8)

        lower_05, lower_25, upper_75, upper_95 = np.percentile(K, [5, 25, 75, 95], axis=1)
        axes[1].fill_between(forecast_years, lower_05, upper_95, color=colors[scenario], alpha=0.15, label="5%--95% simulation band")
        axes[1].fill_between(forecast_years, lower_25, upper_75, color=colors[scenario], alpha=0.28, label="25%--75% simulation band")
        axes[1].plot(forecast_years, K[:, :num_plot_paths], color=colors[scenario], alpha=0.08, lw=0.7)
        axes[1].plot(forecast_years, selected_mean, color="black", lw=2.4, label="Scenario mean")
        axes[1].axhline(threshold, color="crimson", linestyle="--", lw=1.8, label=rf"${threshold:.1f}^\circ$C threshold")
        axes[1].set_title(f"Arithmetic-Brownian paths: {scenario}")
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel(r"Warming relative to 1850--1900 ($^\circ$C)")
        axes[1].grid(alpha=0.25)
        axes[1].legend(fontsize=8)

        exceedance_probability = np.mean(K >= threshold, axis=1)
        axes[2].plot(forecast_years, exceedance_probability, color=colors[scenario], lw=2.5)
        axes[2].fill_between(forecast_years, 0, exceedance_probability, color=colors[scenario], alpha=0.15)
        axes[2].set_title("Probability an individual year exceeds the threshold")
        axes[2].set_xlabel("Year")
        axes[2].set_ylabel("Estimated probability")
        axes[2].set_ylim(0, 1)
        axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, position: f"{value:.0%}"))
        axes[2].grid(alpha=0.25)

        fig.suptitle(
            rf"Climate scenario experiment: historical $\hat\sigma={annual_volatility:.3f}$, "
            rf"simulation $\sigma={volatility_multiplier * annual_volatility:.3f}$",
            fontsize=15,
        )
        plt.tight_layout()
        plt.show()

    controls = widgets.VBox([
        widgets.HBox([scenario_dropdown, threshold_slider]),
        widgets.HBox([volatility_slider, path_slider]),
    ])
    interactive_plot = widgets.interactive_output(
        update_plot,
        {
            "scenario": scenario_dropdown,
            "threshold": threshold_slider,
            "volatility_multiplier": volatility_slider,
            "num_plot_paths": path_slider
        },
    )

    display(controls, interactive_plot)