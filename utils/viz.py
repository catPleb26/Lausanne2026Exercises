from .common import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import ipywidgets as widgets
from IPython.display import display

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

##########################
##########################
##########################

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

def optimal_vs_suboptimal_plot():
    xi_sub_slider = widgets.FloatSlider(
        value=3.2,
        min=-4,
        max=12,
        step=0.1,
        description=r"$\xi_{\mathrm{sub}}$:",
        continuous_update=False
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

        strategies = generate_strategies(simulate_wealth, xi_sub, abs_risk_aversion, market_params=market_params, sde_params=sde_params)

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