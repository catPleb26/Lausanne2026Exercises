import numpy as np
from utils.common import InvestmentType, SDESimulationParameters, SDEOutput, FinancialMarket


def simulate_brownian_path(mu: float = 0, sigma: float = 1, params: SDESimulationParameters = SDESimulationParameters()) -> SDEOutput:
    """
    Generate a Brownian motion path using the Euler-Maruyama scheme.
    """
    # Set seed for reproducibility
    np.random.seed(params.seed)

    # Initialize the Brownian path array
    brownian_path = np.zeros((params.time_steps + 1, params.num_paths))

    # Generate the Brownian motion increments
    dt = params.dt
    dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(params.time_steps, params.num_paths))

    t  = np.linspace(0, params.time_horizon, params.time_steps + 1)
    W  = np.cumsum(dW, axis=0)

    # Apply the Euler-Maruyama method to generate the Brownian path
    brownian_path[1:,:] = mu * t[1:, np.newaxis] + sigma * W

    return SDEOutput(time_grid=t, paths={'W': brownian_path, 'dW': dW})


def simulate_market(initial_stock_price: float, market_params: FinancialMarket, sde_params: SDESimulationParameters) -> SDEOutput:
    """
    Simulate market dynamics using the Euler-Maruyama scheme.
    """
    
    # Initialize the bond and stock price arrays
    bond_prices = np.zeros((sde_params.time_steps + 1, sde_params.num_paths))
    stock_prices = np.zeros((sde_params.time_steps + 1, sde_params.num_paths))

    # Set initial prices
    bond_prices[0, :] = 1.0  # Assuming the bond starts at 1.0
    stock_prices[0, :] = initial_stock_price

    # Generate the Brownian motion increments
    out = simulate_brownian_path(mu=0, sigma=1, params=sde_params)
    dt  = sde_params.dt
    dW  = out.paths['dW']

    # Simulate the market dynamics
    # dB_t = r * B_t dt                     -> B_n = B_{n-1} * (1 + r * dt)
    # dS_t = mu * S_t dt + sigma * S_t dW_t -> S_n = S_{n-1} * (1 + mu * dt + sigma * dW_n)
    t   = np.linspace(0, sde_params.time_horizon, sde_params.time_steps + 1)
    B   = np.cumprod(np.repeat(1 + market_params.risk_free_rate * dt, sde_params.time_steps).reshape(-1, 1))
    S   = stock_prices[0, :] * np.cumprod(1 + market_params.drift * dt + market_params.volatility * dW, axis=0)

    # Store the bond and stock prices
    bond_prices[1:, :] = np.repeat(B[:, np.newaxis], sde_params.num_paths, axis=1)
    stock_prices[1:, :] = S

    return SDEOutput(time_grid=t, paths={'bond': bond_prices, 'stock': stock_prices, 'driver': out.paths['W']})


def simulate_market_exact(initial_stock_price: float, market_params: FinancialMarket, sde_params: SDESimulationParameters) -> SDEOutput:
    """
    Simulate market dynamics using the closed-form solution of the SDEs for bond and stock prices.
    """

    # Generate the driving Brownian motion path
    out = simulate_brownian_path(0, 1, sde_params)

    t = out.time_grid
    W = out.paths['W']

    # Compute the bond and stock prices using the exact solution
    bond_price = np.exp(market_params.risk_free_rate * t[:, np.newaxis])
    stock_price = initial_stock_price * np.exp((market_params.drift - 0.5 * market_params.volatility**2) * t[:, np.newaxis] + market_params.volatility * W)

    return SDEOutput(time_grid=t, paths={'bond': bond_price, 'stock': stock_price, 'driver': W})


def simulate_wealth(initial_wealth: float, constant_investment: float, investment_strategy: InvestmentType, market_params: FinancialMarket, sde_params: SDESimulationParameters) -> SDEOutput:
    """
    Simulate wealth dynamics based on the chosen investment strategy.
    """

    # Initialize the wealth array
    wealth = np.zeros((sde_params.time_steps + 1, sde_params.num_paths))
    
    # Set initial prices
    wealth[0, :] = initial_wealth
    
    # Generate the Brownian motion increments
    out = simulate_brownian_path(mu=0, sigma=1, params=sde_params)
    dt  = sde_params.dt
    dW  = out.paths['dW']

    # Simulate the market dynamics
    match investment_strategy:
        case InvestmentType.ConstantFraction:
            u = lambda x: constant_investment * x  # Investment is a fraction of current wealth
        case InvestmentType.ConstantNominal:
            u = lambda x: constant_investment      # Investment is a fixed nominal amount
        case _:
            raise ValueError("Invalid investment strategy. Choose either 'ConstantFraction' or 'ConstantNominal'.")

    # Simulate wealth dynamics based on the chosen investment strategy
    t = np.linspace(0, sde_params.time_horizon, sde_params.time_steps + 1)
    for i in range(1, sde_params.time_steps + 1):
        current_wealth = wealth[i - 1, :]
        investment_amount = u(current_wealth)
                
        # Update wealth based on the market dynamics
        # dX_t = (rX_t + u_t * (mu - r)) dt + u_t * sigma dW_t
        wealth[i, :] = current_wealth + (market_params.risk_free_rate * current_wealth + investment_amount * (market_params.drift - market_params.risk_free_rate)) * dt + investment_amount * market_params.volatility * dW[i - 1, :]

    return SDEOutput(time_grid=t, paths={'X': wealth, 'driver': out.paths['W']})