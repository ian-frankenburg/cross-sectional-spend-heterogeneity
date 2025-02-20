import numpy as np
from scipy import stats
from dataclasses import dataclass
import polars as pl

@dataclass
class SimConfig:
    """Configuration parameters for the customer lifecycle simulation"""
    r: float = 1.0          # Shape parameter for spend distribution
    alpha: float = 50.0     # Rate parameter for transaction intensity
    a: float = 1.0          # Beta distribution parameter for churn
    b: float = 50.0         # Beta distribution parameter for churn
    tau: float = 13.0       # Time scaling factor
    n_customers: int = 10000
    max_time: float = 52 * 5  # 5 years simulation period
    max_transactions: int = 1000  # Maximum transactions per customer

def generate_customer_base(config, t0=None):
    """Generate initial customer base with their characteristics"""
    if t0 is None:
        t0 = [0] * config.n_customers
        
    base_df = (pl.DataFrame()
        .with_columns(t0=pl.Series(t0))
        .sort('t0')
        .with_row_index('id'))
    
    return add_customer_parameters(base_df, config)

def add_customer_parameters(df, config):
    """Add individual-level parameters for each customer"""
    theta = np.random.beta(config.a, config.b, len(df))
    churn_times = (1 + np.random.geometric(theta)) * config.tau
    
    return df.with_columns(
        (pl.col('t0') + pl.Series(churn_times)).alias('churn_time'),
        pl.Series(np.random.gamma(
            config.r, 
            1/config.alpha, 
            df.shape[0]
        )).alias('lambda')
    )

def generate_arrival_times(customer_base, config):
    """Generate transaction arrival times using exponential distribution"""
    exp_times = np.random.exponential(
        scale=1/customer_base['lambda'].to_numpy()[:, np.newaxis],
        size=(customer_base.shape[0], config.max_transactions)
    )
    exp_times = np.column_stack([customer_base['t0'], exp_times])
    cumsum_times = np.cumsum(exp_times, axis=1)
    
    return pl.DataFrame({
        'id': pl.Series(np.repeat(
            range(customer_base.shape[0]), 
            cumsum_times.shape[1]
        )).cast(pl.UInt32),
        'tx_time': cumsum_times.flatten()
    })

def generate_transactions(customer_base, config):
    """Generate transaction history for the customer base"""
    transaction_times = generate_arrival_times(customer_base, config)
    
    return (transaction_times
        .join(customer_base, on='id', how='left')
        .filter(
            pl.col('tx_time') <= pl.min_horizontal(
                pl.col('churn_time'), 
                config.max_time
            )
        )
        .select('id', 'tx_time')
        .with_columns(
            (pl.col('id').rank(method='dense') - 1).alias('id')
        )
        .sort('id'))

def run_simulation(config=None):
    """Run the complete customer lifecycle simulation"""
    if config is None:
        config = SimConfig()
    
    customer_base = generate_customer_base(config)
    transactions = generate_transactions(customer_base, config)
    
    return transactions

# Example usage:
transactions_df = run_simulation()

# Or with custom configuration
custom_config = SimConfig(
    r=1.5,
    alpha=45,
    n_customers=2000,
    max_time=52 * 3  # 3 years
)

txlog = run_simulation(custom_config)
