import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln  
import jax
from dataclasses import dataclass
import polars as pl

@dataclass
class GammaGammaConfig:
    rho: float = 6.0    # shape parameter for individual-level gamma distribution
    a: float = 10.0     # shape parameter for gamma distribution of rate parameter
    b: float = 3.0      # scale parameter for gamma distribution of rate parameter

@dataclass
class GHRConfig:
    a: float = 1.0
    c: float = 0.01
    nu: float = 0.25

@dataclass
class GHSRConfig:
    a: float = 1.0
    b: float = 1.0
    c: float = 0.01
    d: float = 1.0

def ghr_model(loghyps):
    a, c = np.exp(loghyps)
    rho = numpyro.sample('rho', dist.HalfNormal(10))
    log_prob = (rho - 1.0) * jnp.log(a) - c * gammaln(rho)
    numpyro.factor('likelihood', log_prob)

def ghsr_model(loghyps):
    a, b, c, d = jnp.exp(loghyps)
    rho = numpyro.sample('rho', dist.HalfNormal(10))
    nu = numpyro.sample('nu', dist.HalfNormal(10))
    log_prob = (rho - 1) * jnp.log(a) + rho * b * jnp.log(nu) - nu * c - d * gammaln(rho)
    numpyro.factor('likelihood', log_prob)
    
def generate_gg_spends(txlog, config=None):
    """Generate spend amounts using a Gamma-Gamma model
    
    The Gamma-Gamma model assumes:
    1. Individual transaction amounts follow a Gamma(rho, nu)
    2. The rate parameter nu follows a Gamma(a, b)
    """
    if config is None:
        config = GammaGammaConfig()
    
    N = txlog['id'].n_unique()
    
    # Generate individual-level rate parameters
    nu_i = np.random.gamma(config.a, config.b, size=N)
    
    # Create DataFrame with customer parameters
    customer_params = (pl.DataFrame({'nu': nu_i})
                      .with_row_index('id'))
    
    # Join with transaction log and generate spends
    txlog_with_params = txlog.join(customer_params, on='id', how='left')
    
    return txlog_with_params.with_columns(
        spend=np.random.gamma(
            config.rho,
            1/np.array(txlog_with_params['nu'])
        )
    )


def generate_ghr_spends(txlog, config=None, num_warmup=5000):
    if config is None:
        config = GHRConfig()
        
    N = txlog['id'].n_unique()
    loghyps = np.log([config.a, config.c])
    
    kernel = NUTS(ghr_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=N)
    mcmc.run(jax.random.PRNGKey(0), loghyps=loghyps)
    
    samples_dict = mcmc.get_samples()
    rho_samples = (pl.DataFrame({'rho': np.array(samples_dict['rho'])})
                  .with_row_index("id"))
    
    txlog_with_params = txlog.join(rho_samples, on='id', how='left')
    
    return txlog_with_params.with_columns(
        spend=np.random.gamma(
            np.array(txlog_with_params['rho']), 
            config.nu
        )
    )

def generate_ghsr_spends(txlog, config=None, num_warmup=5000):
    if config is None:
        config = GHSRConfig()
        
    N = txlog['id'].n_unique()
    loghyps = np.log([config.a, config.b, config.c, config.d])
    
    kernel = NUTS(ghsr_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=N)
    mcmc.run(jax.random.PRNGKey(0), loghyps=loghyps)
    
    samples_dict = mcmc.get_samples()
    rho_nu_samples = (pl.DataFrame({
        'rho': np.array(samples_dict['rho']),
        'nu': np.array(samples_dict['nu'])
    }).with_row_index("id"))
    
    rho_nu_samples = rho_nu_samples[-N:,:]
    rho_nu_samples = rho_nu_samples.drop('id').with_row_index("id")
    
    txlog_with_params = txlog.join(rho_nu_samples, on='id', how='left')
    
    return (txlog_with_params
        .with_columns(
            spend=np.random.gamma(
                np.array(txlog_with_params['rho']), 
                1.0 / np.array(txlog_with_params['nu'])
            )
        ))

# Update the model comparison function to include the new model
def run_model_comparison(txlog, seeds=None):
    """Run and compare different spend models"""
    if seeds is None:
        seeds = [0]
        
    results = {}
    
    models = {
        'GHR': lambda: generate_ghr_spends(txlog),
        'GHSR': lambda: generate_ghsr_spends(txlog),
        'GammaGamma': lambda: generate_gg_spends(txlog)
    }
    
    for model_name, model_fn in models.items():
        for seed in seeds:
            key = f"{model_name}_seed_{seed}"
            jax.random.PRNGKey(seed)
            results[key] = model_fn()
            
    return results

# Use with default parameters
# txlog_enriched = generate_gg_spends(txlog)
# 
# # Or with custom configuration
# custom_config = GammaGammaConfig(rho=5.0, a=8.0, b=2.0)
# spends_df = generate_gg_spends(txlog, custom_config)
# 
# # Or as part of model comparison
# model_results = run_model_comparison(txlog)
