import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, grad, vmap
from jax.scipy import special
from scipy.optimize import minimize
import numpy as np

jax.config.update("jax_enable_x64", True)

def ldnorm(x, mu=0.0, sigma=1.0):
    return -0.5 * jnp.log(2 * jnp.pi) - jnp.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2

# --------------------------------
# Model 1: Gamma-Gamma functions
# --------------------------------
def gg_marginal_likelihood(loghyps, zsum, zprod, x):
    rho, q, gamma = jnp.exp(loghyps)
    def gg_term(zsum_i, zprod_i, x_i):
        term1 = (rho - 1) * jnp.log(zprod_i)
        term2 = q * jnp.log(gamma) + special.gammaln(x_i * rho + q)
        term3 = (x * special.gammaln(rho) + special.gammaln(q) + 
                 (x_i * rho + q) * jnp.log(zsum_i + gamma))
        return term1 + term2 - term3
    mll = jnp.sum(vmap(gg_term)(zsum, zprod, x))
    prior = jnp.sum(ldnorm(loghyps))
    return -(mll + prior)

# Global used for trapezoidal numerical integration
RHO = jnp.linspace(1e-3, 100, 1000)
# --------------------------------
# Model 2: latent heterogeneity modeled through shape parameter
# --------------------------------
def ghr_kernel(rho, a, c):
    return (rho - 1.0) * jnp.log(a) - c * special.gammaln(rho)

def ghr_kernel_integral(a, c, M=0):
    log_fn = ghr_kernel(RHO, a, c)
    integrand = jnp.exp(log_fn - M)
    return jnp.trapezoid(integrand, RHO)

def ghr_marginal_likelihood(loghyps, zsum, zprod, x):
    hyps = jnp.exp(loghyps)
    nu, a, c = hyps
    log_fn = ghr_kernel(RHO, a, c)
    M = jnp.max(log_fn)
    log_denom = jnp.log(ghr_kernel_integral(a, c, M))
    def term(zsum_i, zprod_i, x_i):
        a_tilde = a * zprod_i * nu ** x_i
        c_tilde = x_i + c
        return (x_i * jnp.log(nu) - nu * zsum_i + 
                jnp.log(ghr_kernel_integral(a_tilde, c_tilde, M)) - log_denom)
    
    terms = vmap(term)(zsum, zprod, x)
    mll = jnp.sum(terms) + jnp.sum(ldnorm(loghyps))
    return -mll

# --------------------------------------------
# Model 3: Bivariate latent heterogeneity
# --------------------------------------------
def ghsr_kernel(rho, a, b, c, d):
    return ((rho - 1.0) * jnp.log(a) +
            (-rho * b - 1.0) * jnp.log(c) +
            special.gammaln(b * rho + 1.0) -
            d * special.gammaln(rho))

def ghsr_kernel_integral(a, b, c, d, M=0):
    log_fn = ghsr_kernel(RHO, a, b, c, d)
    integrand = jnp.exp(log_fn - M)
    return jnp.trapezoid(integrand, RHO)

def ghsr_marginal_likelihood(loghyps, zsum, zprod, x):
    hyps = jnp.exp(loghyps)
    a, b, c, d = hyps
    log_fn = ghsr_kernel(RHO, a, b, c, d)
    M = jnp.max(log_fn)
    log_denom = jnp.log(ghsr_kernel_integral(a, b, c, d, M))
    def term(zsum_i, zprod_i, x_i):
        a2 = a * zprod_i
        b2 = b + x_i
        c2 = c + zsum_i
        d2 = d + x_i
        return jnp.log(ghsr_kernel_integral(a2, b2, c2, d2, M)) - log_denom
    
    terms = vmap(term)(zsum, zprod, x)
    mll = jnp.sum(terms) + jnp.sum(ldnorm(loghyps))
    return -mll

def convert_to_jax_array(df, columns):
    return {
        col: jnp.array([np.array(sublist) for sublist in df.select(col).to_series().to_list()])
        for col in columns
    }

def prepare_optimization_data(df, feature_columns=['zsum', 'zprod', 'x']):
    features = convert_to_jax_array(df, feature_columns)
    return tuple(features[col] for col in feature_columns)

def create_optimizer(likelihood_fn, n_params, bounds):
    """Create an optimizer for a given likelihood function"""
    jitted_marginal_likelihood = jit(likelihood_fn)
    jitted_gradient = jit(grad(likelihood_fn))
    
    def generate_valid_start():
        """Generate valid starting parameters"""
        logstart = np.random.uniform(-1, 1, n_params)
        if n_params == 4: 
            while logstart[1] <= logstart[3]:
                logstart[[1,3]] = np.random.uniform(-1, 1, 2)
                if logstart[1] <= logstart[3]:  # b <= d constraint
                    return logstart
        return logstart
    
    def optimize(rfm, n_tries=3, verbose=True):
        best_result = None
        best_value = float('inf')
        
        zsum, zprod, x = prepare_optimization_data(rfm)
        
        for i in range(n_tries):
            start = generate_valid_start()
            
            result = minimize(
                jitted_marginal_likelihood,
                args=(zsum, zprod, x),
                x0=start,
                method='L-BFGS-B',
                jac=jitted_gradient,
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if verbose:
                print_status(result, i)
                
            if result.fun < best_value:
                best_value = result.fun
                best_result = result
                
        if best_result is None:
            raise Exception(f"Failed to find valid solution after {n_tries} attempts")
            
        return format_result(best_result)
    
    return optimize

def print_status(result, iteration):
    """Print optimization status information"""
    if result.status == 2:
        print('\033[1;31m⚠ Optimization Terminated!\033[0m')
    else:
        print('\033[1;32m✓ Optimization Success!\033[0m')
        print(f"hyps: {np.round(np.exp(result.x), 2)}")
        print(f"fn: {np.round(result.fun, 2)}\n")
    print(f"Iteration {iteration}")

def format_result(result):
    """Format optimization result"""
    return {
        'logparameters': result.x,
        'log_marginal_likelihood': -result.fun,
        'success': result.success,
        'message': result.message
    }

# Model configurations
MODEL_CONFIGS = {
    'gg': {
        'likelihood': gg_marginal_likelihood,
        'n_params': 3,
        'bounds': [(-8, 8)] * 3
    },
    'ghr': {
        'likelihood': ghr_marginal_likelihood,
        'n_params': 3,
        'bounds': [(-5, 5)] * 3
    },
    'ghsr': {
        'likelihood': ghsr_marginal_likelihood,
        'n_params': 4,
        'bounds': [(-13, 13)] * 4
    }
}

# Main estimation function
def estimate_model(model_type, rfm, n_tries=3):
    """Estimate parameters for specified model type"""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}")
        
    config = MODEL_CONFIGS[model_type]
    optimizer = create_optimizer(
        config['likelihood'],
        config['n_params'],
        config['bounds']
    )
    
    return optimizer(rfm, n_tries)
