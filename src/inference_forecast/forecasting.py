import jax.numpy as jnp
from jax import vmap
import numpy as np
from scipy.optimize import minimize, root_scalar
from scipy.special import digamma, gammaln
from src.inference_forecast import inference

def forecast_gg(params, rfm):
    zsum, zprod, x = inference.prepare_optimization_data(rfm)
    rho, q, gamma = np.exp(params)
    return (rho * zsum + rho * gamma) / (x * rho + q - 1)

def forecast_ghr(params, rmf):
    zsum, zprod, x = inference.prepare_optimization_data(rfm)
    nu, a, c = np.exp(params)
    def maximize_posterior(zsum_i, zprod_i, x_i):
        a_tilde = a * zprod_i * nu ** x_i
        c_tilde = x_i + c
        
        def objective(rho):
            return -((rho - 1) * np.log(a_tilde) - c_tilde * gammaln(rho))
            
        result = minimize(
            objective, x0=1.0, 
            bounds=[(1e-10, 100)], 
            method='L-BFGS-B'
        )
        return result.x[0] / nu
        
    return np.array([maximize_posterior(z, p, xi) 
                    for z, p, xi in zip(zsum, zprod, x)])

def forecast_ghsr(params, rmf):
    """Complex kernel model forecasting"""
    a, b, c, d = np.exp(params)
    
    def maximize_posterior(zsum_i, zprod_i, x_i):
        def rho_equation(rho, a_i, b_i, c_i, d_i):
            return (np.log(a_i) + b_i * np.log(rho) - 
                   b_i * np.log(c_i) - d_i * digamma(rho))
        
        a_i = a * zprod_i
        b_i = b + x_i
        c_i = c + zsum_i
        d_i = d + x_i
        
        sol = root_scalar(
            rho_equation,
            args=(a_i, b_i, c_i, d_i),
            bracket=[1e-10, 100],
            method='brentq'
        )
        
        rho_opt = sol.root
        nu_opt = (rho_opt * b_i) / c_i
        return rho_opt / nu_opt
    
    return np.array([maximize_posterior(z, p, xi) 
                    for z, p, xi in zip(zsum, zprod, x)])

def forecast(model_type, params, rfm):
    zsum, zprod, x = inference.prepare_optimization_data(rfm)
    forecasting_fns = {
        'gg': forecast_gg,
        'ghr': forecast_ghr,
        'ghsr': forecast_ghsr
    }
    
    if model_type not in forecasting_fns:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return forecasting_fns[model_type](params, rfm)
