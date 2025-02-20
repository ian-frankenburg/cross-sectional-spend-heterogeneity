import numpy as np
from scipy.special import gammaln as lgamma
from scipy import integrate
from plotnine import *
import pandas as pd

# Configuration
CONFIG = {
    'rho_min': 1e-5,
    'rho_max': 50,
    'rho_points': 500,
    'nu_fixed': .25,  # Fixed value for nu
}

# Define hyperparameter settings to try
HYPER_SETTINGS = [
    (1, .01)
]

def log_kernel(rho, a, c):
    return (rho - 1.0) * np.log(a) - c * lgamma(rho)

def kernel_int(a, c):
    # Define the grid for RHO inside the function
    RHO = np.linspace(1e-3, 100, 1000)
    log_fn = log_kernel(RHO, a, c)
    integrand = np.exp(log_fn)
    return integrate.trapezoid(integrand, x=RHO)

def compute_density(rho, a, c):
    """Compute density for given rho and hyperparameters a, c"""
    lnumerator = (rho - 1) * np.log(a) - c * lgamma(rho)
    ldenominator = np.log(kernel_int(a, c))
    return np.exp(lnumerator - ldenominator)

def create_plot_data(hyper_settings, config):
    """Create dataframe for plotting from hyperparameter settings"""
    rho_range = np.linspace(
        config['rho_min'], 
        config['rho_max'], 
        config['rho_points']
    )
    
    data = []
    for a, c in hyper_settings:
        label = f'a={a:.2f}, c={c:.2f}'
        for rho in rho_range:
            density = compute_density(rho, a, c)
            data.append({
                'rho': rho,
                'density': density,
                'setting': label
            })
    
    return pd.DataFrame(data)

def plot_densities(df):
    """Create the density plot"""
    plot = (ggplot(df, aes('rho', 'density', color='setting', group='setting'))
        + geom_line(size=1)
        + scale_color_brewer(type='qual', palette='Set2')
        + theme_dark()
        + labs(
            x='ρ (rho)',
            y='Density',
            title='1-D Density for Different Hyperparameter Settings',
            color='Hyperparameters'
        )
        + theme(
            plot_background=element_rect(fill='black'),
            panel_background=element_rect(fill='black'),
            text=element_text(color='white', size=12),
            axis_text=element_text(color='white'),
            axis_title=element_text(color='white', size=12),
            title=element_text(color='white', size=14),
            legend_background=element_rect(fill='black'),
            legend_text=element_text(color='white'),
            legend_title=element_text(color='white'),
            panel_grid_major=element_line(color='white', alpha=0.2),
            panel_grid_minor=element_line(color='white', alpha=0.1),
        )
        + scale_x_continuous(breaks=np.arange(0, CONFIG['rho_max'] + 0.5, 1))
        + scale_y_continuous(expand=(0, 0))
    )
    return plot

# Generate and display the plot
df = create_plot_data(HYPER_SETTINGS, CONFIG)
plot = plot_densities(df)
print(plot)
plot.draw(show=True)
