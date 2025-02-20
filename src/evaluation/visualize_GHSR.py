import numpy as np
from scipy.special import gammaln as lgamma
from scipy import integrate
import matplotlib.pyplot as plt

# Set dark theme
plt.style.use('dark_background')

# Define the grid for RHO
RHO = np.linspace(1e-3, 100, 1000)

def log_kernel(rho, a, c):
    return (rho - 1.0) * np.log(a) - c * lgamma(rho)

def kernel_int(a, c):
    log_fn = log_kernel(RHO, a, c)
    integrand = np.exp(log_fn)
    return integrate.trapezoid(integrand, x=RHO)

def ldensity(params, loghyps, z):
    rho, nu = params
    a = np.exp(loghyps[0])
    c = np.exp(loghyps[1])
    lnumerator = (rho - 1) * np.log(a) - c * lgamma(rho)
    ldenominator = np.log(kernel_int(a,c))
    return np.exp(lnumerator - ldenominator)

# Create a grid of parameters (natural scale)
rho_range = np.linspace(0.1, 50, 100)    # natural scale for rho
nu_range = np.linspace(0.1, 1, 100)     # natural scale for nu
X, Y = np.meshgrid(rho_range, nu_range)

# Example hyperparameters (log scale)
# # a --> expands \rho
# # b --> variation in \nu
# # c --> stronger decay for large nu
# # d --> controls shape weight
hyps = [3.0, 1e-5, 20, .35]  # log(a), log(b), log(c), log(d)
loghyps = np.log(hyps)
# Calculate density for each point
Z = np.zeros_like(X)
for i in range(len(rho_range)):
    for j in range(len(nu_range)):
        params = [X[i,j], Y[i,j]]
        Z[i,j] = ldensity(params, loghyps, None)

# Create the contour plot
plt.figure(figsize=(12, 10))
plt.set_cmap('plasma')  # Using plasma colormap which works well with dark theme

# Create filled contours for better visibility
contourf = plt.contourf(X, Y, Z, levels=20, alpha=0.8)
contour = plt.contour(X, Y, Z, levels=20, colors='white', alpha=0.3, linewidths=0.5)

# Add colorbar with white label
colorbar = plt.colorbar(contourf)
colorbar.set_label('Log Density', color='white', size=12)
colorbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(colorbar.ax.axes, 'yticklabels'), color='white')

# Add labels and title with enhanced visibility
plt.xlabel('ρ (rho)', color='white', size=12)
plt.ylabel('ν (nu)', color='white', size=12)
plt.title('Log Density Contour Plot', color='white', size=14, pad=20)

# Customize grid
plt.grid(True, linestyle='--', alpha=0.2, color='white')

# Customize ticks
plt.tick_params(colors='white')

# Adjust layout
plt.tight_layout()

plt.show()
