import numpy as np

# --- FUNDAMENTAL CONSTANTS ---
G_F = 1.1663787e-5  # Fermi coupling constant in GeV^-2
m_N = 0.939         # Nucleon mass in GeV
KM_TO_EV = 5.06773e12 # Conversion factor: 1 km to eV^-1
GEV_TO_EV = 1e9     # Conversion factor: 1 GeV to eV

# --- DEFAULT EXPERIMENT SETTINGS ---
DEFAULT_BASELINE = 295.0  # L in km (T2K/Hyper-K)
DEFAULT_DENSITY = 2.6     # rho in g/cm^3 (crust average)
DEFAULT_YE = 0.5          # Electron fraction

# --- NuFIT 6.0 BEST FIT PARAMETERS (Normal Ordering) ---
# Format: 'parameter': [best_fit, 1_sigma_error]
NUFIT_PARAMS = {
    'theta12': [np.radians(33.68), np.radians(0.70)],
    'theta13': [np.radians(8.52), np.radians(0.11)],
    'theta23': [np.radians(48.5), np.radians(0.9)],
    'deltaCP': [np.radians(177.0), np.radians(19.5)], # Averaged +19/-20
    'dm21': [7.49e-5, 0.19e-5],      # eV^2
    'dm31': [2.534e-3, 0.025e-3]     # eV^2
}

def get_best_fit():
    """Returns a dictionary of just the best-fit values."""
    return {k: v[0] for k, v in NUFIT_PARAMS.items()}

def get_covariance_matrix():
    """Constructs a diagonal covariance matrix from 1-sigma errors."""
    errors = np.array([v[1] for v in NUFIT_PARAMS.values()])
    return np.diag(errors**2)