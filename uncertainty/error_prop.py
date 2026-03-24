import numpy as np
from utils.constants import get_covariance_matrix, NUFIT_PARAMS

def monte_carlo_propagation(solver_func, E_grid, N_samples, base_params, **solver_kwargs):
    """
    Algorithm 3: Monte Carlo Parameter Sampling.
    """
    keys = list(base_params.keys())
    means = [base_params[k] for k in keys]
    cov_matrix = get_covariance_matrix()
    
    # Sample from multivariate Gaussian
    samples = np.random.multivariate_normal(means, cov_matrix, size=N_samples)
    
    P_mue_samples = np.zeros((N_samples, len(E_grid)))
    
    for i in range(N_samples):
        # Package sample into dictionary
        sampled_params = {k: v for k, v in zip(keys, samples[i])}
        
        # Physical boundary constraints (wrap phase, clip angles)
        sampled_params['deltaCP'] = sampled_params['deltaCP'] % (2 * np.pi)
        for ang in ['theta12', 'theta13', 'theta23']:
            sampled_params[ang] = np.clip(sampled_params[ang], 0, np.pi/2)
            
        P_mue, _ = solver_func(E_grid, params=sampled_params, **solver_kwargs)
        P_mue_samples[i, :] = P_mue
        
    mean_P = np.mean(P_mue_samples, axis=0)
    std_P = np.std(P_mue_samples, axis=0)
    
    return mean_P, std_P

def jacobian_propagation(solver_func, E_grid, base_params, epsilon_frac=0.01, **solver_kwargs):
    """
    Algorithm 4: Linearized Error Propagation.
    Extremely fast derivative-based uncertainty estimation.
    """
    P_0, _ = solver_func(E_grid, params=base_params, **solver_kwargs)
    cov_matrix = get_covariance_matrix()
    keys = list(base_params.keys())
    
    J = np.zeros((len(E_grid), len(keys)))
    
    for i, key in enumerate(keys):
        val = base_params[key]
        eps = val * epsilon_frac if val != 0 else 1e-4
        
        params_up = base_params.copy()
        params_down = base_params.copy()
        
        params_up[key] += eps
        params_down[key] -= eps
        
        P_up, _ = solver_func(E_grid, params=params_up, **solver_kwargs)
        P_down, _ = solver_func(E_grid, params=params_down, **solver_kwargs)
        
        J[:, i] = (P_up - P_down) / (2 * eps)
        
    # Variance = diag(J * Cov * J^T)
    var_P = np.sum((J @ cov_matrix) * J, axis=1)
    return P_0, np.sqrt(var_P)