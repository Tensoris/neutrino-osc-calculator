import numpy as np
from core.physics import construct_pmns, calculate_matter_potential
from utils.constants import KM_TO_EV, GEV_TO_EV

def exact_solver(E_GeV, L_km, rho, Ye, params):
    """
    Algorithm 1: Exact Hamiltonian Diagonalization
    Fully vectorized for ultra-fast energy spectrum computation.
    """
    E_eV = E_GeV * GEV_TO_EV
    L_eV = L_km * KM_TO_EV
    
    U = construct_pmns(params['theta12'], params['theta13'], params['theta23'], params['deltaCP'])
    
    # Mass matrix in eV^2
    M2 = np.diag([0, params['dm21'], params['dm31']])
    
    # Vacuum Hamiltonian: H_vac = U * M2 * U^\dagger / (2E)
    # E_eV shape handling for broadcast: (N, 1, 1)
    E_reshaped = np.atleast_1d(E_eV)[:, np.newaxis, np.newaxis]
    H_vac = (U @ M2 @ U.conj().T) / (2 * E_reshaped)
    
    # Matter potential
    V_cc = calculate_matter_potential(rho, Ye)
    H_mat = np.zeros_like(H_vac)
    H_mat[:, 0, 0] = V_cc
    
    # Full Hamiltonian
    H_f = H_vac + H_mat
    
    # Numerical Symmetrization (ensure Hermitian)
    H_f = 0.5 * (H_f + H_f.conj().transpose(0, 2, 1))
    
    # Eigen-decomposition
    w, v = np.linalg.eigh(H_f)
    
    # Evolution operator: S = V * exp(-i * w * L) * V^\dagger
    phase = np.exp(-1j * w * L_eV)
    S = v @ (phase[:, :, np.newaxis] * np.eye(3)) @ v.conj().transpose(0, 2, 1)
    
    # Return probabilities (Appearance: P(mu->e), Disappearance: P(mu->mu))
    P_mue = np.abs(S[:, 0, 1])**2  # Index: e=0, mu=1
    P_mumu = np.abs(S[:, 1, 1])**2
    
    return P_mue, P_mumu

def perturbative_solver(E_GeV, L_km, rho, Ye, params):
    """
    Algorithm 2: Compact Perturbative Approximation O(alpha^2)
    """
    E_eV = E_GeV * GEV_TO_EV
    L_eV = L_km * KM_TO_EV
    V_cc = calculate_matter_potential(rho, Ye)
    
    alpha = params['dm21'] / params['dm31']
    Delta = (params['dm31'] * L_eV) / (4 * E_eV)
    A_hat = (2 * E_eV * V_cc) / params['dm31']
    
    s12, c12 = np.sin(params['theta12']), np.cos(params['theta12'])
    s13, c13 = np.sin(params['theta13']), np.cos(params['theta13'])
    s23, c23 = np.sin(params['theta23']), np.cos(params['theta23'])
    dcp = params['deltaCP']
    
    J_CP = s12 * c12 * s23 * c23 * s13 * (c13**2)
    
    # Eq 7: Atmospheric, Interference, and Solar terms
    T_atm = 4 * (s23**2) * (s13**2) * (np.sin((1 - A_hat) * Delta)**2) / ((1 - A_hat)**2)
    
    T_int = 8 * alpha * J_CP * np.cos(Delta + dcp) * \
            (np.sin(A_hat * Delta) / A_hat) * \
            (np.sin((1 - A_hat) * Delta) / (1 - A_hat))
            
    T_sol = (alpha**2) * (c23**2) * (np.sin(2 * params['theta12'])**2) * \
            (np.sin(A_hat * Delta)**2) / (A_hat**2)
            
    P_mue = T_atm + T_int + T_sol
    
    # Simplified Disappearance (for complete solver)
    P_mumu = 1.0 - 4 * (c13**2) * (s23**2) * (1 - (c13**2) * (s23**2)) * (np.sin(Delta)**2)
    
    return P_mue, P_mumu

def hybrid_solver(E_GeV, L_km, rho, Ye, params, epsilon=0.15):
    """
    Algorithm 5: Hybrid Solver
    Dynamically switches to exact solver near the MSW resonance.
    """
    E_eV = E_GeV * GEV_TO_EV
    V_cc = calculate_matter_potential(rho, Ye)
    A_hat = (2 * E_eV * V_cc) / params['dm31']
    
    # Mask for resonance condition
    exact_mask = np.abs(A_hat - 1.0) < epsilon
    
    P_mue = np.zeros_like(E_GeV)
    P_mumu = np.zeros_like(E_GeV)
    
    if np.any(exact_mask):
        P_mue[exact_mask], P_mumu[exact_mask] = exact_solver(E_GeV[exact_mask], L_km, rho, Ye, params)
        
    if np.any(~exact_mask):
        P_mue[~exact_mask], P_mumu[~exact_mask] = perturbative_solver(E_GeV[~exact_mask], L_km, rho, Ye, params)
        
    return P_mue, P_mumu