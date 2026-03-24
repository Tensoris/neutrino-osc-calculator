import numpy as np
from utils.constants import G_F, m_N, GEV_TO_EV

def construct_pmns(t12, t13, t23, dcp):
    """
    Constructs the 3x3 PMNS mixing matrix.
    Supports scalar or vectorized inputs.
    """
    c12, s12 = np.cos(t12), np.sin(t12)
    c13, s13 = np.cos(t13), np.sin(t13)
    c23, s23 = np.cos(t23), np.sin(t23)
    
    # Phase factors
    edcp = np.exp(-1j * dcp)
    edcp_conj = np.exp(1j * dcp)
    
    # PMNS elements
    Ue1 = c12 * c13
    Ue2 = s12 * c13
    Ue3 = s13 * edcp
    
    Umu1 = -s12 * c23 - c12 * s23 * s13 * edcp_conj
    Umu2 = c12 * c23 - s12 * s23 * s13 * edcp_conj
    Umu3 = s23 * c13
    
    Utau1 = s12 * s23 - c12 * c23 * s13 * edcp_conj
    Utau2 = -c12 * s23 - s12 * c23 * s13 * edcp_conj
    Utau3 = c23 * c13
    
    # Construct matrix and transpose if vectorized to shape (..., 3, 3)
    U = np.array([[Ue1, Ue2, Ue3],
                  [Umu1, Umu2, Umu3],
                  [Utau1, Utau2, Utau3]])
    
    if U.ndim > 2:
        return np.transpose(U, (2, 0, 1))
    return U

def calculate_matter_potential(rho, Ye):
    """Calculates the CC matter potential V_cc in eV."""
    # V_cc = sqrt(2) * G_F * N_e
    # N_e = rho * Y_e / m_N
    # Convert G_F to eV^-2 (G_F_GeV * 1e-18)
    G_F_eV = G_F * 1e-18
    m_N_eV = m_N * GEV_TO_EV
    
    # Density in eV^4 (1 g/cm^3 ~ 4.3e-18 GeV^3 ~ 4.3e9 eV^3)
    # Using the standard practical conversion directly:
    # V_cc [eV] = 7.56e-14 * rho[g/cm^3] * Ye
    return 7.56e-14 * rho * Ye