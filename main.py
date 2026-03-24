import numpy as np
import matplotlib.pyplot as plt
import time

# Import our modular package
from utils.constants import get_best_fit, DEFAULT_BASELINE, DEFAULT_DENSITY, DEFAULT_YE
from solvers.engines import exact_solver, perturbative_solver, hybrid_solver
from uncertainty.error_prop import jacobian_propagation

# --- Plotting Configuration ---
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'axes.grid': True})

def generate_paper_figures():
    print("Initializing Neutrino Oscillation Simulator...")
    
    E_grid = np.linspace(0.3, 5.0, 500)
    params = get_best_fit()
    L = DEFAULT_BASELINE
    rho = DEFAULT_DENSITY
    Ye = DEFAULT_YE

    print("1. Computing exact probabilities and uncertainties...")
    # Get base exact probabilities
    P_mue_exact, P_mumu_exact = exact_solver(E_grid, L, rho, Ye, params)
    
    # Calculate uncertainty band (1 sigma) using the ultra-fast Jacobian method
    P_mue_mean, P_mue_std = jacobian_propagation(
        exact_solver, E_grid, params, L_km=L, rho=rho, Ye=Ye
    )

    print("2. Generating Figure: Appearance Channel...")
    plt.figure(figsize=(8, 5))
    plt.plot(E_grid, P_mue_exact, 'k-', lw=2, label='Exact Model')
    plt.fill_between(E_grid, P_mue_mean - P_mue_std, P_mue_mean + P_mue_std, 
                     color='red', alpha=0.3, label=r'$1\sigma$ Uncertainty Band')
    plt.xlabel('Neutrino Energy (GeV)')
    plt.ylabel(r'$P(\nu_\mu \rightarrow \nu_e)$')
    plt.title(f'Appearance Probability (L = {L} km)')
    plt.legend()
    plt.savefig('appearance_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("3. Generating Figure: Disappearance Channel...")
    plt.figure(figsize=(8, 5))
    plt.plot(E_grid, P_mumu_exact, 'b-', lw=2, label='Exact Model')
    plt.xlabel('Neutrino Energy (GeV)')
    plt.ylabel(r'$P(\nu_\mu \rightarrow \nu_\mu)$')
    plt.title(f'Disappearance Probability (L = {L} km)')
    plt.legend()
    plt.savefig('disappearance_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("4. Computing approximations and Hybrid solver...")
    P_mue_approx, _ = perturbative_solver(E_grid, L, rho, Ye, params)
    P_mue_hybrid, _ = hybrid_solver(E_grid, L, rho, Ye, params)

    print("5. Generating Figure: Exact vs Approx Comparison...")
    plt.figure(figsize=(8, 5))
    plt.plot(E_grid, P_mue_exact, 'k-', lw=2, label='Exact Diagonalization')
    plt.plot(E_grid, P_mue_approx, 'r--', lw=2, label=r'Perturbative $O(\alpha^2)$')
    plt.plot(E_grid, P_mue_hybrid, 'g:', lw=3, label='Hybrid Solver')
    plt.xlabel('Neutrino Energy (GeV)')
    plt.ylabel(r'$P(\nu_\mu \rightarrow \nu_e)$')
    plt.title('Solver Methodology Comparison')
    plt.legend()
    plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("6. Generating Figure: Percent Error...")
    error_approx = np.abs(P_mue_approx - P_mue_exact) / P_mue_exact * 100
    error_hybrid = np.abs(P_mue_hybrid - P_mue_exact) / P_mue_exact * 100
    
    plt.figure(figsize=(8, 5))
    plt.plot(E_grid, error_approx, 'r-', lw=2, label='Perturbative Error')
    plt.plot(E_grid, error_hybrid, 'g--', lw=2, label='Hybrid Error')
    plt.axvspan(0.5, 0.8, color='grey', alpha=0.2, label='MSW Resonance Region')
    plt.xlabel('Neutrino Energy (GeV)')
    plt.ylabel('Relative Error (%)')
    plt.title('Approximation Accuracy vs Exact Model')
    plt.yscale('log')
    plt.legend()
    plt.savefig('error_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("7. Running Performance Benchmarks...")
    times = {'Exact': [], 'Approx': [], 'Hybrid': []}
    iters = 1000
    
    # Warmup
    _ = exact_solver(E_grid, L, rho, Ye, params)
    
    t0 = time.perf_counter()
    for _ in range(iters): exact_solver(E_grid, L, rho, Ye, params)
    times['Exact'] = ((time.perf_counter() - t0) / iters) * 1e6 # microseconds
    
    t0 = time.perf_counter()
    for _ in range(iters): perturbative_solver(E_grid, L, rho, Ye, params)
    times['Approx'] = ((time.perf_counter() - t0) / iters) * 1e6
    
    t0 = time.perf_counter()
    for _ in range(iters): hybrid_solver(E_grid, L, rho, Ye, params)
    times['Hybrid'] = ((time.perf_counter() - t0) / iters) * 1e6

    plt.figure(figsize=(7, 5))
    bars = plt.bar(times.keys(), times.values(), color=['darkcyan', 'darkorange', 'forestgreen'], edgecolor='black')
    plt.ylabel(r"Time per evaluation ($\mu$s)")
    plt.title(f"Performance Benchmark (Array Size N={len(E_grid)})")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f} $\mu$s", ha='center', va='bottom', fontweight='bold')
    plt.savefig("performance_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✅ All analyses complete and publication figures generated successfully!")

if __name__ == "__main__":
    generate_paper_figures()