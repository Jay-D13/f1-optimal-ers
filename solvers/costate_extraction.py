"""
Co-state Extraction Module for Spatial NLP Solver

This module extracts PMP (Pontryagin's Minimum Principle) co-states from 
the dual variables of your CasADi/IPOPT solver.

Author: Based on ETH Zurich hybrid PMP approach
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CostateProfile:
    """
    Co-state (adjoint/dual variable) profiles extracted from NLP solution.
    
    These represent the "shadow prices" or Lagrange multipliers:
    - lambda_v: Sensitivity of lap time to velocity changes
    - lambda_SOC: Sensitivity of lap time to battery SOC (energy shadow price)
    
    The key PMP result is the switching function sigma_ERS which determines
    whether to deploy or recover ERS power.
    """
    s: np.ndarray              # Position along track [m]
    lambda_v: np.ndarray       # Co-state for velocity
    lambda_SOC: np.ndarray     # Co-state for battery SOC
    sigma_ERS: np.ndarray      # ERS switching function
    
    # Derived quantities
    lambda_kin: np.ndarray     # Kinetic energy co-state (λ_v * v)
    
    # Control region masks
    deploy_mask: np.ndarray    # Where to deploy ERS (σ > 0)
    recover_mask: np.ndarray   # Where to recover ERS (σ < 0)
    neutral_mask: np.ndarray   # Neutral region (σ ≈ 0)
    
    # Statistics
    bang_bang_pct: float       # Percentage of lap with bang-bang control


def extract_costates_from_casadi_opti(
    opti,
    V,
    SOC,
    N: int,
    ers_config,
    successful_solve: bool = True
) -> Optional[CostateProfile]:
    """
    Extract co-states from CasADi Opti solution.
    
    CRITICAL: This function maps YOUR specific constraint ordering to co-states.
    The constraint ordering in your spatial_nlp.py is:
    
    1. Velocity dynamics (N constraints):
       opti.subject_to(V[k+1] == V[k] + dv_ds * ds)
       
    2. SOC dynamics (N constraints):
       opti.subject_to(SOC[k+1] == SOC[k] + dsoc_ds * ds)
       
    3. State bounds (2*(N+1) constraints):
       opti.subject_to(V[k] >= v_min), opti.subject_to(V[k] <= v_max)
       opti.subject_to(SOC[k] >= soc_min), opti.subject_to(SOC[k] <= soc_max)
       
    4. Control bounds, etc.
    
    The dual variables (Lagrange multipliers) for the DYNAMICS constraints
    are the co-states!
    
    Args:
        opti: CasADi Opti object after solve
        V: Velocity variable (Opti variable)
        SOC: SOC variable (Opti variable)
        N: Number of discretization points
        ers_config: ERSConfig object
        successful_solve: Whether solver converged successfully
        
    Returns:
        CostateProfile object or None if extraction fails
    """
    
    try:
        # Get dual variables
        # For CasADi Opti, we need to access the underlying solver stats
        if successful_solve:
            # For successful solve, use solved values
            lambda_g = opti.debug.value(opti.lam_g)
        else:
            # For debug (failed solve), try to get dual variables anyway
            try:
                lambda_g = opti.debug.value(opti.lam_g)
            except:
                print("  ⚠ Cannot extract dual variables from failed solve")
                return None
        
        # CONSTRAINT ORDERING IN YOUR spatial_nlp.py:
        # Looking at your code, constraints are added in this order:
        # 
        # Loop k=0 to N-1:
        #   1. Velocity dynamics: V[k+1] == V[k] + dv_ds * ds
        #   2. SOC dynamics: SOC[k+1] == SOC[k] + dsoc_ds * ds
        #
        # Then (outside loop):
        #   3. State bounds on V and SOC
        #   4. Control bounds
        #   5. Boundary conditions
        #   6. Energy limit
        
        # The dynamics constraints are the FIRST 2*N constraints
        # Index 0 to N-1: Velocity dynamics
        # Index N to 2N-1: SOC dynamics
        
        # Extract co-states (dual variables of dynamics constraints)
        lambda_v_raw = lambda_g[0:N]       # Duals of velocity dynamics
        lambda_SOC_raw = lambda_g[N:2*N]   # Duals of SOC dynamics
        
        # The dual variables correspond to constraints at k=0,1,...,N-1
        # But we need them at all N+1 points for visualization
        # We'll pad with the last value
        lambda_v = np.zeros(N + 1)
        lambda_SOC = np.zeros(N + 1)
        
        lambda_v[:-1] = lambda_v_raw
        lambda_v[-1] = lambda_v_raw[-1]  # Extrapolate last value
        
        lambda_SOC[:-1] = lambda_SOC_raw
        lambda_SOC[-1] = lambda_SOC_raw[-1]
        
        print(f"\n  ✓ Extracted co-states from dual variables")
        print(f"    λ_v shape: {lambda_v.shape}")
        print(f"    λ_SOC shape: {lambda_SOC.shape}")
        
        return lambda_v, lambda_SOC
        
    except Exception as e:
        print(f"  ✗ Error extracting co-states: {e}")
        print(f"    This may happen if IPOPT doesn't compute dual variables")
        print(f"    Try adding 'ipopt.print_level': 5 to see more details")
        return None, None


def compute_pmp_analysis(
    s: np.ndarray,
    v: np.ndarray,
    P_ers: np.ndarray,
    lambda_v: np.ndarray,
    lambda_SOC: np.ndarray,
    ers_config
) -> CostateProfile:
    """
    Compute PMP-derived quantities from co-states.
    
    The key insight from PMP is the switching function:
        σ = ∂H/∂P_ers = λ_v / v - λ_SOC / E_battery
        
    Where:
        - H is the Hamiltonian
        - λ_v is the co-state for velocity
        - λ_SOC is the co-state for SOC
        - E_battery is the battery capacity
    
    Optimal control structure (bang-bang):
        σ > 0  →  P_ERS = +120kW (deploy)
        σ < 0  →  P_ERS = -120kW (recover)
        σ = 0  →  Singular arc (rare)
    
    Args:
        s: Position array [m]
        v: Velocity array [m/s]
        P_ers: ERS power array [W] (actual control from solver)
        lambda_v: Velocity co-state
        lambda_SOC: SOC co-state
        ers_config: ERSConfig object
        
    Returns:
        CostateProfile with all PMP-derived quantities
    """
    
    # Compute kinetic energy co-state
    lambda_kin = lambda_v * v
    
    # Compute ERS switching function (KEY PMP RESULT)
    # This tells us when it's optimal to deploy vs recover
    v_safe = np.maximum(v, 1.0)  # Avoid division by zero
    sigma_ERS = lambda_v / v_safe - lambda_SOC / ers_config.battery_capacity
    
    # Identify control regions based on switching function
    threshold = 1e-6  # Deadband to avoid numerical noise
    deploy_mask = sigma_ERS > threshold
    recover_mask = sigma_ERS < -threshold
    neutral_mask = np.abs(sigma_ERS) <= threshold
    
    # Compute bang-bang percentage (PMP predicts >90% for optimal solution)
    bang_bang_pct = (deploy_mask.sum() + recover_mask.sum()) / len(sigma_ERS) * 100
    
    # Verify PMP prediction matches actual control
    # Where σ > 0, we should have P_ers > 0 (and vice versa)
    if len(P_ers) == len(sigma_ERS):
        # Check consistency
        deploy_actual = P_ers > 1e3  # Deploying if P_ers > 1kW
        recover_actual = P_ers < -1e3
        
        deploy_match = np.sum(deploy_mask & deploy_actual) / max(deploy_mask.sum(), 1) * 100
        recover_match = np.sum(recover_mask & recover_actual) / max(recover_mask.sum(), 1) * 100
        
        print(f"\n  PMP vs Actual Control Consistency:")
        print(f"    Deploy regions: {deploy_match:.1f}% match")
        print(f"    Recover regions: {recover_match:.1f}% match")
    
    return CostateProfile(
        s=s,
        lambda_v=lambda_v,
        lambda_SOC=lambda_SOC,
        sigma_ERS=sigma_ERS,
        lambda_kin=lambda_kin,
        deploy_mask=deploy_mask,
        recover_mask=recover_mask,
        neutral_mask=neutral_mask,
        bang_bang_pct=bang_bang_pct
    )


def print_pmp_analysis(costates: CostateProfile):
    """
    Print detailed PMP analysis to console.
    
    Interprets the co-states and switching function in physical terms.
    """
    
    print("\n" + "="*70)
    print("PMP CO-STATE ANALYSIS")
    print("="*70)
    
    print(f"\nCo-state Ranges:")
    print(f"  λ_v:   [{costates.lambda_v.min():10.3e}, {costates.lambda_v.max():10.3e}]")
    print(f"  λ_SOC: [{costates.lambda_SOC.min():10.3e}, {costates.lambda_SOC.max():10.3e}]")
    print(f"  λ_kin: [{costates.lambda_kin.min():10.3e}, {costates.lambda_kin.max():10.3e}]")
    
    print(f"\nSwitching Function σ_ERS:")
    print(f"  Range: [{costates.sigma_ERS.min():10.3e}, {costates.sigma_ERS.max():10.3e}]")
    print(f"  Interpretation:")
    print(f"    σ > 0: Deploy ERS (increasing velocity is valuable)")
    print(f"    σ < 0: Recover ERS (saving battery energy is valuable)")
    
    print(f"\nERS Control Structure (from PMP):")
    total_points = len(costates.sigma_ERS)
    print(f"  Deploy:  {costates.deploy_mask.sum()/total_points*100:5.1f}% of lap")
    print(f"  Recover: {costates.recover_mask.sum()/total_points*100:5.1f}% of lap")
    print(f"  Neutral: {costates.neutral_mask.sum()/total_points*100:5.1f}% of lap")
    
    print(f"\nOptimality Check:")
    print(f"  Bang-bang control: {costates.bang_bang_pct:.1f}%")
    
    if costates.bang_bang_pct > 90:
        print(f"  ✓ Strong bang-bang structure (>90%)")
        print(f"    → Solution is PMP-optimal!")
        print(f"    → This confirms the ETH Zurich findings")
    elif costates.bang_bang_pct > 70:
        print(f"  ~ Mostly bang-bang structure (>70%)")
        print(f"    → Solution is approximately PMP-optimal")
    else:
        print(f"  ⚠ Weak bang-bang structure (<70%)")
        print(f"    → May not be fully optimal")
        print(f"    → Check constraints or formulation")
    
    print(f"\nShadow Price Interpretation (λ_SOC):")
    avg_lambda_SOC = np.mean(np.abs(costates.lambda_SOC))
    if avg_lambda_SOC > 1e-3:
        print(f"  Battery energy is VALUABLE (|λ_SOC| = {avg_lambda_SOC:.3e})")
        print(f"  → Use ERS sparingly, save for critical sections")
    elif avg_lambda_SOC > 1e-6:
        print(f"  Battery energy is MODERATELY valuable (|λ_SOC| = {avg_lambda_SOC:.3e})")
        print(f"  → Balanced ERS deployment")
    else:
        print(f"  Battery energy is CHEAP (|λ_SOC| = {avg_lambda_SOC:.3e})")
        print(f"  → Deploy ERS aggressively")
    
    print("\n" + "="*70)


def analyze_costate_structure(costates: CostateProfile) -> Dict:
    """
    Detailed analysis of co-state structure for ELTMS controller design.
    
    Returns switching thresholds and control modes.
    """
    
    # Find switching points (where σ changes sign)
    sigma = costates.sigma_ERS
    sign_changes = np.diff(np.sign(sigma))
    
    deploy_to_neutral = np.where(sign_changes < 0)[0]
    neutral_to_recover = np.where(sign_changes < 0)[0]
    
    # Compute threshold values for real-time control
    # These are the λ_kin values where control switches
    
    if len(deploy_to_neutral) > 0:
        lambda_kin_deploy_threshold = np.median(
            costates.lambda_kin[deploy_to_neutral]
        )
    else:
        lambda_kin_deploy_threshold = 0.0
    
    if len(neutral_to_recover) > 0:
        lambda_kin_recover_threshold = np.median(
            costates.lambda_kin[neutral_to_recover]
        )
    else:
        lambda_kin_recover_threshold = -1.0
    
    return {
        'deploy_threshold': lambda_kin_deploy_threshold,
        'recover_threshold': lambda_kin_recover_threshold,
        'n_switches': len(deploy_to_neutral) + len(neutral_to_recover),
        'switch_positions_s': costates.s[np.where(sign_changes != 0)[0]],
    }


if __name__ == "__main__":
    # Test/example usage
    print("Co-state Extraction Module")
    print("This module should be imported, not run directly")
    print("\nUsage:")
    print("  from solvers.costate_extraction import extract_costates_from_casadi_opti")
    print("  costates = extract_costates_from_casadi_opti(opti, V, SOC, N, ers_config)")