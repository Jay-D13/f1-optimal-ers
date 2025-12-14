"""
Test Co-state Extraction from Your Spatial NLP Solver

This script tests the hybrid PMP approach on your existing solver.

WHAT THIS DOES:
1. Loads a track (Monza - simple, mostly straights)
2. Runs Forward-Backward to get velocity limits
3. Solves with your SpatialNLP solver (WITH co-state extraction enabled)
4. Analyzes the PMP structure
5. Creates visualization plots

Run from your repository root:
    python random_scripts/test_costate_extraction.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from solvers import ForwardBackwardSolver, SpatialNLPSolver
from models import VehicleDynamicsModel, F1TrackModel
from config import ERSConfig, VehicleConfig


def main():
    print("="*70)
    print("  TESTING CO-STATE EXTRACTION (HYBRID PMP APPROACH)")
    print("="*70)
    
    # =========================================================================
    # STEP 1: Setup
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: CONFIGURATION")
    print("="*70)
    
    track_name = 'monza'  # Start with simple track (mostly straights)
    
    # Load configs
    ers_config = ERSConfig()
    vehicle_config = VehicleConfig.for_monza()
    
    print(f"\nTrack: {track_name}")
    print(f"ERS: {ers_config.max_deployment_power/1000:.0f}kW, "
          f"{ers_config.battery_usable_energy/1e6:.1f}MJ/lap")
    
    # =========================================================================
    # STEP 2: Load Track
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: LOAD TRACK")
    print("="*70)
    
    track = F1TrackModel(year=2024, gp=track_name, ds=5.0)
    
    # Try TUMFTM first
    tumftm_path = Path(f'data/racelines/{track_name}.csv')
    if tumftm_path.exists():
        print(f"Loading TUMFTM raceline: {tumftm_path}")
        track.load_from_tumftm_raceline(str(tumftm_path))
    else:
        print(f"Loading from FastF1...")
        try:
            track.load_from_fastf1(driver='VER')
        except Exception as e:
            print(f"⚠ FastF1 failed: {e}")
            print("Please provide TUMFTM raceline in data/racelines/")
            return
    
    print(f"✓ Track loaded: {track.total_length/1000:.2f} km")
    
    # =========================================================================
    # STEP 3: Get Velocity Limits (Forward-Backward)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: FORWARD-BACKWARD SOLVER (Velocity Limits)")
    print("="*70)
    
    vehicle = VehicleDynamicsModel(vehicle_config, ers_config)
    
    fb_solver = ForwardBackwardSolver(vehicle, track, ers_config)
    fb_result = fb_solver.solve()
    
    print(f"✓ Velocity limits computed")
    print(f"  Max speed: {fb_result.v.max()*3.6:.1f} km/h")
    print(f"  Min speed: {fb_result.v.min()*3.6:.1f} km/h")
    
    # =========================================================================
    # STEP 4: Solve NLP WITH Co-state Extraction
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: SPATIAL NLP WITH CO-STATE EXTRACTION")
    print("="*70)
    
    nlp_solver = SpatialNLPSolver(vehicle, track, ers_config, ds=5.0)
    
    print("\nSolving with co-state extraction enabled...")
    print("(This will extract PMP co-states from IPOPT dual variables)")
    
    # CRITICAL: This is where we test the new feature
    result = nlp_solver.solve(
        v_limit_profile=fb_result.v,  # Use .v instead of .v_max
        initial_soc=0.5,
        final_soc_min=0.3,
        energy_limit=4e6,
        extract_costates=True  # <-- THE NEW FEATURE
    )
    
    print(f"\n✓ NLP solved")
    print(f"  Lap time: {result.lap_time:.3f}s")
    print(f"  Status: {result.solver_status}")
    
    # =========================================================================
    # STEP 5: Check if Co-states Were Extracted
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: CO-STATE EXTRACTION VERIFICATION")
    print("="*70)
    
    if hasattr(result, 'costates') and result.costates is not None:
        print("\n✓✓ CO-STATES SUCCESSFULLY EXTRACTED!")
        
        costates = result.costates
        
        print(f"\nCo-state Statistics:")
        print(f"  Number of points: {len(costates.s)}")
        print(f"  λ_v range: [{costates.lambda_v.min():.3e}, {costates.lambda_v.max():.3e}]")
        print(f"  λ_SOC range: [{costates.lambda_SOC.min():.3e}, {costates.lambda_SOC.max():.3e}]")
        print(f"  Bang-bang control: {costates.bang_bang_pct:.1f}%")
        
        if costates.bang_bang_pct > 90:
            print("\n  ✓✓ EXCELLENT! >90% bang-bang control")
            print("     This confirms ETH Zurich's findings!")
            print("     Your solution is PMP-optimal!")
        elif costates.bang_bang_pct > 70:
            print("\n  ✓ GOOD! >70% bang-bang control")
            print("    Solution is approximately PMP-optimal")
        else:
            print("\n  ⚠ Low bang-bang percentage")
            print("    May need to check constraints")
        
        # Create visualization
        print("\n" + "="*70)
        print("STEP 6: CREATING VISUALIZATION")
        print("="*70)
        
        create_pmp_visualization(result, costates)
        
        print("\n✓ Plots saved to results/pmp_analysis.png")
        
    else:
        print("\n✗ CO-STATES NOT EXTRACTED")
        print("\nPossible reasons:")
        print("  1. IPOPT not computing dual variables")
        print("  2. Solver failed to converge")
        print("  3. Need to modify solver options")
        print("\nTroubleshooting:")
        print("  - Check that 'ipopt.print_level' >= 3")
        print("  - Ensure solver converged successfully")
        print("  - Try running spatial_nlp_with_costates.py modifications")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)


def create_pmp_visualization(result, costates):
    """Create comprehensive PMP visualization"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    s_km = result.s / 1000
    s_km_ctrl = s_km[:-1]  # Controls are N points, states are N+1
    
    # 1. Co-states
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(s_km, costates.lambda_v, 'b-', linewidth=2, label='λ_v (velocity)')
    ax1.plot(s_km, costates.lambda_SOC, 'g-', linewidth=2, label='λ_SOC (energy price)')
    ax1.set_ylabel('Co-state Value', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Distance (km)', fontsize=10)
    ax1.set_title('PMP Co-states (From NLP Dual Variables)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Kinetic energy co-state
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(s_km, costates.lambda_kin, 'purple', linewidth=2)
    ax2.set_ylabel('λ_kin = λ_v × v', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Distance (km)', fontsize=10)
    ax2.set_title('Kinetic Energy Co-state', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Switching function (KEY PMP RESULT)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(s_km, costates.sigma_ERS, 'purple', linewidth=2.5, label='σ_ERS')
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=1)
    
    # Fill regions
    ax3.fill_between(s_km, 0, costates.sigma_ERS,
                      where=costates.deploy_mask,
                      color='green', alpha=0.3, label=f'Deploy ({costates.deploy_mask.sum()/len(s_km)*100:.1f}%)')
    ax3.fill_between(s_km, costates.sigma_ERS, 0,
                      where=costates.recover_mask,
                      color='red', alpha=0.3, label=f'Recover ({costates.recover_mask.sum()/len(s_km)*100:.1f}%)')
    
    ax3.set_ylabel('Switching Function σ', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Distance (km)', fontsize=10)
    ax3.set_title(f'ERS Switching Function (σ = λ_v/v - λ_SOC/E_batt) | Bang-bang: {costates.bang_bang_pct:.1f}%',
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Add annotation
    ax3.text(0.02, 0.98, 'σ > 0 → Deploy ERS\nσ < 0 → Recover ERS',
             transform=ax3.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Actual ERS Power (verify matches PMP prediction)
    ax4 = fig.add_subplot(gs[2, 0])
    P_ers_kw = result.P_ers_opt / 1000
    ax4.fill_between(s_km_ctrl, 0, P_ers_kw,
                      where=(P_ers_kw > 0),
                      color='green', alpha=0.6, label='Deploy')
    ax4.fill_between(s_km_ctrl, 0, P_ers_kw,
                      where=(P_ers_kw < 0),
                      color='red', alpha=0.6, label='Recover')
    ax4.axhline(y=120, color='gray', linestyle='--', alpha=0.5, label='Limits')
    ax4.axhline(y=-120, color='gray', linestyle='--', alpha=0.5)
    ax4.set_ylabel('ERS Power (kW)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Distance (km)', fontsize=10)
    ax4.set_title('Actual ERS Power (Should Match σ Regions)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Battery Shadow Price
    ax5 = fig.add_subplot(gs[2, 1])
    shadow_price = -costates.lambda_SOC
    ax5.plot(s_km, shadow_price, 'green', linewidth=2)
    ax5.fill_between(s_km, 0, shadow_price, alpha=0.3, color='green')
    ax5.set_ylabel('-λ_SOC (Energy Value)', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Distance (km)', fontsize=10)
    ax5.set_title('Battery Energy Shadow Price (Higher = More Valuable)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Add interpretation
    avg_shadow = np.mean(np.abs(shadow_price))
    ax5.text(0.02, 0.98, f'Avg: {avg_shadow:.3e}\nHigh → Save energy\nLow → Use freely',
             transform=ax5.transAxes, fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # 6. Velocity and SOC
    ax6 = fig.add_subplot(gs[3, 0])
    ax6_twin = ax6.twinx()
    
    l1 = ax6.plot(s_km, result.v_opt * 3.6, 'b-', linewidth=2, label='Velocity')
    ax6.set_ylabel('Velocity (km/h)', fontsize=11, fontweight='bold', color='b')
    ax6.tick_params(axis='y', labelcolor='b')
    
    l2 = ax6_twin.plot(s_km, result.soc_opt * 100, 'g-', linewidth=2, label='SOC')
    ax6_twin.set_ylabel('SOC (%)', fontsize=11, fontweight='bold', color='g')
    ax6_twin.tick_params(axis='y', labelcolor='g')
    
    ax6.set_xlabel('Distance (km)', fontsize=10)
    ax6.set_title('Velocity & Battery State of Charge', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Combined legend
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax6.legend(lns, labs, fontsize=10, loc='best')
    
    # 7. Control comparison scatter
    ax7 = fig.add_subplot(gs[3, 1])
    
    # Compare sigma prediction vs actual control
    sigma_at_ctrl = costates.sigma_ERS[:-1]  # Match control array size
    
    # Normalize for comparison
    P_ers_normalized = P_ers_kw / 120  # Normalize to [-1, 1]
    sigma_normalized = np.sign(sigma_at_ctrl)  # Just sign
    
    ax7.scatter(sigma_at_ctrl, P_ers_kw, alpha=0.5, s=10, c='blue')
    ax7.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax7.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax7.set_xlabel('Switching Function σ', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Actual ERS Power (kW)', fontsize=11, fontweight='bold')
    ax7.set_title('PMP Prediction vs Actual Control', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # Add quadrant labels
    ax7.text(0.7, 0.95, 'σ>0, P>0\nCorrect!', transform=ax7.transAxes,
             fontsize=9, color='green', fontweight='bold',
             ha='center', va='top')
    ax7.text(0.3, 0.05, 'σ<0, P<0\nCorrect!', transform=ax7.transAxes,
             fontsize=9, color='green', fontweight='bold',
             ha='center', va='bottom')
    
    plt.suptitle('Hybrid PMP Analysis: Co-states Extracted from Spatial NLP',
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Save
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'pmp_analysis.png', dpi=150, bbox_inches='tight')
    
    print(f"  ✓ Saved: results/pmp_analysis.png")


if __name__ == "__main__":
    main()