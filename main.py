import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import config

# Add src to path if running directly
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from cdl.models import SimulationParameters
from cdl.database import IONS, SOLVENTS
from cdl.solvers import get_charge_balance_potential, get_composite_diffuse_layer_capacitance, check_separation_validity
from cdl.visualize import plot_spatial_profiles, plot_voltage_dependence

def main():
    parser = argparse.ArgumentParser(description="Composite Diffuse Layer (CDL) Calculator")
    
    parser.add_argument("--cation", type=str, default=config.CATION, help="Name of the cation")
    parser.add_argument("--anion", type=str, default=config.ANION, help="Name of the anion")
    parser.add_argument("--solvent", type=str, default=config.SOLVENT, help="Name of the solvent")
    parser.add_argument("--conc", type=float, default=config.BULK_CONCENTRATION, help="Bulk concentration (M)")
    parser.add_argument("--v-min", type=float, default=-1000.0, help="Minimum voltage (mV)")
    parser.add_argument("--v-max", type=float, default=1000.0, help="Maximum voltage (mV)")
    parser.add_argument("--steps", type=int, default=501, help="Number of voltage steps")
    
    # Single/Double electrode configuration
    # If --single-electrode is present, it's True. If --double-electrode is present, it's False.
    # Default comes from config.
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--single-electrode", action="store_true", default=None, help="Simulate single electrode")
    group.add_argument("--double-electrode", action="store_true", default=None, help="Simulate double electrode")
    
    parser.add_argument("--separation", type=float, default=config.SEPARATION_DISTANCE, help="Separation distance (multiples of Debye length)")
    parser.add_argument("--output", type=str, default=None, help="Output file for plot (e.g. plot.png)")
    
    # New visualization arguments
    parser.add_argument("--plot-profiles", action="store_true", help="Plot spatial profiles (Phi, E, c, rho) at a specific potential")
    parser.add_argument("--plot-voltage-dep", action="store_true", help="Plot voltage dependent properties (C, H, Free Energy)")
    parser.add_argument("--potential", type=float, default=config.POTENTIAL, help="Potential for spatial profiles (mV)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.cation not in IONS:
        print(f"Error: Cation '{args.cation}' not found. Available: {list(IONS.keys())}")
        return
    if args.anion not in IONS:
        print(f"Error: Anion '{args.anion}' not found. Available: {list(IONS.keys())}")
        return
    if args.solvent not in SOLVENTS:
        print(f"Error: Solvent '{args.solvent}' not found. Available: {list(SOLVENTS.keys())}")
        return
        
    # Setup simulation
    cation = IONS[args.cation]
    anion = IONS[args.anion]
    solvent = SOLVENTS[args.solvent]
    
    # Determine single_electrode flag
    if args.single_electrode is not None:
        is_single = True
    elif args.double_electrode is not None:
        is_single = False
    else:
        is_single = config.SINGLE_ELECTRODE

    params = SimulationParameters(
        electrolyte_ions=(cation, anion),
        solvent=solvent,
        bulk_concentration=args.conc,
        single_electrode=is_single,
        separation_distance=args.separation
    )
    
    # Ensure output directory exists
    if config.OUTPUT_FOLDER:
        os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)
        
    def get_output_path(filename):
        if os.path.dirname(filename):
            return filename
        return os.path.join(config.OUTPUT_FOLDER, filename)
    
    # Handle visualization modes
    if args.plot_profiles:
        filename = args.output if args.output else f"spatial_profiles_{args.potential}mV.png"
        filename = get_output_path(filename)
        plot_spatial_profiles(params, args.potential, filename)
        return

    if args.plot_voltage_dep:
        filename = args.output if args.output else "voltage_dependence.png"
        filename = get_output_path(filename)
        plot_voltage_dependence(params, filename, args.v_min, args.v_max, args.steps)
        return
    
    print(f"Running simulation for {cation.name}/{anion.name} in {solvent.name}...")
    print(f"Voltage range: {args.v_min} to {args.v_max} mV")
    
    voltages = np.linspace(args.v_min, args.v_max, args.steps)
    capacitances = []
    
    for v in voltages:
        # Update potential difference in params if needed, or pass it
        params.potential_difference = v 
        
        if not args.single_electrode:
            check_separation_validity(params, v)
        
        # 1. Solve for charge balance potential (potential at the working electrode)
        pot_balance = get_charge_balance_potential(params, v)
        
        # 2. Calculate capacitance at this potential
        c1, c2, c_total = get_composite_diffuse_layer_capacitance(params, pot_balance)
        
        if args.single_electrode:
            capacitances.append(c1)
        else:
            capacitances.append(c_total)
            
    max_cap = max(capacitances)
    print(f"Maximum Capacitance: {max_cap:.4f} uF/cm^2")

    # Plotting
    plt.figure(figsize=(8.5, 5.1))
    plt.plot(voltages / 1000.0, capacitances, label='CDL Model')
    plt.xlabel('Potential (V)')
    plt.ylabel(r'Capacitance ($\mu F/cm^2$)') 
    plt.title(f'Differential Capacitance: {cation.name}-{anion.name} in {solvent.name}')
    plt.grid(True)
    plt.legend()
    
    if args.output:
        filename = get_output_path(args.output)
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
    else:
        plt.show()

    # Generate spatial profiles
    profile_filename = get_output_path(f"spatial_profiles_{args.potential}mV.png")
    plot_spatial_profiles(params, args.potential, profile_filename)

if __name__ == "__main__":
    main()
