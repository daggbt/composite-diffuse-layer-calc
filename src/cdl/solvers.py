import numpy as np
import scipy.constants as sc
from typing import Tuple
from .models import SimulationParameters, Ion
from .physics import (
    get_debye_length,
    get_ion_steric_concentration,
    get_threshold_potential,
    get_effective_dielectric_constant,
    get_differential_capacitance
)

def get_charge_balance_potential(params: SimulationParameters, potential_difference_mv: float) -> float:
    """
    Calculates the potential drop at the working electrode (left side) 
    given the total potential difference.
    
    Args:
        params: Simulation parameters.
        potential_difference_mv: Total potential difference in mV.
        
    Returns:
        Potential at the left electrode in mV.
    """
    pot_diff_v = potential_difference_mv / 1000.0
    
    if params.single_electrode:
        return potential_difference_mv
        
    # Determine counter and co-ions based on potential difference sign
    # Note: The original code logic for counter/co-ion selection in getChargeBalancePotential:
    # if potentialDifference < 0: counter = cath, co = an
    # else: counter = an, co = cath
    
    if pot_diff_v < 0:
        counter_ion = params.cation
        co_ion = params.anion
    else:
        counter_ion = params.anion
        co_ion = params.cation
        
    z1 = counter_ion.charge
    nu1 = 2 * params.bulk_concentration / get_ion_steric_concentration(counter_ion.radius_ang, params.maximum_packing_fraction)
    
    z2 = co_ion.charge
    nu2 = 2 * params.bulk_concentration / get_ion_steric_concentration(co_ion.radius_ang, params.maximum_packing_fraction)
    
    left_potential_v = pot_diff_v / 2
    right_potential_v = left_potential_v - pot_diff_v
    
    # Determine threshold potentials based on the counter-ions at each electrode
    # counter_ion is the ion accumulating at the left electrode (defined above)
    # co_ion is the ion accumulating at the right electrode
    left_threshold_v = get_threshold_potential(counter_ion, params) / 1000.0
    right_threshold_v = get_threshold_potential(co_ion, params) / 1000.0
    
    # Original logic:
    # if abs(leftPotential) > abs(leftPotential_threshold) or abs(rightPotential) > abs(rightPotential_threshold):
    #   leftPotential = ... formula ...
    
    if abs(left_potential_v) > abs(left_threshold_v) or abs(right_potential_v) > abs(right_threshold_v):
        term1 = (sc.k * params.temperature / sc.e)
        numerator = (
            pot_diff_v * z2 * nu1 * sc.e / sc.k / params.temperature + 
            nu1 * (1 - nu2/2)**2 - 
            nu2 * (1 - nu1/2)**2 + 
            nu2 * np.log(2/nu1) - 
            nu1 * np.log(2/nu2)
        )
        denominator = (z2 * nu1 - z1 * nu2)
        left_potential_v = term1 * numerator / denominator
        
    return left_potential_v * 1000.0

def check_separation_validity(params: SimulationParameters, potential_difference_mv: float) -> None:
    """
    Checks if the separation distance is sufficient to accommodate the steric layers.
    Raises ValueError if separation < 2 * max(H_left, H_right).
    """
    if params.single_electrode:
        return
        
    phi_left = get_charge_balance_potential(params, potential_difference_mv)
    phi_right = phi_left - potential_difference_mv
    
    # Note: get_steric_layer_thickness is defined in this file (solvers.py) 
    # but also in physics.py. The one in this file seems to be the one used here.
    # To avoid confusion, we use the one available in the scope.
    H_left = get_steric_layer_thickness(params, phi_left)
    H_right = get_steric_layer_thickness(params, phi_right)
    
    L_debye = get_debye_length(params)
    L_total = params.separation_distance * L_debye
    
    limit = 2 * max(H_left, H_right)
    limit_debye = limit / L_debye
    
    if L_total < limit:
        raise ValueError(
            f"Separation distance insufficient to accommodate steric layers and diffuse region.\n"
            f"Separation: {L_total:.2e} m ({params.separation_distance:.2f} * Debye Length)\n"
            f"Required: > {limit:.2e} m ({limit_debye:.2f} * Debye Length)\n"
            f"H_left: {H_left:.2e} m, H_right: {H_right:.2e} m\n"
            f"Potential Difference: {potential_difference_mv} mV"
        )

def get_steric_layer_thickness(params: SimulationParameters, potential_mv: float) -> float:
    debye_length = get_debye_length(params)
    
    if potential_mv < 0:
        counter_ion = params.cation
    else:
        counter_ion = params.anion
        
    nu = 2 * params.bulk_concentration / get_ion_steric_concentration(counter_ion.radius_ang, params.maximum_packing_fraction)
    threshold_potential_mv = get_threshold_potential(counter_ion, params)
    
    if abs(potential_mv) < abs(threshold_potential_mv):
        return 0.0
    else:
        # thickness = debyeLength*np.sqrt(2*nu)*(-1+0.5*nu+np.sqrt((1-0.5*nu)**2-counter_ion.charge*sc.e*potential/1000/sc.k/temperature+np.log(0.5*nu)))
        term_sqrt_inner = (
            (1 - 0.5 * nu)**2 - 
            counter_ion.charge * sc.e * potential_mv / 1000.0 / sc.k / params.temperature + 
            np.log(0.5 * nu)
        )
        # Ensure non-negative for sqrt
        if term_sqrt_inner < 0:
             term_sqrt_inner = 0 # Should not happen if physics is correct/threshold logic holds
             
        return debye_length * np.sqrt(2 * nu) * (
            -1 + 0.5 * nu + np.sqrt(term_sqrt_inner)
        )

def get_charge_density(params: SimulationParameters, potential_mv: float) -> float:
    debye_length = get_debye_length(params)
    
    if potential_mv < 0:
        counter_ion = params.cation
    else:
        counter_ion = params.anion
        
    nu = 2 * params.bulk_concentration / get_ion_steric_concentration(counter_ion.radius_ang, params.maximum_packing_fraction)
    threshold_potential_mv = get_threshold_potential(counter_ion, params)
    
    rho_bulk = counter_ion.charge * sc.e * sc.N_A * params.bulk_concentration * 1000
    
    epsilon_solvent = params.epsilon_solvent
    
    if abs(potential_mv) < abs(threshold_potential_mv):
        # Gouy-Chapman
        # chargeDensity = np.sqrt(8*sc.N_A*prop.bulkConcentration*epsilon*sc.k*temperature)*np.sinh(abs(counter_ion.charge*sc.e*potential/2000/sc.k/temperature))
        prefactor = np.sqrt(8 * sc.N_A * params.bulk_concentration * 1000 * epsilon_solvent * sc.k * params.temperature)
        # Note: original code used prop.bulkConcentration without *1000 inside sqrt?
        # Let's check original code: np.sqrt(8*sc.N_A*prop.bulkConcentration*epsilon*sc.k*temperature)
        # If bulk is mol/L, then N_A*bulk is particles/L? 
        # Standard GC: sqrt(8 * epsilon * k * T * n_0). n_0 is particles/m^3.
        # So it should be bulk * 1000 * N_A.
        # The original code might have a unit inconsistency or implicit conversion.
        # Wait, in getDebyeLength, it used /1000.
        # Let's assume standard units for the revamp.
        
        # Re-reading original code:
        # chargeDensity = np.sqrt(8*sc.N_A*prop.bulkConcentration*epsilon*sc.k*temperature)*np.sinh(...)
        # If bulk=1, N_A=6e23, epsilon=~7e-10, k=1.38e-23, T=300.
        # 8 * 6e23 * 1 * 7e-10 * 1.38e-23 * 300 approx 8 * 6 * 7 * 1.4 * 300 * 10^-10 approx 1e-5?
        # Standard result is usually C/m^2.
        
        # I will use the correct physics: n0 = bulk * 1000 * N_A
        n0 = params.bulk_concentration * 1000 * sc.N_A
        prefactor = np.sqrt(8 * epsilon_solvent * sc.k * params.temperature * n0)
        
        arg = abs(counter_ion.charge * sc.e * potential_mv / 1000.0) / (2 * sc.k * params.temperature)
        return prefactor * np.sinh(arg) * np.sign(potential_mv) # Wait, original used abs() inside sinh, so result is positive?
        # Original: np.sinh(abs(...)). 
        # Charge density should have sign opposite to electrode? Or same?
        # If potential is positive, electrode is positive, attracts anions (negative charge).
        # But usually we calculate charge on the electrode.
        # If potential > 0, electrode charge > 0.
        # Original code returns positive sinh(abs).
        # But wait, if potential < 0, electrode charge < 0.
        # The original code seems to return a magnitude or assumes specific sign convention.
        # Let's look at the else block:
        # chargeDensity = -2*rho_bulk*debyeLength...
        # rho_bulk depends on counter_ion.charge.
        # If potential > 0, counter is anion (charge < 0). rho_bulk < 0. -2*rho_bulk > 0.
        # So it returns positive charge density for positive potential.
        # So for GC, we should return positive for positive potential.
        # sinh(abs) is positive.
        # But if potential < 0, we want negative charge density.
        # sinh(abs) is positive.
        # So we need np.sign(potential_mv).
        
        # However, let's stick to the original code's behavior if possible, or fix it if it's clearly wrong.
        # Original code:
        # if potential < 0: counter = cath (charge > 0).
        # else: counter = an (charge < 0).
        # GC block: np.sinh(abs(counter_ion.charge ... potential ...)) -> always positive.
        # This implies it returns magnitude?
        # But later: total_chargeDensity += getChargeDensity(pot).
        # If it sums up, signs matter.
        # Let's check the else block again.
        # potential < 0 -> counter=cath (+). rho_bulk (+). -2*(+)*... = (-).
        # potential > 0 -> counter=an (-). rho_bulk (-). -2*(-)*... = (+).
        # So the else block returns signed charge density.
        # The GC block in original code returns POSITIVE always?
        # That would be a bug in the original code for negative potentials.
        # "np.sinh(abs(...))" is always positive.
        # If potential < 0, we expect negative charge.
        # I will fix this by multiplying by np.sign(potential_mv).
        
        return prefactor * np.sinh(arg) * np.sign(potential_mv)

    else:
        # chargeDensity = -2*rho_bulk*debyeLength*np.sqrt(2/nu)*np.sqrt((1-0.5*nu)**2-counter_ion.charge*sc.e*potential/1000/sc.k/temperature+np.log(0.5*nu))
        term_sqrt_inner = (
            (1 - 0.5 * nu)**2 - 
            counter_ion.charge * sc.e * potential_mv / 1000.0 / sc.k / params.temperature + 
            np.log(0.5 * nu)
        )
        if term_sqrt_inner < 0: term_sqrt_inner = 0
        
        return -2 * rho_bulk * debye_length * np.sqrt(2/nu) * np.sqrt(term_sqrt_inner)

def get_composite_diffuse_layer_capacitance(params: SimulationParameters, potential_mv: float) -> Tuple[float, float, float]:
    # Calculate C1 (working electrode)
    # get_differential_capacitance returns F/m^2
    c1_si = get_differential_capacitance(params, potential_mv)
    # Convert to uF/cm^2: 1 F/m^2 = 100 uF/cm^2
    c1_micro = c1_si * 100.0
    
    c2_micro = 0.0
    total_c_micro = 0.0
    
    if not params.single_electrode:
        right_potential_mv = potential_mv - params.potential_difference
        c2_si = get_differential_capacitance(params, right_potential_mv)
        c2_micro = c2_si * 100.0
        
        if c1_micro + c2_micro == 0:
            total_c_micro = 0.0
        else:
            total_c_micro = c1_micro * c2_micro / (c1_micro + c2_micro)
    else:
        total_c_micro = c1_micro
        
    return c1_micro, c2_micro, total_c_micro
