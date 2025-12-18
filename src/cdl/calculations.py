import numpy as np
import scipy.constants as sc
from typing import Tuple, Dict, Union
from .models import SimulationParameters
from .physics import (
    get_debye_length,
    get_potential_profile,
    get_electric_field_profile,
    get_concentration_profile,
    get_steric_layer_thickness,
    get_differential_capacitance,
    get_electrode_charge_density,
    get_free_energy_entropic,
    get_free_energy_electrostatic,
    get_free_energy_steric,
    get_threshold_potential,
    get_ion_steric_concentration
)
from .solvers import get_charge_balance_potential

def get_potential_split(params: SimulationParameters, potential_difference_mv: float) -> Tuple[float, float]:
    """
    Returns the potential at the left and right electrodes.
    For single electrode, right potential is 0 (bulk).
    
    Args:
        params: Simulation parameters.
        potential_difference_mv: Potential difference in mV.
        
    Returns:
        Tuple (phi_left, phi_right) in mV.
    """
    if params.single_electrode:
        return potential_difference_mv, 0.0
    
    phi_left = get_charge_balance_potential(params, potential_difference_mv)
    phi_right = phi_left - potential_difference_mv
    return phi_left, phi_right

def get_potential_at_x(params: SimulationParameters, x: float, potential_difference_mv: float) -> float:
    """
    Calculates the electric potential at position x.
    
    Args:
        params: Simulation parameters.
        x: Position in meters.
        potential_difference_mv: Potential difference in mV.
        
    Returns:
        Potential in Volts.
    """
    phi_left, phi_right = get_potential_split(params, potential_difference_mv)
    
    if params.single_electrode:
        phi_func = get_potential_profile(params, phi_left)
        return phi_func(x)
    else:
        L_debye = get_debye_length(params)
        L = params.separation_distance * L_debye
        
        if x < 0 or x > L:
            # Allow small tolerance for floating point errors at boundaries
            if x < -1e-12 or x > L + 1e-12:
                raise ValueError(f"Position x={x} is out of bounds [0, {L}]")
            
        phi_func_left = get_potential_profile(params, phi_left)
        phi_func_right = get_potential_profile(params, phi_right)
        
        return phi_func_left(x) + phi_func_right(L - x)

def get_electric_field_at_x(params: SimulationParameters, x: float, potential_difference_mv: float) -> float:
    """
    Calculates the electric field at position x.
    
    Args:
        params: Simulation parameters.
        x: Position in meters.
        potential_difference_mv: Potential difference in mV.
        
    Returns:
        Electric field in V/m.
    """
    phi_left, phi_right = get_potential_split(params, potential_difference_mv)
    
    if params.single_electrode:
        e_func = get_electric_field_profile(params, phi_left)
        return e_func(x)
    else:
        L_debye = get_debye_length(params)
        L = params.separation_distance * L_debye
        
        if x < 0 or x > L:
            if x < -1e-12 or x > L + 1e-12:
                raise ValueError(f"Position x={x} is out of bounds [0, {L}]")
            
        e_func_left = get_electric_field_profile(params, phi_left)
        e_func_right = get_electric_field_profile(params, phi_right)
        
        return e_func_left(x) - e_func_right(L - x)

def get_concentration_at_x(params: SimulationParameters, x: float, potential_difference_mv: float) -> Dict[str, float]:
    """
    Calculates ion concentrations at position x.
    
    Args:
        params: Simulation parameters.
        x: Position in meters.
        potential_difference_mv: Potential difference in mV.
        
    Returns:
        Dictionary with 'cation' and 'anion' concentrations in M (mol/L).
    """
    phi_left, phi_right = get_potential_split(params, potential_difference_mv)
    
    def get_single_interface_concs(p_phi, p_x):
        H = get_steric_layer_thickness(params, p_phi)
        phi_func = get_potential_profile(params, p_phi)
        counter_func = get_concentration_profile(params, p_phi)
        
        vt = sc.k * params.temperature / sc.e
        c_bulk = params.bulk_concentration
        
        if p_phi < 0:
            # Counter = Cation, Co = Anion
            c_cation = counter_func(p_x)
            
            if H > 0 and p_x <= H:
                c_anion = 0.0
            else:
                phi = phi_func(p_x)
                c_anion = c_bulk * np.exp(-params.anion.charge * phi / vt)
        else:
            # Counter = Anion, Co = Cation
            c_anion = counter_func(p_x)
            
            if H > 0 and p_x <= H:
                c_cation = 0.0
            else:
                phi = phi_func(p_x)
                c_cation = c_bulk * np.exp(-params.cation.charge * phi / vt)
                
        return c_cation, c_anion

    if params.single_electrode:
        c_cat, c_an = get_single_interface_concs(phi_left, x)
        return {"cation": c_cat, "anion": c_an}
    else:
        L_debye = get_debye_length(params)
        L = params.separation_distance * L_debye
        
        if x < 0 or x > L:
            if x < -1e-12 or x > L + 1e-12:
                raise ValueError(f"Position x={x} is out of bounds [0, {L}]")
            
        c_cat_L, c_an_L = get_single_interface_concs(phi_left, x)
        c_cat_R, c_an_R = get_single_interface_concs(phi_right, L - x)
        
        c_bulk = params.bulk_concentration
        
        c_cat_total = c_bulk + (c_cat_L - c_bulk) + (c_cat_R - c_bulk)
        c_an_total = c_bulk + (c_an_L - c_bulk) + (c_an_R - c_bulk)
        
        return {"cation": c_cat_total, "anion": c_an_total}

def get_volume_charge_density_at_x(params: SimulationParameters, x: float, potential_difference_mv: float) -> float:
    """
    Calculates volume charge density at position x.
    
    Args:
        params: Simulation parameters.
        x: Position in meters.
        potential_difference_mv: Potential difference in mV.
        
    Returns:
        Charge density in C/m^3.
    """
    concs = get_concentration_at_x(params, x, potential_difference_mv)
    c_cat = concs["cation"]
    c_an = concs["anion"]
    
    rho = (c_cat * params.cation.charge + c_an * params.anion.charge) * 1000 * sc.N_A * sc.e
    return rho

def get_surface_charge_density(params: SimulationParameters, potential_difference_mv: float) -> float:
    """
    Calculates the surface charge density of the working electrode (left).
    
    Args:
        params: Simulation parameters.
        potential_difference_mv: Potential difference in mV.
        
    Returns:
        Surface charge density in C/m^2.
    """
    phi_left, _ = get_potential_split(params, potential_difference_mv)
    return get_electrode_charge_density(params, phi_left)

def get_total_capacitance(params: SimulationParameters, potential_difference_mv: float) -> float:
    """
    Calculates the total differential capacitance.
    
    Args:
        params: Simulation parameters.
        potential_difference_mv: Potential difference in mV.
        
    Returns:
        Capacitance in F/m^2.
    """
    phi_left, phi_right = get_potential_split(params, potential_difference_mv)
    
    c1 = get_differential_capacitance(params, phi_left)
    
    if params.single_electrode:
        return c1
    else:
        c2 = get_differential_capacitance(params, phi_right)
        
        if c1 == 0 or c2 == 0:
            return 0.0
        
        return (c1 * c2) / (c1 + c2)

def get_free_energy(params: SimulationParameters, potential_difference_mv: float) -> Dict[str, float]:
    """
    Calculates free energy components.
    
    Args:
        params: Simulation parameters.
        potential_difference_mv: Potential difference in mV.
        
    Returns:
        Dictionary with 'entropic', 'electrostatic', 'steric', and 'total' free energies in J/m^2.
    """
    phi_left, phi_right = get_potential_split(params, potential_difference_mv)
    
    def get_components(p_phi):
        fen = get_free_energy_entropic(params, p_phi)
        fel = get_free_energy_electrostatic(params, p_phi)
        fst = get_free_energy_steric(params, p_phi)
        return fen, fel, fst
        
    fen1, fel1, fst1 = get_components(phi_left)
    
    if params.single_electrode:
        return {
            "entropic": fen1,
            "electrostatic": fel1,
            "steric": fst1,
            "total": fen1 + fel1 + fst1
        }
    else:
        fen2, fel2, fst2 = get_components(phi_right)
        
        return {
            "entropic": fen1 + fen2,
            "electrostatic": fel1 + fel2,
            "steric": fst1 + fst2,
            "total": (fen1 + fel1 + fst1) + (fen2 + fel2 + fst2)
        }
