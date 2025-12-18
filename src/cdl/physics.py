import numpy as np
import scipy.constants as sc
from scipy.integrate import quad
from typing import Tuple, Callable
from .models import SimulationParameters, Ion

def get_debye_length(params: SimulationParameters) -> float:
    """
    Calculates the Debye length.
    
    Returns:
        Debye length in meters.
    """
    epsilon = params.epsilon_solvent
    kt = sc.k * params.temperature
    e2 = sc.e ** 2
    
    # Assuming symmetric concentration for now as per original code structure
    sum_nz2 = params.bulk_concentration * (params.cation.charge**2 + params.anion.charge**2)
    
    # The factor 1000 * N_A converts mol/L to particles/m^3
    denominator = e2 * sum_nz2 * 1000 * sc.N_A
    
    return np.sqrt(epsilon * kt / denominator)

def get_ion_steric_concentration(ion_radius_ang: float, max_packing_fraction: float) -> float:
    """
    Calculates the steric concentration limit for an ion.
    
    Args:
        ion_radius_ang: Ion radius in Angstroms.
        max_packing_fraction: Maximum packing fraction.
        
    Returns:
        Steric concentration in M (mol/L).
    """
    ion_radius = ion_radius_ang * 1e-10
    ion_volume = 4 * np.pi * ion_radius**3 / 3
    
    return max_packing_fraction / ion_volume / 1000 / sc.N_A

def get_threshold_potential(ion: Ion, params: SimulationParameters) -> float:
    """
    Calculates the threshold potential for an ion.
    
    Returns:
        Threshold potential in mV.
    """
    steric_conc = get_ion_steric_concentration(ion.radius_ang, params.maximum_packing_fraction)
    
    val = -sc.k * params.temperature * np.log(steric_conc / params.bulk_concentration) / sc.e / ion.charge
    return val * 1000

def get_effective_dielectric_constant(params: SimulationParameters, potential_mv: float) -> float:
    """
    Calculates the effective dielectric constant.
    
    Args:
        params: Simulation parameters.
        potential_mv: Potential in mV.
        
    Returns:
        Effective dielectric constant (relative permittivity).
    """
    if potential_mv < 0:
        counter_ion = params.cation
    else:
        counter_ion = params.anion
        
    epsilon_solvent = params.epsilon_solvent # epsilon_0 * dielectric
    
    ion_radius_m = counter_ion.radius_ang * 1e-10
    ion_volume_m3 = 4 * np.pi * ion_radius_m**3 / 3
    
    polarizability_factor = counter_ion.polarizability / 3 / epsilon_solvent / ion_volume_m3
    
    steric_conc = get_ion_steric_concentration(counter_ion.radius_ang, params.maximum_packing_fraction)
    
    reduced_dielectric = params.solvent.dielectric * 1000 * sc.N_A * steric_conc * ion_volume_m3 * polarizability_factor
    
    if reduced_dielectric == 0:
        return params.solvent.dielectric
        
    return reduced_dielectric

def get_steric_layer_thickness(params: SimulationParameters, potential_mv: float) -> float:
    """
    Calculates the steric layer thickness (H).
    
    Args:
        params: Simulation parameters.
        potential_mv: Electrode potential in mV.
        
    Returns:
        Steric layer thickness in meters.
    """
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
        # Eq. SLT in manuscript
        # term_potential = -z * e * Phi / kT
        term_potential = -counter_ion.charge * sc.e * (potential_mv / 1000.0) / (sc.k * params.temperature)
        
        term_sqrt_inner = (
            (1 - 0.5 * nu)**2 + 
            term_potential - 
            np.log(2/nu)
        )
        
        if term_sqrt_inner < 0:
             term_sqrt_inner = 0 
             
        return debye_length * np.sqrt(2 * nu) * (
            -1 + 0.5 * nu + np.sqrt(term_sqrt_inner)
        )

def get_electrode_charge_density(params: SimulationParameters, potential_mv: float) -> float:
    """
    Calculates the electrode charge density (sigma).
    
    Args:
        params: Simulation parameters.
        potential_mv: Electrode potential in mV.
        
    Returns:
        Charge density in C/m^2.
    """
    debye_length = get_debye_length(params)
    
    if potential_mv < 0:
        counter_ion = params.cation
    else:
        counter_ion = params.anion
        
    nu = 2 * params.bulk_concentration / get_ion_steric_concentration(counter_ion.radius_ang, params.maximum_packing_fraction)
    threshold_potential_mv = get_threshold_potential(counter_ion, params)
    
    # rho_bulk = z * e * c_bulk
    rho_bulk = counter_ion.charge * sc.e * params.bulk_concentration * 1000 * sc.N_A
    
    if abs(potential_mv) < abs(threshold_potential_mv):
        # Gouy-Chapman
        # sigma = (2 * epsilon * kappa * kT / (z * e)) * sinh(z * e * Phi / (2 * kT))
        # kappa = 1/lambda_D
        # prefactor = 2 * epsilon * kT / (z * e * lambda_D)
        
        epsilon = params.epsilon_solvent
        kt = sc.k * params.temperature
        z = abs(counter_ion.charge)
        e = sc.e
        
        prefactor = 2 * epsilon * kt / (z * e * debye_length)
        arg = z * e * (potential_mv / 1000.0) / (2 * kt)
        
        return prefactor * np.sinh(arg)

    else:
        # Eq. SLT in manuscript
        term_potential = -counter_ion.charge * sc.e * (potential_mv / 1000.0) / (sc.k * params.temperature)
        
        term_sqrt_inner = (
            (1 - 0.5 * nu)**2 + 
            term_potential - 
            np.log(2/nu)
        )
        
        if term_sqrt_inner < 0: term_sqrt_inner = 0
        
        # sigma = -2 * rho_bulk * lambda_D * sqrt(2/nu) * sqrt(term_sqrt_inner)
        return -2 * rho_bulk * debye_length * np.sqrt(2/nu) * np.sqrt(term_sqrt_inner)

def get_potential_profile(params: SimulationParameters, potential_mv: float) -> Callable[[float], float]:
    """
    Returns a function that calculates the electric potential at distance x.
    
    Args:
        params: Simulation parameters.
        potential_mv: Electrode potential in mV.
        
    Returns:
        Function Phi(x) returning potential in V.
    """
    H = get_steric_layer_thickness(params, potential_mv)
    sigma = get_electrode_charge_density(params, potential_mv)
    phi_s = potential_mv / 1000.0
    
    if potential_mv < 0:
        counter_ion = params.cation
    else:
        counter_ion = params.anion
        
    c_cap = get_ion_steric_concentration(counter_ion.radius_ang, params.maximum_packing_fraction)
    rho_cap = counter_ion.charge * sc.e * c_cap * 1000 * sc.N_A
    epsilon = params.epsilon_solvent
    
    kappa = 1.0 / get_debye_length(params)
    vt = sc.k * params.temperature / sc.e
    z = abs(counter_ion.charge)
    
    if H > 0:
        phi_h = get_threshold_potential(counter_ion, params) / 1000.0
    else:
        phi_h = phi_s
        
    # Gamma for GC tail
    gamma = np.tanh( (z * abs(phi_h)) / (4 * vt) )
    
    def profile(x: float) -> float:
        if H > 0 and x <= H:
            # Eq. 14: Phi(x) = Phi_s - rho_cap/(2*epsilon)*x^2 - sigma/epsilon*x
            return phi_s - (rho_cap / (2 * epsilon)) * x**2 - (sigma / epsilon) * x
        else:
            # GC profile starting at x=H
            x_diff = x - H
            val = gamma * np.exp(-kappa * x_diff)
            phi_mag = (4 * vt / z) * np.arctanh(val)
            return np.sign(phi_h) * phi_mag
            
    return profile

def get_concentration_profile(params: SimulationParameters, potential_mv: float) -> Callable[[float], float]:
    """
    Returns a function that calculates the counter-ion concentration at distance x.
    
    Args:
        params: Simulation parameters.
        potential_mv: Electrode potential in mV.
        
    Returns:
        Function c(x) returning concentration in M (mol/L).
    """
    H = get_steric_layer_thickness(params, potential_mv)
    
    if potential_mv < 0:
        counter_ion = params.cation
    else:
        counter_ion = params.anion
        
    c_cap = get_ion_steric_concentration(counter_ion.radius_ang, params.maximum_packing_fraction)
    c_bulk = params.bulk_concentration
    
    phi_profile = get_potential_profile(params, potential_mv)
    vt = sc.k * params.temperature / sc.e
    
    def profile(x: float) -> float:
        if H > 0 and x <= H:
            return c_cap
        else:
            phi = phi_profile(x)
            return c_bulk * np.exp( -counter_ion.charge * phi / vt )
            
    return profile

def get_electric_field_profile(params: SimulationParameters, potential_mv: float) -> Callable[[float], float]:
    """
    Returns a function that calculates the electric field at distance x.
    
    Args:
        params: Simulation parameters.
        potential_mv: Electrode potential in mV.
        
    Returns:
        Function E(x) returning electric field in V/m.
    """
    H = get_steric_layer_thickness(params, potential_mv)
    sigma = get_electrode_charge_density(params, potential_mv)
    
    if potential_mv < 0:
        counter_ion = params.cation
    else:
        counter_ion = params.anion
        
    c_cap = get_ion_steric_concentration(counter_ion.radius_ang, params.maximum_packing_fraction)
    rho_cap = counter_ion.charge * sc.e * c_cap * 1000 * sc.N_A
    epsilon = params.epsilon_solvent
    
    phi_profile = get_potential_profile(params, potential_mv)
    kappa = 1.0 / get_debye_length(params)
    vt = sc.k * params.temperature / sc.e
    z = abs(counter_ion.charge)
    
    def profile(x: float) -> float:
        if H > 0 and x <= H:
            # Eq. 13: E(x) = sigma/epsilon + rho_cap/epsilon * x
            return sigma / epsilon + (rho_cap / epsilon) * x
        else:
            phi = phi_profile(x)
            # E = 2 k T kappa / (z e) * sinh( z e Phi / 2 k T )
            return (2 * vt * kappa / z) * np.sinh( (z * phi) / (2 * vt) )
            
    return profile

def get_free_energy_entropic(params: SimulationParameters, potential_mv: float) -> float:
    """
    Calculates the entropic free energy component.
    
    Args:
        params: Simulation parameters.
        potential_mv: Electrode potential in mV.
        
    Returns:
        Free energy in J/m^2.
    """
    H = get_steric_layer_thickness(params, potential_mv)
    
    if H > 0:
        # Analytical CDL (Eq. TFen)
        if potential_mv < 0:
            counter_ion = params.cation
        else:
            counter_ion = params.anion
            
        c_cap = get_ion_steric_concentration(counter_ion.radius_ang, params.maximum_packing_fraction)
        c_bulk = params.bulk_concentration
        n_cap = c_cap * 1000 * sc.N_A
        
        return sc.k * params.temperature * n_cap * np.log(c_cap / c_bulk) * H
    else:
        # Numerical GC
        phi_profile = get_potential_profile(params, potential_mv)
        vt = sc.k * params.temperature / sc.e
        n_bulk = params.bulk_concentration * 1000 * sc.N_A
        
        def integrand(x):
            phi = phi_profile(x)
            val = 0
            for ion in params.electrolyte_ions:
                # c = c_bulk * exp(-z e phi / kT)
                n = n_bulk * np.exp(-ion.charge * phi / vt)
                # term: n * ln(n/n_bulk)
                # ln(n/n_bulk) = -z e phi / kT
                # So term = n * (-z e phi / kT)
                val += n * (-ion.charge * phi / vt)
            return val * sc.k * params.temperature

        lambda_d = get_debye_length(params)
        limit = 20 * lambda_d
        result, error = quad(integrand, 0, limit)
        return result

def get_free_energy_electrostatic(params: SimulationParameters, potential_mv: float) -> float:
    """
    Calculates the electrostatic free energy component.
    
    Args:
        params: Simulation parameters.
        potential_mv: Electrode potential in mV.
        
    Returns:
        Free energy in J/m^2.
    """
    H = get_steric_layer_thickness(params, potential_mv)
    
    if H > 0:
        # Analytical CDL (Eq. TFel)
        sigma = get_electrode_charge_density(params, potential_mv)
        if potential_mv < 0:
            counter_ion = params.cation
        else:
            counter_ion = params.anion
            
        c_cap = get_ion_steric_concentration(counter_ion.radius_ang, params.maximum_packing_fraction)
        rho_cap = counter_ion.charge * sc.e * c_cap * 1000 * sc.N_A
        epsilon = params.epsilon_solvent
        
        term1 = (rho_cap**2 / 3) * H**3
        term2 = sigma * rho_cap * H**2
        term3 = sigma**2 * H
        
        return (1 / (2 * epsilon)) * (term1 + term2 + term3)
    else:
        # Numerical GC
        e_profile = get_electric_field_profile(params, potential_mv)
        epsilon = params.epsilon_solvent
        
        def integrand(x):
            E = e_profile(x)
            return 0.5 * epsilon * E**2
            
        lambda_d = get_debye_length(params)
        limit = 20 * lambda_d
        result, error = quad(integrand, 0, limit)
        return result

def get_free_energy_steric(params: SimulationParameters, potential_mv: float) -> float:
    """
    Calculates the steric free energy component.
    
    Args:
        params: Simulation parameters.
        potential_mv: Electrode potential in mV.
        
    Returns:
        Free energy in J/m^2.
    """
    H = get_steric_layer_thickness(params, potential_mv)
    
    if H > 0:
        # Analytical CDL (Eq. TFst)
        sigma = get_electrode_charge_density(params, potential_mv)
        phi_s = potential_mv / 1000.0
        
        if potential_mv < 0:
            counter_ion = params.cation
        else:
            counter_ion = params.anion
            
        c_cap = get_ion_steric_concentration(counter_ion.radius_ang, params.maximum_packing_fraction)
        rho_cap = counter_ion.charge * sc.e * c_cap * 1000 * sc.N_A
        epsilon = params.epsilon_solvent
        
        n_cap = c_cap * 1000 * sc.N_A
        mu_cap = -sc.k * params.temperature * np.log(c_cap / params.bulk_concentration)
        
        term_inner = phi_s - (sigma * H) / (2 * epsilon) - (rho_cap * H**2) / (6 * epsilon)
        
        return (mu_cap * n_cap - rho_cap * term_inner) * H
    else:
        # GC regime: Steric energy is zero
        return 0.0

def get_differential_capacitance(params: SimulationParameters, potential_mv: float, delta_mv: float = 1e-3) -> float:
    """
    Calculates the differential capacitance (C_diff = d_sigma / d_phi) analytically.
    
    Args:
        params: Simulation parameters.
        potential_mv: Electrode potential in mV.
        delta_mv: Unused, kept for compatibility.
        
    Returns:
        Differential capacitance in F/m^2.
    """
    if potential_mv < 0:
        counter_ion = params.cation
    else:
        counter_ion = params.anion
        
    threshold_potential_mv = get_threshold_potential(counter_ion, params)
    
    epsilon = params.epsilon_solvent
    lambda_d = get_debye_length(params)
    kt = sc.k * params.temperature
    z = abs(counter_ion.charge)
    e = sc.e
    
    if abs(potential_mv) < abs(threshold_potential_mv):
        # GC Regime
        # C = (epsilon / lambda_D) * cosh(z * e * Phi / (2 * kT))
        arg = z * e * (potential_mv / 1000.0) / (2 * kt)
        return (epsilon / lambda_d) * np.cosh(arg)
    else:
        # Steric Regime
        nu = 2 * params.bulk_concentration / get_ion_steric_concentration(counter_ion.radius_ang, params.maximum_packing_fraction)
        
        # term_potential = -z * e * Phi / kT
        # Note: In the formula provided, the term under sqrt is:
        # (1 - nu/2)^2 + z*e*|Phi|/kT - ln(2/nu)
        # Wait, the user provided: + z*e*|Phi|/kT
        # But in H calculation it was: - z*e*Phi/kT (which is positive)
        # Let's use the term_sqrt_inner from H calculation which we know is positive and correct for the physics.
        
        term_potential = -counter_ion.charge * sc.e * (potential_mv / 1000.0) / kt
        
        term_sqrt_inner = (
            (1 - 0.5 * nu)**2 + 
            term_potential - 
            np.log(2/nu)
        )
        
        if term_sqrt_inner <= 0:
             term_sqrt_inner = 1e-10 # Avoid division by zero
             
        # C = (1/sqrt(2*nu)) * (epsilon/lambda_D) / sqrt(term_sqrt_inner)
        return (1.0 / np.sqrt(2 * nu)) * (epsilon / lambda_d) / np.sqrt(term_sqrt_inner)
