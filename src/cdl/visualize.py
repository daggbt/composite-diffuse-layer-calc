import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
from .models import SimulationParameters
from .database import IONS, SOLVENTS
from .solvers import get_charge_balance_potential, check_separation_validity
from .physics import (
    get_debye_length,
    get_potential_profile,
    get_electric_field_profile,
    get_concentration_profile,
    get_steric_layer_thickness,
    get_differential_capacitance,
    get_free_energy_entropic,
    get_free_energy_electrostatic,
    get_free_energy_steric,
    get_electrode_charge_density
)

def get_co_ion_profile(params: SimulationParameters, potential_mv: float, phi_profile_func, H: float):
    """
    Calculates the co-ion concentration profile.
    Assumes co-ion concentration is 0 in the steric layer (x < H).
    """
    if potential_mv < 0:
        co_ion = params.anion
    else:
        co_ion = params.cation
        
    c_bulk = params.bulk_concentration
    vt = sc.k * params.temperature / sc.e
    
    def profile(x: float) -> float:
        if H > 0 and x <= H:
            return 0.0
        else:
            phi = phi_profile_func(x)
            return c_bulk * np.exp(-co_ion.charge * phi / vt)
            
    return profile

def plot_spatial_profiles(params: SimulationParameters, phi_s_mv: float, filename: str = "spatial_profiles.png"):
    """
    Plots Phi(x), E(x), c(x), rho(x).
    Handles both single electrode (half-cell) and double electrode (full cell) cases.
    """
    print(f"Generating spatial profiles for Potential = {phi_s_mv} mV...")
    
    if params.single_electrode:
        # --- Single Electrode Logic ---
        H = get_steric_layer_thickness(params, phi_s_mv)
        phi_func = get_potential_profile(params, phi_s_mv)
        e_field_func = get_electric_field_profile(params, phi_s_mv)
        counter_ion_func = get_concentration_profile(params, phi_s_mv)
        co_ion_func = get_co_ion_profile(params, phi_s_mv, phi_func, H)
        
        # Determine x range
        x_max = max(2 * H, 5e-9) if H > 0 else 5e-9
        x_vals = np.linspace(0, x_max, 1000)
        
        phi_vals = np.array([phi_func(x) for x in x_vals])
        e_vals = np.array([e_field_func(x) for x in x_vals])
        c_counter_vals = np.array([counter_ion_func(x) for x in x_vals])
        c_co_vals = np.array([co_ion_func(x) for x in x_vals])
        
        if phi_s_mv < 0:
            c_cation_vals = c_counter_vals
            c_anion_vals = c_co_vals
        else:
            c_cation_vals = c_co_vals
            c_anion_vals = c_counter_vals
            
        rho_vals = (c_cation_vals * params.cation.charge + c_anion_vals * params.anion.charge) * 1000 * sc.N_A * sc.e
        
        title_text = rf"Single Electrode Profiles at $\Phi_s = {phi_s_mv/1000:.3f}$ V"
        
    else:
        # --- Double Electrode Logic ---
        # phi_s_mv is the total potential difference
        check_separation_validity(params, phi_s_mv)
        
        phi_left = get_charge_balance_potential(params, phi_s_mv)
        phi_right = phi_left - phi_s_mv
        
        L_debye = get_debye_length(params)
        L = params.separation_distance * L_debye
        
        # Left profiles (at x=0)
        H_left = get_steric_layer_thickness(params, phi_left)
        phi_func_left = get_potential_profile(params, phi_left)
        e_func_left = get_electric_field_profile(params, phi_left)
        
        # Right profiles (at x=L, extending leftwards)
        H_right = get_steric_layer_thickness(params, phi_right)
        phi_func_right = get_potential_profile(params, phi_right)
        e_func_right = get_electric_field_profile(params, phi_right)
        
        # Helper to get specific ion profiles
        def get_ion_funcs(p_phi, p_H):
            p_phi_func = get_potential_profile(params, p_phi)
            p_counter = get_concentration_profile(params, p_phi)
            p_co = get_co_ion_profile(params, p_phi, p_phi_func, p_H)
            if p_phi < 0:
                return p_counter, p_co # cation, anion
            else:
                return p_co, p_counter # cation, anion
                
        cat_func_left, an_func_left = get_ion_funcs(phi_left, H_left)
        cat_func_right, an_func_right = get_ion_funcs(phi_right, H_right)
        
        x_vals = np.linspace(0, L, 1000)
        
        # Superposition
        # Phi(x) = Phi_L(x) + Phi_R(L-x)
        phi_vals = np.array([phi_func_left(x) + phi_func_right(L-x) for x in x_vals])
        
        # E(x) = E_L(x) - E_R(L-x)
        e_vals = np.array([e_func_left(x) - e_func_right(L-x) for x in x_vals])
        
        c_bulk = params.bulk_concentration
        
        # c(x) = c_bulk + delta_c_L(x) + delta_c_R(L-x)
        c_cation_vals = np.array([
            c_bulk + (cat_func_left(x) - c_bulk) + (cat_func_right(L-x) - c_bulk)
            for x in x_vals
        ])
        c_anion_vals = np.array([
            c_bulk + (an_func_left(x) - c_bulk) + (an_func_right(L-x) - c_bulk)
            for x in x_vals
        ])
        
        rho_vals = (c_cation_vals * params.cation.charge + c_anion_vals * params.anion.charge) * 1000 * sc.N_A * sc.e
        
        title_text = rf"Double Electrode Profiles ($\Delta V = {phi_s_mv/1000:.3f}$ V)" + "\n" + rf"($\Phi_L \approx {phi_left/1000:.3f}$ V, $\Phi_R \approx {phi_right/1000:.3f}$ V)"

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(10.2, 8.5))
    fig.suptitle(title_text, fontsize=14)
    
    # 1. Potential Phi(x)
    axs[0, 0].plot(x_vals * 1e9, phi_vals, 'b-', linewidth=2)
    axs[0, 0].set_title(r"Electric Potential $\Phi(x)$")
    axs[0, 0].set_xlabel("Distance (nm)")
    axs[0, 0].set_ylabel("Potential (V)")
    axs[0, 0].grid(True, alpha=0.3)
    if params.single_electrode and 'H' in locals() and H > 0: 
        axs[0, 0].axvline(H*1e9, color='r', linestyle='--', alpha=0.5, label='Steric Layer')
    
    # 2. Electric Field E(x)
    axs[0, 1].plot(x_vals * 1e9, e_vals, 'g-', linewidth=2)
    axs[0, 1].set_title(r"Electric Field $E(x)$")
    axs[0, 1].set_xlabel("Distance (nm)")
    axs[0, 1].set_ylabel("Electric Field (V/m)")
    axs[0, 1].grid(True, alpha=0.3)

    # 3. Concentration c(x)
    axs[1, 0].plot(x_vals * 1e9, c_cation_vals, 'r-', label=params.cation.name)
    axs[1, 0].plot(x_vals * 1e9, c_anion_vals, 'b--', label=params.anion.name)
    axs[1, 0].set_title(r"Ion Concentration $c(x)$")
    axs[1, 0].set_xlabel("Distance (nm)")
    axs[1, 0].set_ylabel("Concentration (M)")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    
    # 4. Charge Density rho(x)
    axs[1, 1].plot(x_vals * 1e9, rho_vals, 'k-', linewidth=2)
    axs[1, 1].set_title(r"Charge Density $\rho(x)$")
    axs[1, 1].set_xlabel("Distance (nm)")
    axs[1, 1].set_ylabel(r"Charge Density ($C/m^3$)")
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")
    plt.close()

def plot_voltage_dependence(params: SimulationParameters, filename: str = "voltage_dependence.png", v_min: float = -1000.0, v_max: float = 1000.0, steps: int = 501):
    """
    Plots C_diff, H, and Free Energies vs Phi_s.
    """
    print(f"Generating voltage dependence plots ({v_min} to {v_max} mV)...")
    
    # Voltage range
    voltages_mv = np.linspace(v_min, v_max, steps)
    
    C_diff = []
    H_vals = []
    F_en = []
    F_el = []
    F_st = []
    F_total = []
    
    for v in voltages_mv:
        C_diff.append(get_differential_capacitance(params, v) * 100) # Convert to uF/cm^2 usually? Or F/m^2?
        # 1 F/m^2 = 100 uF/cm^2
        
        H_vals.append(get_steric_layer_thickness(params, v) * 1e9) # nm
        
        fen = get_free_energy_entropic(params, v)
        fel = get_free_energy_electrostatic(params, v)
        fst = get_free_energy_steric(params, v)
        
        F_en.append(fen * 1000) # mJ/m^2
        F_el.append(fel * 1000)
        F_st.append(fst * 1000)
        F_total.append((fen + fel + fst) * 1000)
        
    fig, axs = plt.subplots(2, 2, figsize=(10.2, 8.5))
    fig.suptitle("Voltage Dependent Properties", fontsize=16)
    
    # 1. Differential Capacitance
    axs[0, 0].plot(voltages_mv / 1000, C_diff, 'b-', linewidth=2)
    axs[0, 0].set_title("Differential Capacitance")
    axs[0, 0].set_xlabel("Potential (V)")
    axs[0, 0].set_ylabel(r"Capacitance ($\mu F/cm^2$)") # Assuming conversion factor 100
    axs[0, 0].grid(True, alpha=0.3)
    
    # 2. Steric Layer Thickness
    axs[0, 1].plot(voltages_mv / 1000, H_vals, 'r-', linewidth=2)
    axs[0, 1].set_title("Steric Layer Thickness $H$")
    axs[0, 1].set_xlabel("Potential (V)")
    axs[0, 1].set_ylabel("Thickness (nm)")
    axs[0, 1].grid(True, alpha=0.3)
    
    # 3. Free Energy Components
    axs[1, 0].plot(voltages_mv / 1000, F_en, label="$F_{en}$")
    axs[1, 0].plot(voltages_mv / 1000, F_el, label="$F_{el}$")
    axs[1, 0].plot(voltages_mv / 1000, F_st, label="$F_{st}$")
    axs[1, 0].set_title("Free Energy Components")
    axs[1, 0].set_xlabel("Potential (V)")
    axs[1, 0].set_ylabel("Energy ($mJ/m^2$)")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    
    # 4. Total Free Energy
    axs[1, 1].plot(voltages_mv / 1000, F_total, 'k-', linewidth=2, label="$F_{total}$")
    
    # Compare with 1/2 C V^2 using integral capacitance? 
    # Or just 1/2 * C_diff(0) * V^2 as reference?
    # Let's just plot total free energy for now.
    
    axs[1, 1].set_title("Total Free Energy")
    axs[1, 1].set_xlabel("Potential (V)")
    axs[1, 1].set_ylabel("Energy ($mJ/m^2$)")
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")
    plt.close()

def main():
    # Setup parameters
    cation = IONS["Li"]
    anion = IONS["PF_6"]
    solvent = SOLVENTS["Propylene_Carbonate"]
    
    params = SimulationParameters(
        electrolyte_ions=(cation, anion),
        solvent=solvent,
        bulk_concentration=1.0, # 1 M
        temperature=298.15
    )
    
    # 1. Plot Spatial Profiles at 1V
    plot_spatial_profiles(params, 1000.0, "spatial_profiles_1V.png")
    
    # 2. Plot Voltage Dependence
    plot_voltage_dependence(params, "voltage_dependence.png")

if __name__ == "__main__":
    main()
