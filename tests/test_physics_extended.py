import pytest
import numpy as np
import scipy.constants as sc
from cdl.models import Ion, Solvent, SimulationParameters
from cdl.physics import (
    get_debye_length,
    get_ion_steric_concentration,
    get_threshold_potential,
    get_steric_layer_thickness,
    get_electrode_charge_density,
    get_concentration_profile,
    get_electric_field_profile,
    get_potential_profile,
    get_free_energy_entropic,
    get_free_energy_electrostatic,
    get_free_energy_steric,
    get_differential_capacitance
)

@pytest.fixture
def params():
    cation = Ion(name="Cat", charge=1, radius_ang=2.0)
    anion = Ion(name="An", charge=-1, radius_ang=2.0)
    solvent = Solvent(name="Sol", radius_ang=2.0, dielectric=10.0)
    return SimulationParameters(
        electrolyte_ions=(cation, anion),
        solvent=solvent,
        bulk_concentration=1.0, # 1 M
        temperature=298.15
    )

def test_threshold_potential(params):
    # Check that threshold potential is calculated
    v_th = get_threshold_potential(params.anion, params)
    assert isinstance(v_th, float)
    # Should be positive for anion (since z=-1, log(c_st/c_bulk) > 0 usually)
    # Wait, steric_conc is usually > bulk_conc.
    # log(c_st/c_bulk) > 0.
    # val = -kT * log / e / z.
    # If z = -1: val = -kT * (+) / e / (-) = (+).
    # So threshold for anion (positive potential side) should be positive.
    assert v_th > 0

def test_steric_layer_thickness(params):
    v_th = get_threshold_potential(params.anion, params)
    
    # Below threshold, H should be 0
    assert get_steric_layer_thickness(params, v_th * 0.5) == 0.0
    
    # Above threshold, H should be > 0
    assert get_steric_layer_thickness(params, v_th * 1.5) > 0.0

def test_charge_density_gouy_chapman(params):
    v_th = get_threshold_potential(params.anion, params)
    
    # Below threshold, should match Gouy-Chapman
    v_test = v_th * 0.5
    sigma = get_electrode_charge_density(params, v_test)
    
    # GC formula check
    n0 = params.bulk_concentration * 1000 * sc.N_A
    prefactor = np.sqrt(8 * params.epsilon_solvent * sc.k * params.temperature * n0)
    arg = abs(params.anion.charge) * sc.e * abs(v_test / 1000.0) / (2 * sc.k * params.temperature)
    expected = np.sign(v_test) * prefactor * np.sinh(arg)
    
    assert sigma == pytest.approx(expected)

def test_profiles(params):
    v_th = get_threshold_potential(params.anion, params)
    v_test = v_th * 2.0 # Ensure steric layer exists
    
    H = get_steric_layer_thickness(params, v_test)
    assert H > 0
    
    # Concentration profile
    c_profile = get_concentration_profile(params, v_test)
    c_cap = get_ion_steric_concentration(params.anion.radius_ang, params.maximum_packing_fraction)
    assert c_profile(0) == pytest.approx(c_cap)
    assert c_profile(H * 0.5) == pytest.approx(c_cap)
    
    # Just outside H, concentration should be continuous (c_cap) or decaying?
    # At x=H, Phi=Phi_c, so c = c_bulk * exp(-z e Phi_c / kT) = c_cap.
    # So it should be continuous.
    assert c_profile(H * 1.001) == pytest.approx(c_cap, rel=0.01)
    
    # Far away, it should approach bulk
    assert c_profile(H + 100e-9) == pytest.approx(params.bulk_concentration, rel=0.1)

    
    # Electric field profile
    e_profile = get_electric_field_profile(params, v_test)
    # E should be continuous? Or at least defined.
    assert isinstance(e_profile(0), float)
    
    # Potential profile
    p_profile = get_potential_profile(params, v_test)
    assert p_profile(0) == pytest.approx(v_test / 1000.0)
    
    # Check diffuse tail (x > H)
    # Potential should decay
    assert abs(p_profile(H * 2.0)) < abs(p_profile(H))

def test_profiles_gc_regime(params):
    v_th = get_threshold_potential(params.anion, params)
    v_test = v_th * 0.5 # GC regime
    
    H = get_steric_layer_thickness(params, v_test)
    assert H == 0.0
    
    p_profile = get_potential_profile(params, v_test)
    assert p_profile(0) == pytest.approx(v_test / 1000.0)
    
    # Check decay
    assert abs(p_profile(1e-9)) < abs(p_profile(0))
    
    # Check concentration profile
    c_profile = get_concentration_profile(params, v_test)
    # At x=0, c should be > bulk (for counter-ion)
    assert c_profile(0) > params.bulk_concentration
    # At large x, c should approach bulk
    assert c_profile(1e-6) == pytest.approx(params.bulk_concentration, rel=1e-2)


def test_free_energies(params):
    v_th = get_threshold_potential(params.anion, params)
    v_test = v_th * 2.0
    
    f_en = get_free_energy_entropic(params, v_test)
    f_el = get_free_energy_electrostatic(params, v_test)
    f_st = get_free_energy_steric(params, v_test)
    
    assert isinstance(f_en, float)
    assert isinstance(f_el, float)
    assert isinstance(f_st, float)

def test_differential_capacitance(params):
    v_th = get_threshold_potential(params.anion, params)
    
    # Capacitance should be positive
    c_diff = get_differential_capacitance(params, v_th * 0.5)
    assert c_diff > 0
    
    c_diff_high = get_differential_capacitance(params, v_th * 2.0)
    assert c_diff_high > 0
