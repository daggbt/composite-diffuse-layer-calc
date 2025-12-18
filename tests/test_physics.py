import pytest
import numpy as np
import scipy.constants as sc
from cdl.models import Ion, Solvent, SimulationParameters
from cdl.physics import get_debye_length, get_ion_steric_concentration

@pytest.fixture
def simulation_params():
    cation = Ion(name="Cat", charge=1, radius_ang=2.0)
    anion = Ion(name="An", charge=-1, radius_ang=2.0)
    solvent = Solvent(name="Sol", radius_ang=2.0, dielectric=10.0)
    return SimulationParameters(
        electrolyte_ions=(cation, anion),
        solvent=solvent,
        bulk_concentration=1.0,
        temperature=298.15
    )

def test_debye_length(simulation_params):
    # Calculate expected Debye length manually
    # lambda_D = sqrt( epsilon * k * T / (e^2 * sum(n_i * z_i^2)) )
    # n_i = 1.0 * 1000 * N_A
    # sum(n_i * z_i^2) = 2 * 1000 * N_A * 1^2
    
    epsilon = 10.0 * sc.epsilon_0
    kt = sc.k * 298.15
    denominator = sc.e**2 * 2 * 1000 * sc.N_A
    expected = np.sqrt(epsilon * kt / denominator)
    
    calculated = get_debye_length(simulation_params)
    assert calculated == pytest.approx(expected, abs=1e-10)

def test_steric_concentration(simulation_params):
    # max_packing / volume / 1000 / N_A
    radius_m = 2.0 * 1e-10
    volume = 4/3 * np.pi * radius_m**3
    max_packing = simulation_params.maximum_packing_fraction
    
    expected = max_packing / volume / 1000 / sc.N_A
    calculated = get_ion_steric_concentration(2.0, max_packing)
    assert calculated == pytest.approx(expected, abs=1e-10)
