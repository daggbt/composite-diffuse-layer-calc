import scipy.constants as sc
import numpy as np
from .models import Ion, Solvent
from .utils import get_hard_sphere_radius_from_gaussian

# Constant used in original code
# 1.6487773e-41 seems to be a conversion factor used in the original code.
# It is likely related to atomic units or specific polarizability units.
POLARIZABILITY_FACTOR = 1.6487773e-41
POLARIZABILITY_ANGSTROM3 = 4 * np.pi * sc.epsilon_0 * 1e-30

def create_ion(name: str, charge: int, gaussian_radius_ang: float, polarizability: float = 0.0, radius_ang: float = None) -> Ion:
    """Helper to create an Ion, automatically calculating hard sphere radius if not provided."""
    if radius_ang is None:
        radius_ang = get_hard_sphere_radius_from_gaussian(gaussian_radius_ang)
    return Ion(
        name=name,
        charge=charge,
        gaussian_radius_ang=gaussian_radius_ang,
        radius_ang=radius_ang,
        polarizability=polarizability
    )

IONS = {
    "Na": create_ion(
        name="Na",
        charge=1,
        gaussian_radius_ang=0.607246,
        polarizability=1.0015 * POLARIZABILITY_FACTOR
    ),
    "hydNaw3": create_ion(
        name="hydNaw3",
        charge=1,
        gaussian_radius_ang=2.24981,
        polarizability=0.0
    ),
    "hydronium": create_ion(
        name="hydronium",
        charge=1,
        gaussian_radius_ang=0.973925,
        polarizability=0.0
    ),
    "OH": create_ion(
        name="OH",
        charge=-1,
        gaussian_radius_ang=1.25953,
        polarizability=0.0
    ),
    "hydOHw3": create_ion(
        name="hydOHw3",
        charge=-1,
        gaussian_radius_ang=2.39243,
        polarizability=0.0
    ),
    "Cl": create_ion(
        name="Cl",
        charge=-1,
        gaussian_radius_ang=1.86058,
        polarizability=3.45 * POLARIZABILITY_ANGSTROM3
    ),
    "hydLi5Ion": create_ion(
        name="hydLi5Ion",
        charge=1,
        gaussian_radius_ang=2.56195,
        polarizability=0.193 * POLARIZABILITY_FACTOR
    ),
    "hydNa3Ion": create_ion(
        name="hydNa3Ion",
        charge=1,
        gaussian_radius_ang=2.24981,
        polarizability=1.0015 * POLARIZABILITY_FACTOR
    ),
    "hydOH3Ion": create_ion(
        name="hydOH3Ion",
        charge=-1,
        gaussian_radius_ang=2.39243,
        polarizability=0.0
    ),
    "hydOHIon": create_ion(
        name="hydOHIon",
        charge=-1,
        gaussian_radius_ang=0.0,
        radius_ang=3.0,
        polarizability=0.0
    ),
    "Li": create_ion(
        name="Li",
        charge=1,
        gaussian_radius_ang=0.38467,
        polarizability=0.193 * POLARIZABILITY_FACTOR
    ),
    "K": create_ion(
        name="K",
        charge=1,
        gaussian_radius_ang=3.010343,
        polarizability=5.47 * POLARIZABILITY_FACTOR
    ),
    "PF_6": create_ion(
        name="PF_6",
        charge=-1,
        gaussian_radius_ang=2.31,
        polarizability=4.18 * POLARIZABILITY_ANGSTROM3
    ),
    "BF_4": create_ion(
        name="BF_4",
        charge=-1,
        gaussian_radius_ang=2.09,
        polarizability=2.8 * POLARIZABILITY_ANGSTROM3
    ),
    "ClO_4": create_ion(
        name="ClO_4",
        charge=-1,
        gaussian_radius_ang=2.17,
        polarizability=6.02 * POLARIZABILITY_ANGSTROM3
    ),
    "BrO_4": create_ion(
        name="BrO_4",
        charge=-1,
        gaussian_radius_ang=2.27,
        polarizability=0.0
    ),
    "IO_4": create_ion(
        name="IO_4",
        charge=-1,
        gaussian_radius_ang=2.36,
        polarizability=0.0
    ),
    "TFSI": create_ion(
        name="TFSI",
        charge=-1,
        gaussian_radius_ang=3.42,
        polarizability=13.59 * POLARIZABILITY_ANGSTROM3
    ),
    "EM": create_ion(
        name="EM",
        charge=1,
        gaussian_radius_ang=3.47,
        polarizability=0.0
    ),
}

SOLVENTS = {
    "Propylene_Carbonate": Solvent(
        name="Propylene_Carbonate",
        radius_ang=2.75,
        dielectric=66.14
    ),
    "Water": Solvent(
        name="Water",
        radius_ang=1.715,
        dielectric=78.0
    ),
}
