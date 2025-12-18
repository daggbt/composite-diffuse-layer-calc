from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import scipy.constants as sc
import numpy as np

@dataclass(frozen=True)
class Ion:
    """Represents an ion species in the electrolyte."""
    name: str
    charge: int
    radius_ang: float  # Hard sphere radius in Angstroms
    gaussian_radius_ang: float = 0.0
    polarizability: float = 0.0

@dataclass(frozen=True)
class Solvent:
    """Represents the solvent medium."""
    name: str
    radius_ang: float
    dielectric: float
    reduced_dielectric: float = 0.0

@dataclass
class SimulationParameters:
    """Holds the configuration for a simulation run."""
    electrolyte_ions: Tuple[Ion, Ion]  # (Cation, Anion) usually, or (Counter, Co) depending on context
    solvent: Solvent
    bulk_concentration: float = 1.0  # Molar (mol/L)
    temperature: float = 298.15  # Kelvin
    
    # Packing fraction for steric effects
    # Default: sc.pi*np.sqrt(3)/8 for bcc cell
    maximum_packing_fraction: float = np.pi * np.sqrt(3) / 8
    
    single_electrode: bool = False
    separation_distance: float = 100.0 # Multiples of Debye length
    potential_difference: float = 0.0 # mV

    @property
    def cation(self) -> Ion:
        # Assuming the first one is cation if positive, but let's just return index 0
        # The original code did: cath, an = prop.electrolyteIons
        return self.electrolyte_ions[0]

    @property
    def anion(self) -> Ion:
        return self.electrolyte_ions[1]

    @property
    def epsilon_solvent(self) -> float:
        return sc.epsilon_0 * self.solvent.dielectric
