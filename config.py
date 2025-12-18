# Configuration parameters for CDL Calculator

# Ion and Solvent selection
# Must match keys in src/cdl/database.py
CATION = "hydLi5Ion"
ANION = "PF_6"
SOLVENT = "Propylene_Carbonate"

# Simulation parameters
BULK_CONCENTRATION = 1.0  # M (mol/L)
TEMPERATURE = 298.15      # K
SINGLE_ELECTRODE = False  # True for single electrode, False for two-electrode cell
SEPARATION_DISTANCE = 1.0 # Separation distance in multiples of Debye length (for two-electrode)

# Visualization parameters
POTENTIAL = 1000.0  # mV (for spatial profiles)
OUTPUT_FOLDER = "output" # Directory to save plots
