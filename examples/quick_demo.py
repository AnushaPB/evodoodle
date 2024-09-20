# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import geonomics as gnx
import evodoodle as evo
import geonomics as gnx

# Import example parameters
# %%
import sys
import os
sys.path.append(os.path.dirname(__file__))
from gnx_params import params

# %%
# Draw landscapes
population_size = evo.draw_landscape(d = 10)

# %%
connectivity = evo.draw_landscape(d = 10)

# %%
environment = evo.draw_landscape(d = 10)

# %%
# Plot the landscapes
evo.plot_landscapes(population_size, connectivity, environment)

# %%
# Start the model
mod = evo.init_mod(params, population_size, connectivity, environment)

# %%
# Run the model for 200 steps
mod.walk(200)

# %%
# Plot the results
evo.plot_popgen(mod)