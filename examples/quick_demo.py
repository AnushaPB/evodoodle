# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import geonomics as gnx
import evodoodle as evo
import geonomics as gnx

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
# Load example parameters dictionary
params = evo.example_params()

# %%
# Start the model
mod = evo.init_mod(params, population_size, connectivity, environment)

# %%
# Run the model for 200 steps
mod.walk(200)

# %%
# Plot the results
evo.plot_popgen(mod)

# %%
