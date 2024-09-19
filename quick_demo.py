# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import geonomics as gnx
from evodoodle import init_mod, draw_landscape, edit_landscape, plot_popgen, plot_landscapes
from gnx_params import params
import geonomics as gnx

# %%
# Draw landscapes
population_size = draw_landscape(d = 10)
connectivity = draw_landscape(d = 10)
environment = draw_landscape(d = 10)

# %%
# Plot the landscapes
plot_landscapes(population_size, connectivity, environment)

# %%
# Start the model
mod = init_mod(params, population_size, connectivity, environment)

# %%
# Run the model for 200 steps
mod.walk(200)

# %%
# Plot the results
plot_popgen(mod)