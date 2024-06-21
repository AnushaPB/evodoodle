import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import geonomics as gnx
from functions import make_landscapes, edit_landscapes, init_mod, draw_landscape, plot_popgen
from gnx_params import params
import geonomics as gnx

# Draw landscapes
population_size = draw_landscape(d = 10)
connectivity = draw_landscape(d = 10)
environment = draw_landscape(d = 10)

# Start model
mod = init_mod(params, population_size, connectivity, environment)

# Run the model for 200 steps
mod.walk(200)

# Plot results
plot_popgen(mod)