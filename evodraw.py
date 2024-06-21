import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import geonomics as gnx
from functions import make_landscapes, edit_landscapes, init_mod, draw_landscape, plot_popgen
from gnx_params import params
from geonomics import gnx

# Create a custom landscape by clicking on the cells
# population_size, connectivity, environment = make_landscapes(d = 9, binary = False)
# To edit the landscapes, uncomment the line below
# population_size, connectivity, environment = edit_landscapes(population_size2, connectivity, environment, binary = True)

# Draw landscapes
population_size = draw_landscape(d = 10)
connectivity = draw_landscape(d = 10)
environment = draw_landscape(d = 10)
# To edit the landscapes, uncomment the line below
# population_size, connectivity, environment = edit_landscapes(population_size, connectivity, environment, binary = False)

# Save your landscapes
#np.savetxt('population_size.csv', population_size, delimiter=',')
#np.savetxt('connectivity.csv', connectivity, delimiter=',')
#np.savetxt('environment.csv', environment, delimiter=',')
# Read in landscapes
#population_size = np.loadtxt('population_size.csv', delimiter=',')
#connectivity = np.loadtxt('connectivity.csv', delimiter=',')
#environment = np.loadtxt('environment.csv', delimiter=',')

# Start model
mod = init_mod(params, population_size, connectivity, environment)

# Plot initial model
plot_popgen(mod)

# Run the model for 100 steps
mod.walk(100)
plot_popgen(mod)

# Run the model for 100 steps
mod.walk(100)
plot_popgen(mod)
