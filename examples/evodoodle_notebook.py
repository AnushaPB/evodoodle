
"""
Evodoodle Notebook
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import geonomics as gnx
import evodoodle as evo

# 1. DRAW YOUR LANDSCAPE 

# %%
# Draw your landscape carrying capacity
# When you run this chunk a pop-up will appear where you can draw on your own landscape
# Note: d sets the dimensions of your square landscape in terms of the number of cells
carrying_capacity = evo.draw_landscape(d = 10)

# %% 
# Draw your landscape connectivity
connectivity = evo.draw_landscape(d = 10)

# %% 
# Draw your environment for local adaptation
environment = evo.draw_landscape(d = 10)

# %% 
# You can plot all your landscapes using plot_landscapes()
evo.plot_landscapes(carrying_capacity, connectivity, environment)

# %%
# If you want to change something, you can edit your landscapes using the edit_landscape() function
environment = evo.edit_landscape(environment)

# %% 
# Finally, you can optionally save your landscapes
np.savetxt('carrying_capacity.csv', carrying_capacity, delimiter=',')
np.savetxt('connectivity.csv', connectivity, delimiter=',')
np.savetxt('environment.csv', environment, delimiter=',')

# Uncomment these lines to read in landscapes
#carrying_capacity = np.loadtxt('carrying_capacity.csv', delimiter=',')
#connectivity = np.loadtxt('connectivity.csv', delimiter=',')
#environment = np.loadtxt('environment.csv', delimiter=',')

# 2. PLAY OUT EVOLUTION ACROSS YOUR LANSCAPE

# %%
# To define the parameters of our model for Geonomics we need to have a parameters dictionary
# Evodoodle comes with an example parameters dictionary to start with:
params = evo.example_params()

# %%
# You can also create your own Geonomics parameters file by running gnx.make_parameters_file()
# Here we call this file "example_parameters" and it will automatically be saved to our working directory
gnx.make_parameters_file("example_parameters")
# You can manually edit this file to change the simulation parameters, just note that the landscape layers will be overwritten when we create our custom landscapes, so best to leave that section, and any other section related to the landscape, alone

# You can then load your parameters python dictionary
from example_parameters import params
print(params)

# But for now we will use the example parameters
params = evo.example_params()

# %%
# Start model
mod = evo.init_mod(params, carrying_capacity, connectivity, environment)

# %%
# Plot initial model (no selection has occurred yet)
evo.plot_popgen(mod)

# %% 
# Run the model for 100 timesteps
mod.walk(100)

# %%
# Plot the results
evo.plot_popgen(mod)

# %%
# Run the model for another 100 steps
mod.walk(100)

# %%
evo.plot_popgen(mod)