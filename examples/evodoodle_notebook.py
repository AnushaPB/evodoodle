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
population_size = evo.draw_landscape(d = 10)

# %% 
# Draw your landscape connectivity
connectivity = evo.draw_landscape(d = 10)

# %% 
# Draw your environment for local adaptation
environment = evo.draw_landscape(d = 10)

# %% 
# You can plot all your landscapes using plot_landscapes()
evo.plot_landscapes(population_size, connectivity, environment)

# %%
# If you want to change something, you can edit your landscapes using the edit_landscape() function
environment = evo.edit_landscape(environment)

# %% 
# Finally, you can optionally save your landscapes
np.savetxt('population_size.csv', population_size, delimiter=',')
np.savetxt('connectivity.csv', connectivity, delimiter=',')
np.savetxt('environment.csv', environment, delimiter=',')

# Read in landscapes
#population_size = np.loadtxt('population_size.csv', delimiter=',')
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
# You can manually edit this file to change the simulation parameters, just note that the landscape layers will be overwritten when we create our custom landscapes, so best to leave that section of the parameters alone

# You can then load your parameters python dictionary
from example_parameters import params
print(params)

# But for now we will use the example parameters
params = evo.example_params()

# %%
# Start model
mod = evo.init_mod(params, population_size, connectivity, environment)

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

# %% [markdown]
"""
1. You are consulting with a conservation organization that is trying to decide whether to build (Option 1) a large, connected, single preserve or (Option 2) several, smaller, disconnected preserves. Design a simulation (or a set of simulations) to help them make their decision. Bonus challenge: try and simulate one scenario where option (1) is best and one scenario where (2) is best (Hint: think about different things you might want to conserve)

2. Wildlife corridors are a popular method for restoring connectivity between areas fragmented by human development, such as roads. Create a simulation (or a pair of simulations) that shows the effect of a wildlife corridor across a hypothetical road on genetic diversity. What happens if the corridor connects two very different environments?

3. Your friend comes to you super excited because they just analyzed their landscape genetics data on hedgehogs and found a super clear gradient in PC values that aligns with a gradient in temperature. They tell you that they think this is an indication that hedgehogs are locally adapted to temperature. What do you think? Can you come up with a simulation to show your friend what else could be causing this pattern?
"""