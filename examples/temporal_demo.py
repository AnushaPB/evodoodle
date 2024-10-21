
"""
Looking at evolution across space over time (BETA)
"""

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
# Initialize a statistics object to store results as you run the simulations
stats = evo.stats_handler()

# %%
# Run the model for 100 steps, collecting stats object every 10 steps
stats.walk(mod, t = 100, inc = 10)

# %%
# Plot the results
evo.plot_popgen(mod)
stats.plot_fitness()

# %%
# Run the model for another 100 steps and plot the results
stats.walk(mod, t=100, inc = 10)
evo.plot_popgen(mod)
stats.plot_fitness()

# %% 
# Now, let's run another simulation and plot the results
mod2 = evo.init_mod(params, population_size, connectivity, environment)
stats2 = evo.stats_handler()
stats2.walk(mod2, t = 200, inc = 10)
stats2.plot_fitness()
evo.plot_popgen(mod2)

# %%
# We can also plot the fitness of the two simulations together

# Get the stats from the two simulations
stats1 = stats.get_stats()
stats2 = stats2.get_stats()

# Combine the stats into a single dictionary
stats_combined = {
    'model 1': stats1,
    'model 2': stats2
}

# Plot the fitness of the two simulations together
evo.plot_fitness(stats_combined)