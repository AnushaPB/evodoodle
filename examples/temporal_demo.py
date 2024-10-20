
"""
Looking at evolution across space over time
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
# Run the model for 100 steps
stats = evo.stats_walk(mod, t=100, inc = 10)

# %%
# Plot the results
evo.plot_popgen(mod)
evo.plot_fitness(stats)

# %%
# Run the model for another 100 steps
stats2 = evo.stats_walk(mod, t=100, inc = 10)

# Initialize the combined dictionary
combined_stats = {}

# Iterate over the keys and concatenate the lists
for key in stats:
    combined_stats[key] = stats[key] + stats2[key]

print(combined_stats)

evo.plot_fitness(combined_stats)
