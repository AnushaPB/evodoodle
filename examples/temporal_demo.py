
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
# Run the model for 200 steps and collect statistics every 10 time steps
stats = evo.stats_walk(mod, t=200, inc=10)

# %%
evo.plot_stats(stats)

# %%
# Run another model for comparison with a different population size
params['comm']['species']['spp_0']['init']['K_factor'] = 2
mod2 = evo.init_mod(params, population_size, connectivity, environment)
stats2 = evo.stats_walk(mod2, t=200, inc=10)

#%%
# Create a dictionary with your two models
stats = {
    'Model 1': stats,
    'Model 2': stats2
}

# %%
# Plot the statistics for both models
plot_multistats(stats)