"""
Barrier Simulations
------------------
A central question in landscape and conservation genetics is how landscape
features influence gene flow. As human development increasingly fragments
natural habitats, roads, fences, and other anthropogenic features are often
hypothesized to act as barriers that reduce movement and genetic connectivity
among populations. In this activity, we will use simulations to explore how a 
landscape barrier can shape patterns of genetic differentiation, using salamanders 
as our focal species to test how a road influences gene flow across space.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import geonomics as gnx
import evodoodle as evo

# This function plots individuals in genetic PCA space and colors them
# by which side of the landscape they occupy (left vs right).
# It is used to visually assess whether individuals on different sides
# of the landscape are genetically distinguishable.
def plot_half(mod, spp_idx=0, x_split=5, ax=None, title=None, alpha=0.8, s=18):
    from sklearn.decomposition import PCA

    spp = mod.comm[spp_idx]

    # Convert genotypes to a 2D matrix for PCA
    G = spp._get_genotypes()            # (n_indiv, loci, ploidy)
    n, L, ploidy = G.shape
    Xg = G.reshape(n, L * ploidy).astype(float)

    # Add tiny noise if genotypes are identical (avoids PCA errors)
    if np.allclose(Xg, Xg[0, :]):
        Xg += np.random.normal(scale=1e-8, size=Xg.shape)

    # Run PCA on genotypes
    pcs = PCA(n_components=2).fit_transform(Xg)

    # Extract x-coordinates to classify individuals by landscape side
    xs = []
    for ind in spp.values():
        if hasattr(ind, "x"):
            xs.append(ind.x)
        elif hasattr(ind, "coord"):
            xs.append(ind.coord[0])
        elif hasattr(ind, "coords"):
            xs.append(ind.coords[0])
        else:
            raise AttributeError("Could not find individual x-coordinates.")
    xs = np.asarray(xs)

    left = xs < x_split
    right = xs > x_split

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    # Plot individuals colored by landscape side
    ax.scatter(pcs[left, 0], pcs[left, 1],
               c="#273cc6", alpha=alpha, s=s, label="Left")
    ax.scatter(pcs[right, 0], pcs[right, 1],
               c="#3dd9dc", alpha=alpha, s=s, label="Right")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_box_aspect(1)
    ax.legend(frameon=False)

    if title is not None:
        ax.set_title(title)

    return ax

# ---------------------------------------------------------------------
# QUESTION 1: DEFINING A NULL HYPOTHESIS
# ---------------------------------------------------------------------
# Before testing whether a road causes genetic differentiation, we need
# to define an appropriate *null hypothesis*.
#
# Q1) What would a reasonable null hypothesis be in this system?
# Q2) What kinds of simulations would you run to evaluate that null?
#
# Take a moment to write down your ideas before continuing.
#
# ---------------------------------------------------------------------
# PART 1: BASELINE SCENARIO (NO BARRIER)
# ---------------------------------------------------------------------
# We will begin by simulating a landscape with no barrier.
#
# The species we simulate has limited dispersal, similar to a salamander.
# Dispersal limitation is controlled by a movement parameter in the model.
#
# Before running the simulation:
#
# Q3) In the absence of a barrier, what genetic pattern do you expect
#     across the landscape?
#
# Write down your prediction and reasoning.
#
# ---------------------------------------------------------------------
# Set up a uniform landscape (no barrier, no environmental differences)
uniform_landscape = np.ones((10, 10))
environment = uniform_landscape
connectivity = uniform_landscape
carrying_capacity = uniform_landscape

# Visualize the landscape
evo.plot_landscapes(environment, connectivity, carrying_capacity)

# ---------------------------------------------------------------------
# Load example parameters
params = evo.example_params()

# Set dispersal scale:
# movement_distance_distr_param_1 controls the mean distance individuals
# move each time step in cells. Smaller values represent more limited dispersal.
params["comm"]["species"]["spp_0"]["movement"]["movement_distance_distr_param_1"] = 0.5

# ---------------------------------------------------------------------
# Initialize and run the model (NO barrier)
mod_null = evo.init_mod(params, carrying_capacity, connectivity, environment)
mod_null.walk(100)

# After running the simulation:
#
# Q4) Describe the genetic pattern you observe.
#     How does it compare to your prediction?
plot_half(mod_null, x_split=5, title="No barrier scenario")
plt.show()

# ---------------------------------------------------------------------
# PART 2: ADDING A BARRIER
# ---------------------------------------------------------------------
# Now we will introduce a barrier across the landscape, representing a road
# that reduces habitat and/or movement.
#
# Before running the simulation:
#
# Q5) How do you expect genetic patterns to change relative to the
#     no-barrier scenario?
#
# Draw a barrier by setting cells to zero (uninhabitable) down the middle.
env_with_barrier = evo.draw_landscape(10)

# Initialize and run the model with the barrier
mod_barrier = evo.init_mod(params, env_with_barrier, env_with_barrier, environment)
mod_barrier.walk(100)

# After running the simulation:
#
# Q6) How does the genetic pattern differ from the no-barrier case?
plot_half(mod_barrier, x_split=5, title="Barrier scenario")
plt.show()

# ---------------------------------------------------------------------
# INTERPRETATION
# ---------------------------------------------------------------------
# Suppose a researcher collected salamander genetic data from either side of
# a road and found minor genetic differentiation.
#
# They conclude: “The road is acting as a barrier to gene flow.”
#
# Q7) Based on what you observed in your simulations, what do you think
#     about this conclusion?
# Q8) What additional comparisons or information would you want before
#     making a causal claim about the road?
#
# ---------------------------------------------------------------------
# EXTENSION: CHANGE DISPERSAL
# ---------------------------------------------------------------------
# Now imagine a species with higher dispersal ability (e.g., a deer).
#
# Increase the dispersal parameter and repeat BOTH simulations:
#   - No barrier
#   - With barrier
#
# Example:
# params["comm"]["species"]["spp_0"]["movement"]["movement_distance_distr_param_1"] = 2.0
#
# Q9) How does increased dispersal change the baseline genetic pattern?
# Q10) How does it affect your interpretation of barrier effects?
