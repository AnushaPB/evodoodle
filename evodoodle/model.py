#!/usr/bin/python
#model.py

'''
Functions for running and visualizing basic evodoodle models
'''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import geonomics as gnx
import pygame

# Update the parameters with the custom landscapes
def set_landscapes(params, population_size, connectivity, environment):    
    """
    Update the parameters with custom landscapes.

    This function takes a parameter dictionary and three custom landscape matrices,
    and updates the parameter dictionary with these landscapes.

    Args:
        params (dict): A dictionary of model parameters.
        population_size (numpy.ndarray): A 2D array representing the population size landscape.
        connectivity (numpy.ndarray): A 2D array representing the connectivity landscape.
        environment (numpy.ndarray): A 2D array representing the environmental landscape.

    Returns:
        dict: The updated parameter dictionary.
    """
    params['landscape']['main']['dim'] = population_size.shape
    params['landscape']['layers']['population_size']['init']['defined']['rast'] = population_size
    params['landscape']['layers']['connectivity']['init']['defined']['rast'] = connectivity
    params['landscape']['layers']['environment']['init']['defined']['rast'] = environment
    return(params)


# Initialize the model with the custom landscapes
def init_mod(params, population_size, connectivity, environment):
    """
    Initialize the model with custom landscapes.

    This function takes a parameter dictionary and three custom landscape matrices,
    initializes a Geonomics model with these parameters, and runs a burn-in period.

    Args:
        params (dict): A dictionary of model parameters.
        population_size (numpy.ndarray): A 2D array representing the population size landscape.
        connectivity (numpy.ndarray): A 2D array representing the connectivity landscape.
        environment (numpy.ndarray): A 2D array representing the environmental landscape.

    Returns:
        gnx.Model: An initialized Geonomics model.
    """
    # Add our custom matrices to the geonomics parameters
    params = set_landscapes(params, population_size, connectivity, environment)
    # Make our params dict into a proper Geonomics ParamsDict object
    params = gnx.make_params_dict(params, 'demo')
    # Then use it to make a model
    mod = gnx.make_model(parameters=params, verbose=True)
    # Burn in the model for 10000 steps
    mod.walk(T=10000, mode='burn')
    return mod

# function for running and plotting genetic PCA
def plot_PCA(mod, ax=None):
    """
    Plot the genetic Principal Component Analysis (PCA) of the model.

    Args:
        mod (gnx.Model): A Geonomics model object.
        ax (matplotlib.axes.Axes, optional): An axes object to plot on. If None, a new figure is created.

    Returns:
        None: This function displays the plot but does not return any value.
    """
    from copy import deepcopy
    from sklearn.decomposition import PCA
    figsize = 6
    species = mod.comm[0]
    speciome = np.mean(species._get_genotypes(), axis=2)
    pca = PCA(n_components=3)
    PCs = pca.fit_transform(speciome)
    norm_PCs = (PCs - np.min(PCs, axis=0)) / (np.max(PCs, axis=0) - np.min(PCs, axis=0))
    PC_colors = norm_PCs * 255  # use normalized PC values to get colors
    if ax is None:
        fig, ax = plt.subplots(figsize=(figsize, figsize), dpi= 80, facecolor='w', edgecolor='k')
    ax.scatter(norm_PCs[:,0], norm_PCs[:, 1], color = PC_colors/255.0)  # use PC_colors as colors
    ax.set_xlabel('genetic PC 1')
    ax.set_ylabel('genetic PC 2')

# function for running and plotting genetic PCA
def map_PCA(mod, lyr_num=0, mask=True, ax=None):
    """
    Map the genetic Principal Component Analysis (PCA) results onto the landscape.

    Args:
        mod (gnx.Model): A Geonomics model object.
        lyr_num (int, optional): The layer number to use as background. Default is 0.
        mask (bool, optional): Whether to mask out areas with no individuals. Default is True.
        ax (matplotlib.axes.Axes, optional): An axes object to plot on. If None, a new figure is created.

    Returns:
        None: This function displays the plot but does not return any value.
    """
    if lyr_num is None:
        lyr_num = 0

    from copy import deepcopy
    from sklearn.decomposition import PCA
    cmaps = {0: plt.cm.RdBu.copy(), 1: plt.cm.BrBG_r.copy()}
    mark_size = 60
    figsize = 8
    species = mod.comm[0]
    land = mod.land
    # get array of resulting genomic data (i.e. 'speciome'),
    # genotypes meaned by individual
    speciome = np.mean(species._get_genotypes(), axis=2)
    # run PCA on speciome
    pca = PCA(n_components=3)
    PCs = pca.fit_transform(speciome)
    # normalize the PC results
    norm_PCs = (PCs - np.min(PCs,
                             axis=0)) / (np.max(PCs,
                                                axis=0) - np.min(PCs,
                                                                 axis=0))
    # use first 3 PCs to get normalized values for R, G, & B colors
    PC_colors = norm_PCs * 255
    # scatter all individuals on top of landscape, colored by the
    # RBG colors developed from the first 3 geonmic PCs
    xs = mod.comm[0]._get_x()
    ys = mod.comm[0]._get_y()
    # get environmental raster, with barrier masked out
    env = deepcopy(mod.land[lyr_num].rast)
    if mask:
        env[mod.land[lyr_num].rast == 0] = np.nan
    # create light colormap for plotting landscape
    #cmap = cmaps[lyr_num]
    #cmap.set_bad(color='#8C8C8C')
    cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True, reverse=True)
    if ax is None:
        fig, ax = plt.subplots(figsize=(figsize, figsize), dpi= 80, facecolor='w', edgecolor='k')
    ax.pcolormesh(land._x_cell_bds, land._y_cell_bds, env, cmap=cmap, vmax=1, vmin=0)
    ax.scatter(xs, ys, c=PC_colors/255.0, s=mark_size, edgecolors='black')
    [f([dim for dim in (0, mod.land.dim[0])]) for f in (ax.set_xlim, ax.set_ylim)]

    # Flip the y-axis
    ax.invert_yaxis()

    ax.set_xticks([])
    ax.set_yticks([])

# Combined function for plotting PCA and PCA map
def plot_pca(mod):
    """
    Plot both the genetic PCA and its mapping on the landscape.

    Args:
        mod (gnx.Model): A Geonomics model object.

    Returns:
        None: This function displays the plot but does not return any value.
    """
    # Create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the PCA on the first subplot
    plot_PCA(mod, ax=ax1)

    # Plot the PCA map on the second subplot
    map_PCA(mod, lyr_num=0, mask=False, ax=ax2)

    # Display the figure
    plt.tight_layout()
    plt.show()

# Function for plotting heterozygosity
def plot_popgen(mod, lyr_num=None):
    """
    Plot population genetics information including PCA, PCA map, heterozygosity, and phenotype.

    Args:
        mod (gnx.Model): A Geonomics model object.
        lyr_num (tuple of int, optional): Layer numbers to use for each plot. Default is (0, 1, 2).

    Returns:
        None: This function displays the plot but does not return any value.
    """

    if lyr_num is None:
        lyr_num = (0, 1, 2)

    # Get layer numbers
    if isinstance(lyr_num, int):
        (lyr_num, lyr_num, lyr_num)

    # Create a GridSpec object
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1.2, 1.2])  # Adjust the width ratios to give more space to the heterozygosity plot

    # Create a figure
    fig = plt.figure(figsize=(20,5))

    # Create the subplots using the GridSpec object
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])

    # Set the titles for each subplot
    ax1.set_title('PCA')
    ax2.set_title('PCA map')
    ax3.set_title('Heterozygosity')
    ax4.set_title('Phenotype')

    # Plot the PCA on the first subplot
    plot_PCA(mod, ax=ax1)

    # Plot the PCA map on the second subplot
    map_PCA(mod, lyr_num=lyr_num[0], mask=False, ax=ax2)

    # Plot the heterozygosity on the third subplot
    plot_heterozygosity(mod, lyr_num=lyr_num[1], ax=ax3)

    # Plot the phenotype on the fourth subplot
    plot_phenotype(mod, trait=0, lyr_num=lyr_num[2], ax=ax4)

    # Display the figure
    plt.tight_layout()
    plt.show()

# Function for plotting genetic diversity
def plot_heterozygosity(mod, lyr_num=0, ax=None):
    """
    Plot the heterozygosity of individuals on the landscape.

    Args:
        mod (gnx.Model): A Geonomics model object.
        lyr_num (int, optional): The layer number to use as background. Default is 0.
        ax (matplotlib.axes.Axes, optional): An axes object to plot on. If None, a new figure is created.

    Returns:
        None: This function displays the plot but does not return any value.
    """
    if lyr_num is None:
        lyr_num = 0

    # Calculate heterozygosity
    genotypes = mod.comm[0]._get_genotypes()
    heterozygosity = _calculate_heterozygosity(genotypes)

    # Get the x and y coordinates of each individual
    xs = mod.comm[0]._get_x()
    ys = mod.comm[0]._get_y()

    # Get the specified layer
    layer = mod.land[lyr_num].rast

    # Define the color maps
    #cmaps = {0: plt.cm.RdBu.copy(), 1: plt.cm.BrBG_r.copy()}
    #cmap = cmaps[lyr_num]
    #cmap.set_bad(color='#8C8C8C')
    cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True, reverse=True)

    # If no axes object is passed, create a new figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')

    # Plot the layer as the background
    ax.pcolormesh(mod.land._x_cell_bds, mod.land._y_cell_bds, layer, cmap=cmap, vmax=1, vmin=0)

    # Create a scatter plot of individuals, colored by heterozygosity
    scatter = ax.scatter(xs, ys, c=heterozygosity, edgecolors='black')

    # Set the colorbar without a label
    plt.colorbar(scatter, ax=ax)

    # Set the x and y limits
    [f([dim for dim in (0, mod.land.dim[0])]) for f in (ax.set_xlim, ax.set_ylim)]

    # Flip the y-axis
    ax.invert_yaxis()

    # Get rid of x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

# Calculate heterozygosity for individuals

def _calculate_heterozygosity(genotypes):
    """Internal helper function to calculate heterozygosity."""
    # Count the number of heterozygous loci for each individual
    heterozygous_loci = np.sum(genotypes[:, :, 0] != genotypes[:, :, 1], axis=1)

    # Calculate the proportion of heterozygous loci
    heterozygosity = heterozygous_loci / genotypes.shape[1]

    return heterozygosity

# Plot the phenotype of a trait
def plot_phenotype(mod, trait=0, lyr_num=2, ax=None):
    """
    Plot the phenotype of a trait on the landscape.

    Args:
        mod (gnx.Model): A Geonomics model object.
        trait (int, optional): The trait number to plot. Default is 0.
        lyr_num (int, optional): The layer number to use as background. Default is 2.
        ax (matplotlib.axes.Axes, optional): An axes object to plot on. If None, a new figure is created.

    Returns:
        None: This function displays the plot but does not return any value.
    """
    if lyr_num is None:
        lyr_num = 2

    # Get the phenotype of the trait
    zs = mod.comm[0]._get_z()[:, trait]
    xs = mod.comm[0]._get_x()
    ys = mod.comm[0]._get_y() 

    # Get the specified layer
    layer = mod.land[lyr_num].rast

    # Define the color map
    cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True, reverse=True)

    # If no axes object is passed, create a new figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')

    # Plot the layer as the background
    ax.pcolormesh(mod.land._x_cell_bds, mod.land._y_cell_bds, layer, cmap=cmap, vmin=0, vmax=1)

    # Create a scatter plot of individuals, colored by phenotype
    scatter = ax.scatter(xs, ys, c=zs, edgecolors='black', cmap=cmap, vmin=0, vmax=1)

    # Set the colorbar with a label
    plt.colorbar(scatter, ax=ax)

    # Set the x and y limits
    [f([dim for dim in (0, mod.land.dim[0])]) for f in (ax.set_xlim, ax.set_ylim)]

    # Flip the y-axis
    ax.invert_yaxis()

    # Get rid of x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Set the title of the plot
    ax.set_title('Phenotype for trait {}'.format(trait))


# Plot the landscape models (default gnx plotting)
def plot_model(mod):
    """
    Plot the model landscapes using default Geonomics plotting.

    Args:
        mod (gnx.Model): A Geonomics model object.

    Returns:
        None: This function displays the plot but does not return any value.
    """
    # Create a figure
    fig = plt.figure()

    # Create the first subplot and set its title
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title('Population size')
    plt.sca(ax1)
    mod.plot(spp=0, lyr=0)

    # Create the second subplot and set its title
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title('Connectivity')
    plt.sca(ax2)
    mod.plot(spp=0, lyr=1)

    # Create the third subplot and set its title
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title('Phenotype')
    plt.sca(ax3)
    mod.plot_phenotype(spp=0, trt=0)

    # Display the figure
    plt.show()