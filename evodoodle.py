import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import geonomics as gnx
import pygame

# Draw a landscape
def draw_landscape(d=10):
    landscape = draw_landscape_helper(d=d)
    return landscape

# Edit a landscape
def edit_landscape(landscape):
    # Colors
    cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True, reverse=True, light = 0.7) 
    BLACK = tuple((np.array(cmap(0.0)[:3]) * 255).astype(int))
    GRAY = tuple((np.array(cmap(0.5)[:3]) * 255).astype(int))
    WHITE = tuple((np.array(cmap(1.0)[:3]) * 255).astype(int))
    
    # Create a dictionary that maps vallues to colors
    value_to_color = {0: BLACK, 0.5: GRAY, 1: WHITE}
    
    # Convert the 2D landscape array back to a 3D drawing matrix
    drawing_matrix = np.array([[value_to_color.get(landscape[i, j], BLACK) for j in range(landscape.shape[1])] for i in range(landscape.shape[0])])
    
    # Edit the landscape
    landscape = draw_landscape_helper(d=landscape.shape[0], drawing_matrix=drawing_matrix)
    
    return landscape
    
# Helper function to draw a landscape
def draw_landscape_helper(d=10, drawing_matrix=None):
    # Initialize Pygame
    pygame.init()

    # Set the dimensions of the window and the drawing canvas
    canvas_size = (d, d)  # The canvas is dxd cells
    cell_size = 50  # Each cell is 50x50 pixels
    window_size = (canvas_size[0] * cell_size, canvas_size[1] * cell_size + 100)  # Extra 100 pixels for buttons

    # Create a window
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Drawing App")
    
    # Colors
    cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True, reverse=True, light = 0.7) 
    BLACK = tuple((np.array(cmap(0.0)[:3]) * 255).astype(int))
    GRAY = tuple((np.array(cmap(0.5)[:3]) * 255).astype(int))
    WHITE = tuple((np.array(cmap(1.0)[:3]) * 255).astype(int))
    GREEN = (170, 255, 0) 

    colors = [BLACK, GRAY, WHITE]
    color_index = 0  # Default value set to 1

    # Create a matrix to store the drawing if one is not provided
    if drawing_matrix is None:
        drawing_matrix = np.full((canvas_size[0], canvas_size[1], 3), WHITE)

    # Button dimensions
    button_width = 50
    button_height = 50
    button_y = canvas_size[1] * cell_size + 25  # Position the buttons 25 pixels below the canvas
    button_spacing = 20

    # Adjust button layout based on landscape size
    if d <= 6:
        button_spacing = 5  # Decrease spacing
        button_width = 40  # Decrease button size
        button_height = 40

    button_x_values = [
        button_spacing,  # First button aligned to the left
        button_spacing + button_width + button_spacing,  # Second button
        button_spacing + button_width * 2 + button_spacing * 2,  # Third button
        window_size[0] - button_width * 2 - button_spacing  # Save button aligned to the right
    ]
    button_widths = [button_width, button_width, button_width, button_width * 2]  # Double the width for the Save button
    button_rects = [pygame.Rect(x, button_y, w, button_height) for x, w in zip(button_x_values, button_widths)]
    
    # Main loop
    running = True
    while running:
        # Fill the screen with white color
        screen.fill((255, 255, 255))

        # Draw a black rectangle over the canvas area
        pygame.draw.rect(screen, BLACK, pygame.Rect(0, 0, canvas_size[0] * cell_size, canvas_size[1] * cell_size))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[0]:  # Left mouse button is held down
                    x, y = pygame.mouse.get_pos()
                    grid_x, grid_y = x // cell_size, y // cell_size
                    if 0 <= grid_x < canvas_size[0] and 0 <= grid_y < canvas_size[1]:
                        drawing_matrix[grid_y, grid_x] = colors[color_index]
                    
                    for i, rect in enumerate(button_rects):
                        if rect.collidepoint(x, y):
                            if i < 3:  # Color buttons
                                color_index = i # 0, 0.5, 1
                            else:  # Save button
                                running = False  # End the game when the Save button is clicked
        
        # Draw the grid
        for y in range(canvas_size[1]):
            for x in range(canvas_size[0]):
                color = drawing_matrix[y, x]
                pygame.draw.rect(screen, color, pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size))

        # Draw grid lines
        for x in range(0, canvas_size[0] * cell_size, cell_size):
            pygame.draw.line(screen, BLACK, (x, 0), (x, canvas_size[1] * cell_size))
        for y in range(0, canvas_size[1] * cell_size, cell_size):
            pygame.draw.line(screen, BLACK, (0, y), (canvas_size[0] * cell_size, y))
        
        # Draw the buttons
        for i, rect in enumerate(button_rects):
            if i < 3:  # Color buttons
                pygame.draw.rect(screen, colors[i], rect)
                font = pygame.font.Font(None, 32)  # Change the size as needed
                text_surface = font.render(str(i/2), True, (255, 255, 255))  
                text_rect = text_surface.get_rect(center=rect.center)  # Center the text
                screen.blit(text_surface, text_rect)
            else:  # Save button
                pygame.draw.rect(screen, GREEN, rect)
                font = pygame.font.Font(None, 32)  # Change the size as needed
                text_surface = font.render("SAVE", True, (0, 0, 0))  # Black text
                text_rect = text_surface.get_rect(center=rect.center)  # Center the text
                screen.blit(text_surface, text_rect)
            pygame.draw.rect(screen, BLACK, rect, 2)  # Draw a black outline around the button
        
        # Update the display
        pygame.display.flip()

    # Quit Pygame
    pygame.quit()

    # Convert the 3D drawing matrix to a 2D array with the specified values
    # Create a dictionary that maps colors to values
    color_to_value = {tuple(BLACK): 0, tuple(GRAY): 0.5, tuple(WHITE): 1}
    # Create the 2D array
    landscape = np.array([[color_to_value.get(tuple(drawing_matrix[i, j]), 0) for j in range(drawing_matrix.shape[1])] for i in range(drawing_matrix.shape[0])])
    
    # Return the drawing matrix
    return landscape

# Plot the landscapes
def plot_landscapes(population_size, connectivity, environment):
    # Create a figure and a set of subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    # Set titles for each subplot
    ax1.set_title('Population Size')
    ax2.set_title('Connectivity')
    ax3.set_title('Environment')

    # Define a colormap that uses the seaborn "crest" color palette
    cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True, reverse=True)

    # Display the matrices
    im1 = ax1.imshow(population_size, cmap=cmap, vmin=0, vmax=1)
    im2 = ax2.imshow(connectivity, cmap=cmap, vmin=0, vmax=1)
    im3 = ax3.imshow(environment, cmap=cmap, vmin=0, vmax=1)

    # Add a colorbar
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    fig.colorbar(im3, ax=ax3)

    plt.show()

# Update the parameters with the custom landscapes
def set_landscapes(params, population_size, connectivity, environment):    
    params['landscape']['main']['dim'] = population_size.shape
    params['landscape']['layers']['population_size']['init']['defined']['rast'] = population_size
    params['landscape']['layers']['connectivity']['init']['defined']['rast'] = connectivity
    params['landscape']['layers']['environment']['init']['defined']['rast'] = environment
    return(params)

# Initialize the model with the custom landscapes
def init_mod(params, population_size, connectivity, environment):
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
    if lyr_num is None:
        lyr_num = 0

    # Calculate heterozygosity
    genotypes = mod.comm[0]._get_genotypes()
    heterozygosity = calculate_heterozygosity(genotypes)

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
def calculate_heterozygosity(genotypes):
    # Count the number of heterozygous loci for each individual
    heterozygous_loci = np.sum(genotypes[:, :, 0] != genotypes[:, :, 1], axis=1)

    # Calculate the proportion of heterozygous loci
    heterozygosity = heterozygous_loci / genotypes.shape[1]

    return heterozygosity

# Plot the phenotype of a trait
def plot_phenotype(mod, trait=0, lyr_num=2, ax=None):
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