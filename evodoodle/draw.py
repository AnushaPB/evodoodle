

'''
Functions for drawing custom landscapes and editing existing landscapes
'''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import geonomics as gnx
import pygame

# Draw a landscape
def draw_landscape(d=10):
    """
    Draw a custom landscape. 
    
    This function opens a Pygame window where the user can draw a landscape by clicking and dragging the mouse. The landscape is represented as a grid of cells with different values that can be changed by clicking on them. Once you are done drawing, click the "SAVE" button.

    Args:
        d (int): The dimension of the landscape grid. Default is 10.

    Returns:
        numpy.ndarray: A 2D numpy array representing the drawn landscape.
    """
    landscape = _draw_landscape_helper(d=d)
    return landscape

# Edit a landscape
def edit_landscape(landscape):
    """
    Edit an existing landscape using a Pygame-based interface.

    This function opens a Pygame window with the provided landscape loaded, allowing the user to modify it by clicking and dragging the mouse.

    Args:
        landscape (numpy.ndarray): A 2D numpy array representing the landscape to edit.

    Returns:
        numpy.ndarray: A 2D numpy array representing the edited landscape.
    """
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
    landscape = _draw_landscape_helper(d=landscape.shape[0], drawing_matrix=drawing_matrix)
    
    return landscape
    
# Helper function to draw a landscape
def _draw_landscape_helper(d=10, drawing_matrix=None):
    """Internal helper function for drawing landscapes."""
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
    """
    Plot the three landscape matrices side by side.

    Args:
        population_size (numpy.ndarray): A 2D array representing the population size landscape.
        connectivity (numpy.ndarray): A 2D array representing the connectivity landscape.
        environment (numpy.ndarray): A 2D array representing the environmental landscape.

    Returns:
        None: This function displays the plot but does not return any value.
    """
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
