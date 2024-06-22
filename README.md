# Evodoodle

Draw your own custom landscapes and watch as your species evolves across them! Evodoodle is a drawing game for learning how evolution plays out across landscapes. Evodoodle allows you to control population density, connectivity, and selection across space and then simulates evolution using Geonomics, a powerful landscape genomic simulation package.

# Setup

First, clone or download this repository. Once inside the evodoodle directory, you can then use the [evodoodle.yml](evodoodle.yml) file to set-up a conda environment and install the required packages

```bash
conda env create -f evodoodle.yml
conda activate evodoodle
```

You can also manually install the required packages:
```bash
pip install numpy
pip install matplotlib
pip install seaborn
pip install geonomics
```

# Quick start

To start evodoodle, simply run the following code. Whenever `draw_landscape()` is run a pop-up will appear that allows you to draw on a landscape. Once you have drawn your landscape, click `SAVE` and the code will continue:

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import geonomics as gnx
from evodoodle import make_landscapes, edit_landscapes, init_mod, draw_landscape, plot_popgen
from gnx_params import params
import geonomics as gnx

# Draw landscapes
population_size = draw_landscape()
connectivity = draw_landscape()
environment = draw_landscape()

# Start model
mod = init_mod(params, population_size, connectivity, environment)

# Run the model for 200 steps
mod.walk(200)

# Plot results
plot_popgen(mod)

# From here you can continue to run the model for more steps and plot the results
```
