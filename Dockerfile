# Use a specific version of the official Python image as a parent image
FROM python:3.9.13

# Set the working directory
WORKDIR /workspace

# Install system dependencies for pygame and other libraries
RUN apt-get update && apt-get install -y \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libsmpeg-dev \
    libportmidi-dev \
    libavformat-dev \
    libswscale-dev \
    libjpeg-dev \
    zlib1g-dev \
    libgdal-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    geopandas \
    rasterio \
    matplotlib \
    scipy \
    bitarray \
    tskit \
    scikit-learn \
    statsmodels \
    msprime \
    psutil \
    nlmpy \
    numpy \
    matplotlib \
    seaborn \
    geonomics \
    pygame

# Create a non-root user and switch to it
RUN useradd -m vscode && chown -R vscode /workspace
USER vscode

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Set the default command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

