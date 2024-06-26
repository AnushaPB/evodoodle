# Use the official Python image as a parent image
FROM python:3.9

# Set the working directory
WORKDIR /workspace

# Install system dependencies for pygame and Xvfb
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
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install geopandas rasterio matplotlib scipy bitarray tskit scikit-learn statsmodels msprime psutil nlmpy numpy matplotlib seaborn geonomics pygame

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Set the default command to run Jupyter Notebook
CMD ["xvfb-run", "-s", "-screen 0 1400x900x24", "jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
