# Use the official Miniconda image as a parent image
FROM mambaorg/micromamba:latest

# Switch to root user to create the directory
USER root

# Create the working directory
RUN mkdir -p /workspaces

# Set the working directory
WORKDIR /workspaces

# Install system dependencies for pygame
USER root
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
    libgsl-dev \
    x11-apps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to jovyan user to avoid permission issues
USER ${NB_UID}

# Configure Mamba to use the conda-forge channel and install packages
RUN micromamba install -y -n base -c conda-forge msprime geopandas rasterio bitarray statsmodels psutil pip numpy==1.26.4 && \
    micromamba clean --all --yes

# Activate the base environment and install additional Python packages using pip
RUN echo "micromamba activate base" >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc && pip install --upgrade NLMpy geonomics"

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Set the default command to run Jupyter Notebook
CMD ["start-notebook.sh", "--NotebookApp.token=''"]

# From: https://github.com/microsoft/vscode-dev-containers/blob/main/script-library/docs/desktop-lite.md
COPY .devcontainer/library-scripts/desktop-lite-debian.sh /tmp/library-scripts/
RUN apt-get update && bash /tmp/library-scripts/desktop-lite-debian.sh
ENV DBUS_SESSION_BUS_ADDRESS="autolaunch:" \
    VNC_RESOLUTION="1440x768x16" \
    VNC_DPI="96" \
    VNC_PORT="5901" \
    NOVNC_PORT="6080" \
    DISPLAY=":1" \
    LANG="en_US.UTF-8" \
    LANGUAGE="en_US.UTF-8"
ENTRYPOINT ["/usr/local/share/desktop-init.sh"]
CMD ["sleep", "infinity"]

# Switch back to the default user
USER root