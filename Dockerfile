# Use the official Python image as a parent image
FROM python:3.12.4

# Set the working directory
WORKDIR /workspace

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

# Install Python packages
RUN pip install geopandas rasterio matplotlib scipy bitarray tskit scikit-learn statsmodels msprime psutil nlmpy numpy matplotlib seaborn geonomics pygame ipykernel

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Set the default command to run Jupyter Notebook
CMD ["start-notebook.sh", "--NotebookApp.token=''"]

# From: https://github.com/microsoft/vscode-dev-containers/blob/main/script-library/docs/desktop-lite.md
COPY.devcontainer/library-scripts/desktop-lite-debian.sh /tmp/library-scripts/
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

