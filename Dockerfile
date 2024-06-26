# Use the official Python image as a parent image
FROM python:3.9

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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to jovyan user to avoid permission issues
USER ${NB_UID}

# Install Python packages
RUN pip install numpy matplotlib seaborn geonomics pygame

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Set the default command to run Jupyter Notebook
CMD ["start-notebook.sh", "--NotebookApp.token=''"]
