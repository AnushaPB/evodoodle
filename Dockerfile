# Use the official Python image as a parent image
FROM python:3.9

# Set the working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter
RUN pip install jupyter

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Set the default command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
