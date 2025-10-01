# ----------------------------------------------------------
# Base: Ubuntu with NVIDIA CUDA (runtime only, GPU support)
# ----------------------------------------------------------
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------------------------------------
# System dependencies
# ----------------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    git wget curl unzip \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default "python"
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# ----------------------------------------------------------
# Python dependencies
# ----------------------------------------------------------
# Install PyTorch (CUDA version matching base image), FiftyOne, UMAP
# torch==2.1.0+cu121 is compatible with CUDA 12.1 runtime
RUN pip install --no-cache-dir \
    torch torchvision torchaudio numpy ipywidgets anywidget \
    -f https://download.pytorch.org/whl/torch_stable.html \
    fiftyone umap-learn

# ----------------------------------------------------------
# App setup
# ----------------------------------------------------------
WORKDIR /app
COPY viz_embeddings_fiftyone.py /app/

# Expose FiftyOne App port
EXPOSE 5151

# Default command: run the visualization script
CMD ["python", "viz_embeddings_fiftyone.py"]
