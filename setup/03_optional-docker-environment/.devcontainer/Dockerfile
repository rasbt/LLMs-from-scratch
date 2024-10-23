# Install PyTorch 2.5 with CUDA 12.4
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# Install Ubuntu packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y rsync && \
    apt-get install -y git && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
