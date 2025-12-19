FROM docker.io/nvidia/cuda:12.6.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    vim \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=/usr/local bash

# Install Nsight Systems for profiling
ARG NSYS_URL=https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_5/
ARG NSYS_PKG=nsight-systems-2025.5.1_2025.5.1.121-1_amd64.deb
RUN cd /tmp && \
    wget ${NSYS_URL}${NSYS_PKG} && \
    apt-get update && \
    dpkg -i ./${NSYS_PKG} || apt-get install -y -f && \
    rm -rf /tmp/*

# Create a user that matches the host user
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=developer

RUN if ! getent group ${GROUP_ID} >/dev/null 2>&1; then \
        groupadd -g ${GROUP_ID} ${USERNAME}; \
    fi && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash ${USERNAME}

# Set environment variables for pixi
ENV PATH="/usr/local/bin:${PATH}"
ENV PIXI_HOME="/usr/local"

# Set working directory
WORKDIR /workspace

# Note: pixi install and recovar installation will be done at runtime
# This allows the container to work with different versions of the code

ENTRYPOINT ["/bin/bash"]