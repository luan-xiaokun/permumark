FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgmp-dev \
    wget \
    ca-certificates \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- Miniforge Installation ---
ENV MINIFORGE_VERSION="Miniforge3-Linux-x86_64"
ENV MINIFORGE_PATH="/opt/miniforge3"

RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/${MINIFORGE_VERSION}.sh -O ~/miniforge.sh

RUN bash ~/miniforge.sh -b -p ${MINIFORGE_PATH} && \
    rm ~/miniforge.sh

ENV PATH="${MINIFORGE_PATH}/bin:${PATH}"

# --- Conda Environment Setup for PermuMark ---
RUN conda create -n permumark python=3.11 -y

RUN conda init bash

RUN echo "conda activate permumark" >> ~/.bashrc

SHELL ["conda", "run", "-n", "permumark", "/bin/bash", "-c"]

RUN echo "Conda environment 'permumark' activated." && \
    echo "Python version in permumark env:" && python --version

RUN conda install sage=10.4 -y

RUN echo "Verifying SageMath installation..." && \
    sage -c "print('SageMath version:', sage.version.version); print('SageMath configured correctly.')" || \
    (echo "SageMath verification command failed, but continuing build. Check SageMath manually if issues arise." && exit 0)


# --- Application Setup ---
WORKDIR /app

COPY . /app/

# Build the shared library for derangement
RUN echo "Building shared library..." && \
    ./scripts/build.sh && \
    echo "Shared library build script executed."

# Install the permumark package and its dependencies using pip
RUN echo "Installing PermuMark package and its dependencies..." && \
    pip install -e . && \
    echo "PermuMark installation complete."

# --- Evaluation Specific Dependencies ---
RUN pip install "huggingface_hub[cli]" gekko

RUN mkdir -p datasets

RUN huggingface-cli download --repo-type dataset Salesforce/wikitext --local-dir datasets/Salesforce/wikitext

# Download Llama-3.2-1B from unsloth's Hugging Face repository, as it does not require an account token
RUN huggingface-cli download --repo-type model unsloth/Llama-3.2-1B-Instruct --local-dir models/meta-llama/Llama-3.2-1B

# Install AutoGPTQ from the specified fork
RUN echo "Installing AutoGPTQ..." && \
    git clone https://github.com/PanQiWei/AutoGPTQ.git /opt/AutoGPTQ && \
    cd /opt/AutoGPTQ && \
    BUILD_CUDA_EXT=0 pip install -vvv --no-build-isolation -e . && \
    cd /app && \
    echo "AutoGPTQ installation complete."

# --- Final Instructions and Default Command ---
RUN echo "Build complete. Environment 'permumark' is ready." && \
    echo "To use this image:" && \
    echo "1. Run the container: docker run -it --rm your-image-name:tag bash" && \
    echo "2. Inside the container, you may need to log in to Hugging Face:" && \
    echo "   huggingface-cli login" && \
    echo "3. Then, you can run your download and evaluation scripts, e.g.:" && \
    echo "   ./scripts/download.sh" && \
    echo "   python evaluation/evaluation.py ..."

# Set a default command to start a bash shell when the container runs
CMD ["bash"]

# TODO
# [x] Make sure conda environment is activated by default when the container starts
# [x] Download the dataset (WikiText), this can be done without any account token
# [x] Download a proximal Llama3.2-1B model for illustration purposes
# [ ] Download and extract fine-tuned model weights