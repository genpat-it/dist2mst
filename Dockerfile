# Use Miniconda as the base image
FROM continuumio/miniconda3

LABEL maintainer="andreaderuvo"
LABEL org.opencontainers.image.source="https://github.com/genpat-it/dist2mst"
LABEL org.opencontainers.image.description="High-performance MST construction from distance matrices with Numba acceleration"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.version="0.0.2"

# Set the working directory
WORKDIR /app

# Copy and create the Conda environment
COPY environment.yml .
RUN conda env create -f environment.yml && \
    conda clean --all --yes

# Ensure the Conda environment is activated by setting the PATH
ENV PATH="/opt/conda/envs/dist2mst_env/bin:$PATH"

# Copy the application script directly from the build context
COPY dist2mst.py .

# Set the entrypoint to run the dist2mst.py script
ENTRYPOINT ["python", "dist2mst.py"]

# Default command (displays help)
CMD ["--help"]
