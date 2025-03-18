# Use Miniconda as the base image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy your custom environment.yml into the container
COPY environment.yml .

# Create the Conda environment using your environment.yml
RUN conda env create -f environment.yml

# Ensure the Conda environment is activated by setting the PATH
ENV PATH="/opt/conda/envs/dist2mst_env/bin:$PATH"

# Clone the repository into a temporary directory and move only dist2mst.py
RUN git clone https://github.com/genpat-it/dist2mst.git /tmp/dist2mst --depth 1 && \
    mv /tmp/dist2mst/dist2mst.py . && \
    rm -rf /tmp/dist2mst

# Set the entrypoint to run the dist2mst.py script
ENTRYPOINT ["python", "dist2mst.py"]

# Default command (displays help)
CMD ["--help"]