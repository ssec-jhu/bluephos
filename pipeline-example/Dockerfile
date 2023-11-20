# Use an official Miniconda runtime as a parent image
FROM continuumio/miniconda3

# Set the working directory in the container to /app
WORKDIR /app/mol_discovery/pipeline-example

# Copy the Conda environment file into the container
COPY mol_discovery_env.yml /app/mol_discovery_env.yml

# Install the Conda environment
RUN conda env create -f /app/mol_discovery_env.yml

# Ensure that the environment is activated on login
RUN echo "source activate mol_discovery_env" > ~/.bashrc
ENV PATH /opt/conda/envs/mol_discovery_env/bin:$PATH

# Copy the mol_discovery project directory into the container at /app
COPY mol_discovery /app/mol_discovery

# Define environment variables for ORCA paths
ENV ORCA_PATH /home/idies/workspace/ssec/mol_discover/orca
ENV EBROOTORCA /home/idies/workspace/ssec/mol_discover/orca
ENV PATH /home/idies/workspace/ssec/mol_discover/orca:$PATH
ENV LD_LIBRARY_PATH /lib/x86_64-linux-gnu:/opt/conda/envs/mol_discovery_env/lib:/opt/conda/pkgs/openmpi-4.1.5-h414af15_101/lib:/home/idies/workspace/ssec/mol_discover/orca 

# Set the environment variables for Open MPI
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Define the default command to run when starting the container
CMD ["conda", "run", "-n", "mol_discovery_env", "/bin/bash"]

