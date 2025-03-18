# Usa Miniconda come immagine base
FROM continuumio/miniconda3

# Imposta la directory di lavoro
WORKDIR /app

# Clona il repository da GitHub
RUN git clone https://github.com/genpat-it/dist2mst.git .

# Crea l'ambiente Conda usando l'environment.yml del repository
RUN conda env create -f environment.yml

# Attiva l'ambiente Conda automaticamente
ENV PATH /opt/conda/envs/dist2mst/bin:$PATH

# Imposta l'entrypoint per eseguire lo script dist2mst.py
ENTRYPOINT ["python", "dist2mst.py"]

# Comando predefinito (mostra l'help)
CMD ["--help"]