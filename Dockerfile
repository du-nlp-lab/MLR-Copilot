FROM anibali/pytorch:2.0.0-cuda11.8-ubuntu22.04

# Set up time zone.
ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

WORKDIR /app
USER root
RUN apt update && apt install -y gcc-10 g++-10 && ln /usr/bin/gcc-10 /usr/bin/gcc && ln /usr/bin/g++-10 /usr/bin/g++ && apt install -y zlib1g-dev && rm -r /var/lib/apt/lists/*
USER user

# setup llama
COPY codellama .
COPY ReactAgent .

# Add the current directory contents into the container at /app
COPY AutoResearch/install.sh .
COPY requirements.txt .

# Install libraries 

# RUN conda create -n autogpt python=3.10
# Make RUN commands use the new environment:
# RUN conda init bash
# SHELL ["conda", "run", "-n", "autogpt", "/bin/bash", "-c"]

RUN bash install.sh

# RUN echo "conda init bash" > ~/.bashrc
# RUN echo "source activate autogpt" > ~/.bashrc
# ENV PATH /opt/conda/envs/envname/bin:$PATH

