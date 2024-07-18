FROM anibali/pytorch:2.0.0-cuda11.8-ubuntu22.04

# Set up time zone.
ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

WORKDIR /app
USER root
RUN apt update && apt install -y gcc-10 g++-10 && ln /usr/bin/gcc-10 /usr/bin/gcc && ln /usr/bin/g++-10 /usr/bin/g++ && apt install -y zlib1g-dev && rm -r /var/lib/apt/lists/*
USER user

# copy files
COPY codellama .
COPY reactagent .

# Install libraries 
WORKDIR /app/reactagent
RUN bash install.sh
RUN python3 -m pip install -e .

WORKDIR /app/codellama
RUN python3 -m pip install -e .

WORKDIR /app

# start bash shell
CMD bash
