#Download base image ubuntu 20.04
FROM ubuntu:20.04

# LABEL about the custom image
LABEL maintainer="tom"
LABEL version="0.1"
LABEL description="This is custom Docker Image for somo-rl"

# Disable Prompt During Packages Installation; not sure this is necessary
ARG DEBIAN_FRONTEND=noninteractive

# Update Ubuntu Software repository
RUN apt update

# install this to have command apt-add-repository available
RUN apt install software-properties-common -y

### Install python3.7 (bcs pybullet doesnt seem to work with 3.8 / 3.9) ##

# add repo for python 3.7 on ubuntu20.04 (not avail by default)
RUN add-apt-repository ppa:deadsnakes/ppa -y

# install python3.7
RUN apt-get update \
  && apt-get install -y python3.7 python3.7-dev

# Install pip
RUN apt install python3-pip -y
RUN python3.7 -m pip install --upgrade pip

# install pytest
RUN pip install pytest --no-cache-dir

# Make python 3.7 the default and alias it to py
RUN echo "alias py=python3.7" >> ~/.bashrc
RUN export PATH=${PATH}:/usr/bin/python3.7
RUN /bin/bash -c "source ~/.bashrc"

COPY somo repos/somo
RUN pip install -r repos/somo/requirements.txt --no-cache-dir
RUN pip install -e repos/somo

COPY somo_rl_requirements.txt repos/somo_rl_requirements.txt
RUN pip install -r repos/somo_rl_requirements.txt --no-cache-dir