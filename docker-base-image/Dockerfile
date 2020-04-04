FROM python:3-slim
# REFERENCE:
# 1. https://github.com/LucasRoesler/pydatascience-template/blob/master/template/pydatascience/Dockerfile
# 2. https://pythonspeed.com/articles/base-image-python-docker-images/

RUN apt-get update \
    && apt-get -y install make \
        make \
        automake \
        gcc \
        g++ \
        subversion \
        musl-dev \
        libffi-dev \
        git \
    && apt-get clean
        #python3-dev \

# Add the testing https://stackoverflow.com/a/50443531/8853476
# Chose to move away from alpine due to issues in installing numpy pandas etc and in future Gurobi.
# With alpine and without numpy pandas etc the size was 140 MB| Now it is 413MB
# TODO: Decrease image size somehow

# Add non root user
RUN groupadd -r app && useradd app -r -m -g app
RUN chown app /home/app

USER app

ENV PATH=$PATH:/home/app/.local/bin

WORKDIR /home/app/

USER app
COPY requirements.txt   .
USER root
RUN pip install --no-cache-dir -r requirements.txt
USER app

WORKDIR /home/app/