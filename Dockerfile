FROM python:3.12-slim

# Build args let the container create files owned by YOUR host user (passed from
# docker-compose), so notebooks and run artifacts written into the mounted volume
# aren't root-owned on your laptop.
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=worker

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user matching the host UID/GID.
RUN groupadd -g ${GROUP_ID} ${USER_NAME} || groupadd ${USER_NAME} && \
    useradd -l -u ${USER_ID} -g ${GROUP_ID} -m ${USER_NAME}

# CPU-only PyTorch (this demo trains on a laptop CPU in well under a minute).
# Swap this index for a CUDA build if you run on a GPU box.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Framework dependencies.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
RUN chown ${USER_NAME}:${USER_NAME} /app
USER ${USER_NAME}

EXPOSE 8888 5000
