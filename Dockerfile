FROM python:3.12-slim

# Build args make container-written files owned by your host user (passed from
# docker-compose), so notebooks and generated plots aren't root-owned on your host.
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=worker

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g ${GROUP_ID} ${USER_NAME} || groupadd ${USER_NAME} && \
    useradd -l -u ${USER_ID} -g ${GROUP_ID} -m ${USER_NAME}

# CPU-only PyTorch — this demo trains on a laptop CPU in under a minute.
# Swap the index URL for a CUDA build if you run on a GPU.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
RUN chown ${USER_NAME}:${USER_NAME} /app
USER ${USER_NAME}

EXPOSE 8888
