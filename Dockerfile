FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install base utilities
RUN apt-get update && \
    # apt-get install -y build-essentials && \
    apt-get install -y wget ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Mamba
ENV MAMBA_DIR /opt/mamba
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh -O ~/mamba.sh && \
     /bin/bash ~/mamba.sh -b -p /opt/mamba

# Put mamba in path so we can use mamba activate
ENV PATH=$MAMBA_DIR/bin:$PATH

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES video,compute,utility


WORKDIR /apperception

COPY ./environment.yml .
RUN mamba env create -f environment.yml

COPY ./pyproject.toml .
RUN mamba run -n apperception poetry install

RUN mamba run -n apperception pip uninstall -y -q torch torchaudio torchvision torchtext
RUN mamba run -n apperception pip install --progress-bar off --pre torch torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu117 2> /dev/null
RUN mamba run -n apperception pip3 install --progress-bar off boto3 2> /dev/null
RUN wget -q -O input.mp4 "https://download.pytorch.org/torchaudio/tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4"

COPY ./apperception ./apperception
COPY ./benchmarks ./benchmarks
COPY ./optimized_ingestion ./optimized_ingestion
COPY ./playground ./playground
COPY ./scripts ./scripts
COPY ./submodules ./submodules
COPY ./typings ./typings
COPY ./weights ./weights
COPY ./yolov5s.pt ./yolov5s.pt

ENV AP_PORT='25441'
ENV NUSCENES_DATASET='Mini'

ENV NUSCENES_RAW_DATA="/work/apperception/data/raw/nuScenes/full-dataset-v1.0/$NUSCENES_DATASET"
ENV NUSCENES_PROCESSED_DATA="/data/apperception-data/processed/nuscenes/full-dataset-v1.0/$NUSCENES_DATASET"

ENV EXPERIMENT_DATA="/work/apperception/data/raw/scenic/experiment_data"
ENV NUSCENES_RAW_ROAD="/work/apperception/data/raw/road-network"
ENV NUSCENES_PROCESSED_ROAD="/data/apperception-data/processed/road-network"
ENV NUSCENES_RAW_MAP="/work/apperception/data/raw/nuScenes/Map-expansion"
ENV NUSCENES_PROCESSED_MAP="/data/apperception-data/processed/nuscenes/Map-expansion"


CMD ["mamba", "run", "-n", "apperception", "python", "-m", "jupyter", "notebook", "--port=8085", "--allow-root"]
