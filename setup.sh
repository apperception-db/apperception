#!/bin/bash
git clone https://github.com/theAIGuysCode/yolov4-deepsort.git
mv ./yolov4-tiny.weights ./yolov4-deepsort/data
mv ./yolov4.weights ./yolov4-deepsort/data
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
mv checkpoints/ ./yolov4-deepsort
mv ./config.py ./yolov4-deepsort/core
mkdir output
