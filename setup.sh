#!/bin/bash
git clone https://github.com/theAIGuysCode/yolov4-deepsort.git
mv ./yolov4-tiny.weights ./yolov4-deepsort/data
mv ./yolov4.weights ./yolov4-deepsort/data
pip3 install -r requirements.txt
mv checkpoint/ ./yolov4-deepsort
mv ./config.py ./yolov4-deepsort/core
mkdir output
