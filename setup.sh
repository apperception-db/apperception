#!/bin/bash
git clone https://github.com/theAIGuysCode/yolov4-deepsort.git
mv ./yolov4-tiny.weights ./yolov4-deepsort/data
mv ./yolov4.weights ./yolov4-deepsort/data
pip3 install -r requirements.txt
cd yolov4-deepsort
pip3 install -r requirements.txt
python3 save_model.py --model yolov4
cd ..
mv ./config.py ./yolov4-deepsort/core

