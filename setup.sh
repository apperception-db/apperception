#!/bin/bash
git clone https://github.com/theAIGuysCode/yolov4-deepsort.git
mv ./yolov4-tiny.weights ./yolov4-deepsort/data
mv ./yolov4.weights ./yolov4-deepsort/data
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
mv checkpoint/ ./yolov4-deepsort
pip install -r requirements.txt
cd yolov4-deepsort
python3 save_model.py --model yolov4
cd ..
mv ./config.py ./yolov4-deepsort/core
mkdir output
