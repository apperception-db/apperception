import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),"../yolov4-deepsort"))
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
# from absl import app, flags, logging
# from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

from mono_depth_estimator import create_depth_frames

from collections import namedtuple
FLAGS = namedtuple('Flags', ['framework', 'weights', 'size', 'tiny',
					 'model', 'iou', 'score', 'dont_show', 'info', 'count'])\
		  (framework='tf',
		   weights=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../yolov4-deepsort/checkpoints/yolov4-416'),
		   size=416, tiny=True, model='yolov4',
		   iou=0.45, score=0.50, dont_show=True, info=False, count=False)

# flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
# flags.DEFINE_string('weights', './checkpoints/yolov4-416',
#                     'path to weights file')
# flags.DEFINE_integer('size', 416, 'resize images to')
# flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
# flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
# flags.DEFINE_float('iou', 0.45, 'iou threshold')
# flags.DEFINE_float('score', 0.50, 'score threshold')
# flags.DEFINE_boolean('dont_show', False, 'dont show video output')
# flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
# flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

def yolov4_deepsort_video_track(video_file, default_depth=True):
	# import json
	# with open('text_result.txt') as json_file:
	# 	formatted_result = json.load(json_file)
	# return formatted_result
	# Definition of the parameters
	max_cosine_distance = 0.4
	nn_budget = None
	nms_max_overlap = 1.0

	# initialize deep sort

	model_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
									 '../yolov4-deepsort/model_data/mars-small128.pb')
	encoder = gdet.create_box_encoder(model_filename, batch_size=1)
	# calculate cosine distance metric
	metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
	# initialize tracker
	tracker = Tracker(metric)

	# load configuration for object detector
	config = ConfigProto()
	config.gpu_options.allow_growth = True
	session = InteractiveSession(config=config)
	STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
	input_size = 416

	# load standard tensorflow saved model
	saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
	infer = saved_model_loaded.signatures['serving_default']

	formatted_result = {}
	cap = cv2.VideoCapture(video_file)
	frame_num = 0
	# while video is running
	while(cap.isOpened()):
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret == True:
			image = Image.fromarray(frame)
			
			frame_num +=1
			# print('Frame #: ', frame_num)
			frame_size = frame.shape[:2]
			image_data = cv2.resize(frame, (input_size, input_size))
			image_data = image_data / 255.
			image_data = image_data[np.newaxis, ...].astype(np.float32)
			start_time = time.time()


			batch_data = tf.constant(image_data)
			pred_bbox = infer(batch_data)
			for key, value in pred_bbox.items():
				boxes = value[:, :, 0:4]
				pred_conf = value[:, :, 4:]

			boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
				boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
				scores=tf.reshape(
					pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
				max_output_size_per_class=50,
				max_total_size=50,
				iou_threshold=FLAGS.iou,
				score_threshold=FLAGS.score
			)

			# convert data to numpy arrays and slice out unused elements
			num_objects = valid_detections.numpy()[0]
			bboxes = boxes.numpy()[0]
			bboxes = bboxes[0:int(num_objects)]
			scores = scores.numpy()[0]
			scores = scores[0:int(num_objects)]
			classes = classes.numpy()[0]
			classes = classes[0:int(num_objects)]

			# format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
			original_h, original_w, _ = frame.shape
			bboxes = utils.format_boxes(bboxes, original_h, original_w)

			# store all predictions in one parameter for simplicity when calling functions
			pred_bbox = [bboxes, scores, classes, num_objects]

			# read in all class names from config
			class_names = utils.read_class_names(cfg.YOLO.CLASSES)

			# by default allow all classes in .names file
			allowed_classes = list(class_names.values())

			# custom allowed classes (uncomment line below to customize tracker for only people)
			#allowed_classes = ['person']

			# loop through objects and use class index to get class name, allow only classes in allowed_classes list
			names = []
			deleted_indx = []
			for i in range(num_objects):
				class_indx = int(classes[i])
				class_name = class_names[class_indx]
				if class_name not in allowed_classes:
					deleted_indx.append(i)
				else:
					names.append(class_name)
			names = np.array(names)
			if FLAGS.count:
				cv2.putText(frame, "Objects being tracked: {}".format(len(names)), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
				print("Objects being tracked: {}".format(len(names)))
			# delete detections that are not in allowed_classes
			bboxes = np.delete(bboxes, deleted_indx, axis=0)
			scores = np.delete(scores, deleted_indx, axis=0)

			# encode yolo detections and feed to tracker
			features = encoder(frame, bboxes)
			detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

			#initialize color map
			cmap = plt.get_cmap('tab20b')
			colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

			# run non-maxima supression
			boxs = np.array([d.tlwh for d in detections])
			scores = np.array([d.confidence for d in detections])
			classes = np.array([d.class_name for d in detections])
			indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
			detections = [detections[i] for i in indices]

			# Call the tracker
			tracker.predict()
			tracker.update(detections)

			for track in tracker.tracks:
				if not track.is_confirmed() or track.time_since_update > 1:
					continue
				bbox = track.to_tlbr()
				class_name = track.get_class()
				item_id = class_name+"-"+str(track.track_id)
				# tl = [int(bbox[0]), int(bbox[1])]
				# br = [int(bbox[2]), int(bbox[3])]
				tl = [400, 235, 3]
				br = [650, 350, 2]
				# if not default_depth:
				# 	frame_depth = create_depth_frames(np.array([np.asarray(image)]))
				# 	try:
				# 		tl_depth = frame_depth[0][tl[1], tl[0]]
				# 		br_depth = frame_depth[0][br[1], br[0]]
				# 	except:
				# 		tl_depth = 1
				# 		br_depth = 1
				# else:
				# 	tl_depth = 1
				# 	br_depth = 1
				# tl.append(tl_depth)
				# br.append(br_depth)
				if item_id in formatted_result:
					formatted_result[item_id]["bboxes"].append([tl, br])
					formatted_result[item_id]["tracked_cnt"].append(frame_num)
				else:
					formatted_result[item_id]={"object_type": class_name,
											"bboxes":[[tl, br]],
											"tracked_cnt":[frame_num]}
		else:
			break
	cap.release()
	print("# of tracked items:", len(formatted_result))
	return formatted_result

