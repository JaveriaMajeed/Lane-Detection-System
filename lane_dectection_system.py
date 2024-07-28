# importing libraries
import cv2
import numpy as np
import os
import tensorflow as tf
from moviepy.editor import VideoFileClip

# import lane detection and lane classification model
from tensorflow.keras.models import load_model
model_dir = "/model"
model_detection = load_model(os.path.join(model_dir, 'laneDetection.h5'))
model_classification = load_model(os.path.join(model_dir, 'laneClassification.h5'))


# Assuming 'model_detection' and 'model_classification' are pre-loaded models

class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

lanes = Lanes()

def road_lines(image):
    """ Applies lane detection on the image and returns the modified image """
    image_shape = image.shape
    small_img = cv2.resize(image, (160, 80))
    small_img = np.array(small_img)[None, :, :, :]
    prediction = model_detection.predict(small_img)[0] * 255
    lanes.recent_fit.append(prediction)
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))
    lane_image = cv2.resize(lane_drawn, (image_shape[1], image_shape[0])).astype(image.dtype)
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)
    return result

def preprocess_frame(frame):
    """ Preprocesses the frame for classification model """
    resize = tf.image.resize(frame, (80,160))
    yhat = model_classification.predict(np.expand_dims(resize/255, 0))
    yhat = np.round(yhat)
    return yhat

def preprocess_frame(frame):
    """ Preprocesses the frame for classification model """
    resize = tf.image.resize(frame, (80,160))
    yhat = model_classification.predict(np.expand_dims(resize/255, 0))
    yhat = np.round(yhat)
    return yhat

def Prediction(predictions):
    if np.array_equal(predictions, np.array([[0, 1, 0]])):
        return "Double Solid Yellow Lane"
    elif np.array_equal(predictions, np.array([[1, 0, 0]])):
        return "Dashed White Lane"
    elif np.array_equal(predictions, np.array([[0, 0, 1]])):
        return "Solid White Lane"
    else:
        return "Unidentified"

def process_frame(frame):
    """ Processes a single frame for both lane detection and classification """
    # Lane detection
    lane_image = road_lines(frame)

    # Lane classification
    predictions = preprocess_frame(frame)
    lane_type = Prediction(predictions)

    # Display classification result on the image
    cv2.putText(lane_image, lane_type, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return lane_image

# Video processing
input_video_path = "/content/sample_data/MyDrive/MyDrive/lane_detection/testing/testing_video3.mp4"
output_video_path = "/content/sample_data/MyDrive/MyDrive/lane_detection/testing/1234.mp4"

clip = VideoFileClip(input_video_path)
processed_clip = clip.fl_image(process_frame)
processed_clip.write_videofile(output_video_path, audio=False)

