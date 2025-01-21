import cv2
import os
import datetime as dt
import numpy as np
import tensorflow as tf
import torch
from keras.models import load_model

# oneDNN custom operations are on.
# You may see slightly different numerical results due to floating-point round-off errors from different computation orders.
# To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import datetime as dt
import numpy as np
import tensorflow as tf

print("Tensorflow version:", tf.__version__)

import torch
from keras.models import load_model

print("Script loaded. Import complete")

OBJECT_DETECTION_MODEL_PATH = "models/best.pt"
PREDICTOR_MODEL_PATH = "models/keras_model.h5"
CLASS_NAMES = {0: "No Seatbelt worn", 1: "Seatbelt Worn"}

# Configuration settings
THRESHOLD_SCORE = 0.99  # Configurable threshold for confidence scores
SKIP_FRAMES = 1         # Process every Nth frame
MAX_FRAME_RECORD = 500  # Maximum frames to process
INPUT_VIDEO = 'sample/test_2.mp4'
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "test_result_" + dt.datetime.now().strftime("%Y%m%d%H%M%S") + ".mp4"
)
# Colors for drawing bounding boxes
CLASS_NAMES = {0: "No Seatbelt Worn", 1: "Seatbelt Worn"}
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)

# Function to predict the class of a given image
def prediction_func(img):
    # Resize the image
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    # Normalize the image
    img = (img / 127.5) - 1
    # Expand the image
    img = tf.expand_dims(img, axis=0)
    # Predict the class
    pred = predictor.predict(img)
    # Get the index of the class with the highest score
    index = np.argmax(pred)
    # Get the name of the class
    class_name = CLASS_NAMES[index]
    # Get the confidence score of the class
    confidence_score = pred[0][index]
    return class_name, confidence_score

# Load the predictor model
try:
    predictor = load_model("models/keras_model.h5", compile=False)
    print("Predictor model loaded successfully.")
except Exception as e:
    print(f"Error loading predictor model: {e}")
    exit()

# Ultralytics object detection model : https://docs.ultralytics.com/yolov5/
try:
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", path="models/best.pt", force_reload=True
    )
    print("Object detection model loaded successfully.")
except Exception as e:
    print(f"Error loading object detection model: {e}")
    exit()

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load video
cap = cv2.VideoCapture(INPUT_VIDEO)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
size = (frame_width, frame_height)
writer = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'MP4V'), 30, size)

print(f"Processing video: {INPUT_VIDEO}")

# Function to draw a bounding box on an image
def draw_bounding_box(img, x1, y1, x2, y2, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

# Function to draw text on an image
def draw_text(img, x, y, text, color):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# Function to classify the driver in an image
def classify_driver(img):
    y_pred, score = prediction_func(img)
    # If the class is 0 (no seatbelt worn), draw red bounding box
    if y_pred == CLASS_NAMES[0]:
        draw_color = COLOR_RED
    # If the class is 1 (seatbelt worn), draw green bounding box
    elif y_pred == CLASS_NAMES[1]:
        draw_color = COLOR_GREEN
    return y_pred, score, draw_color

# Function to process a frame from the video
def process_frame(frame):
    # Convert the frame to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Run the object detection model
    results = model(img)
    # Get the bounding boxes
    boxes = results.xyxy[0]
    # Convert the bounding boxes to numpy
    boxes = boxes.cpu()
    # Iterate over the bounding boxes
    for j in boxes:
        # Get the coordinates of the bounding box
        x1, y1, x2, y2, score, y_pred = j.numpy()
        # Convert the coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Crop the image to the bounding box
        img_crop = img[y1:y2, x1:x2]

        # Classify the driver in the cropped image
        y_pred, score, draw_color = classify_driver(img_crop)

        # If the score is above the threshold, draw the bounding box
        if score >= THRESHOLD_SCORE:
            draw_bounding_box(frame, x1, y1, x2, y2, draw_color)
            draw_text(frame, x1 - 10, y1 - 10, f"{y_pred} {str(score)[:4]}", draw_color)
    return frame

# Initialize the frame count
frame_count = 0
# While the video is not finished
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If the frame is read successfully
    if ret:
        # Increment the frame count
        frame_count += 1

        # If the frame count is a multiple of the skip frames, process the frame
        if frame_count % SKIP_FRAMES == 0:
            frame = process_frame(frame)

        # Show the frame
        cv2.imshow("Video feed", frame)

        # If the frame count is above the maximum frame record, break
        if frame_count > MAX_FRAME_RECORD:
            break
    else:
        break

    # If the user presses q, break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
# writer.release()

# Destroy all the windows
cv2.destroyAllWindows()

print("Script run complete. Results saved to :", OUTPUT_FILE)
