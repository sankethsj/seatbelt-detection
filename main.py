import cv2
import os
import datetime as dt
import numpy as np
import tensorflow as tf
import torch
from keras.models import load_model

print("Script loaded. Import complete")

OBJECT_DETECTION_MODEL_PATH = "models/best.pt"
PREDICTOR_MODEL_PATH = "models/keras_model.h5"
CLASS_NAMES = {0: 'No Seatbelt worn', 1: 'Seatbelt Worn'}

THRESHOLD_SCORE = 0.99

SKIP_FRAMES = 1 # skips every 2 frames
MAX_FRAME_RECORD = 500
INPUT_VIDEO = 0 #'sample/test_2.mp4'
OUTPUT_FILE = 'output/test_result_'+ dt.datetime.strftime(dt.datetime.now(), "%Y%m%d%H%M%S") +'.mp4'

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)

def prediction_func(img):
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = (img/127.5)-1
    img = tf.expand_dims(img, axis=0)
    pred = predictor.predict(img)
    index = np.argmax(pred)
    class_name = CLASS_NAMES[index]
    confidence_score = pred[0][index]
    return class_name, confidence_score

predictor = load_model(PREDICTOR_MODEL_PATH, compile=False)
print("Predictor loaded")

# Ultralytics object detection model : https://docs.ultralytics.com/yolov5/
model = torch.hub.load("ultralytics/yolov5", "custom", path=OBJECT_DETECTION_MODEL_PATH, force_reload=False)

cap = cv2.VideoCapture(INPUT_VIDEO)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

os.makedirs(OUTPUT_FILE.rsplit("/", 1)[0], exist_ok=True)
# writer = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'MP4V'), 30, size)

print("Analyzing input video...")

frame_count = 0
while True:

    ret, img = cap.read()

    if ret:

        frame_count += 1

        if frame_count % SKIP_FRAMES == 0:
            # read each frame
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # detects driver
            results = model(img)

            boxes = results.xyxy[0]
            boxes = boxes.cpu()
            for j in boxes:
                x1, y1, x2, y2, score, y_pred = j.numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                img_crop = img[y1:y2, x1:x2]

                # predictor to classify whether driver is wearing seatbelt or not
                y_pred, score = prediction_func(img_crop)

                if y_pred == CLASS_NAMES[0]:
                    draw_color = COLOR_RED
                elif y_pred == CLASS_NAMES[1]:
                    draw_color = COLOR_GREEN

                if score >= THRESHOLD_SCORE:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f'{y_pred} {str(score)[:4]}', (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color, 2)
                
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # writer.write(img)
            cv2.imshow('Video feed', img)

            # if frame_count > MAX_FRAME_RECORD:
            #     break
    else:
        break

    # press the 'q' button to stop the video feed 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    

cap.release()
# writer.release()

# Destroy all the windows
cv2.destroyAllWindows()

print("Script run complete. Results saved to :", OUTPUT_FILE)
