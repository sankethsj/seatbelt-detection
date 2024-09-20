# Seatbelt Detection using YOLOv5 and Keras Models

This project aims to detect whether a person is wearing a seatbelt using a combination of object detection with YOLOv5 models and classification with the keras model. The YOLOv5 model is used to detect objects such as people and car seats, while the Keras model is used to predict whether the detected person is wearing a seatbelt or not, with a confidence score of 0.99.

## Prerequisites

Before running the project, ensure that you have Python installed (preferably version 3.8 or higher). You'll also need to install required dependencies and set up a virtual environment for better package management.

1. Set Up a Virtual Environment (Optional but Recommended)
To create a virtual environment, follow these steps:

```bash
# Install virtualenv if you don't have it
pip install virtualenv

# Create a virtual environment named 'venv'
virtualenv venv

# Activate the virtual environment (Windows)
.\venv\Scripts\activate

# Activate the virtual environment (MacOS/Linux)
source venv/bin/activate
```

2. Install Required Packages

Once the virtual environment is activated, install the necessary packages listed in requirements.txt:

```bash
pip install -r requirements.txt
```

## Running the Project

To run the seatbelt detection script, execute the following command:

```bash
python main.py
```

## Notes

- On the initial run, the script may take extra time as it will download the YOLOv5 model.
- The confidence score threshold for seatbelt detection is set to 0.99, meaning the system is highly accurate in predicting whether a person is wearing a seatbelt.

## Workflow Overview

1. ### YOLOv5 Model (Object Detection)

- The YOLOv5 model is used to detect objects in the input video, such as the person and car seat.
- The object detection part helps isolate the regions of interest for further analysis.

2. ### Keras Model (Seatbelt Detection)

- Once the person and seat are identified, the Keras model is applied to classify whether the person is wearing a seatbelt.
- The prediction is made with a confidence score of 0.99, ensuring high accuracy in the detection process.

3. ### Video Processing

- The input video is processed frame by frame, and detection results are overlaid on the video.
- The output video highlights the detected objects (people, seats) and displays whether the seatbelt is worn.

## Input Video feed

You can test the model on an input video file. Check the code and uncomment necessary lines for using video instead of camera feed.

An example video file is provided in the project:
[Test video file](/sample/test_2.mp4)

## Output Video feed

Optionally, After running the detection, the output will be a video with predictions (seatbelt detection overlay). Check the code and uncomment necessary lines for saving the video output to a file.

A sample output is provided:
[Predicted video](/output/test_result_20230610153232.mp4)

## Pre-trained model source

The YOLOv5 model used for seatbelt detection is pre-trained and available on Kaggle:
https://www.kaggle.com/datasets/sachinmlwala/seatbelt3

## Troubleshooting

If you encounter any issues, ensure that:

- Your virtual environment is activated before installing packages or running the script.
- The required packages in requirements.txt are installed correctly.
- You have an active internet connection during the first run, as the YOLOv5 model will be downloaded.

## License

This project is licensed under the MIT License.

