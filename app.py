from imageai.Detection import VideoObjectDetection
import json
from flask import (
    Flask,
    request
)
import os
execution_path = os.getcwd()
detector = VideoObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(os.path.join(
    execution_path, "model.h5"))
detector.loadModel()

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # this method receives form data from the client which is video file
    # and returns the predicted class

    # get the file from the request
    file = request.files['file']
    # save the file to disk
    file.save('video.mp4')
    video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "sample1.mp4"),
                                                 output_file_path=os.path.join(execution_path, "traffic_detected"), frames_per_second=20, log_progress=True)
    print(video_path)
    return json.dumps(video_path)


# defalut route


@app.route('/')
def home():
    return """<div>
    <h1>API is up and running!</h1>
    <p>Send a POST request to /predict with a video file to get the predicted class</p>
    </div>"""

