from imageai.Detection import VideoObjectDetection
import json
from flask_cors import CORS, cross_origin
from flask import (
    Flask,
    request,
    make_response
)
import os


app = Flask(__name__)
cors = CORS(app)


@app.route('/detect', methods=['POST', 'OPTIONS'])
@cross_origin()
def predict():
    execution_path = os.getcwd()
    detector = VideoObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(os.path.join(
    execution_path, "model.h5"))
    detector.loadModel()
    # get form data with key 'file' from the request using request.form.get('file')
    videoFile = request.files['file']
    videoFile.save('video.mp4')
    # video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "video.mp4"),
                                                    # output_file_path=os.path.join(execution_path, "traffic_detected"), frames_per_second=20, log_progress=True)

    sampleMessage = "success"
    return json.dumps({'message': sampleMessage})


@app.route('/')
@cross_origin()
def home():
    return """<div>
    <h1>API is up and running!</h1>
    <p>Send a POST request to /predict with a video file to get the predicted class</p>
    </div>"""
