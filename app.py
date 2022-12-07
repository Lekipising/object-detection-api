from imageai.Detection import VideoObjectDetection
import json
from flask_cors import CORS, cross_origin
from flask import (
    Flask,
    request,
    make_response
)
import os
execution_path = os.getcwd()
detector = VideoObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(os.path.join(
    execution_path, "model.h5"))
detector.loadModel()

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/predict', methods=['POST', 'OPTIONS'])
@cross_origin()
def predict():
    if request.method == "OPTIONS": # CORS preflight
        return _build_cors_preflight_response()
    # this method receives form data from the client which is video file
    # and returns the predicted class



    # get the file from the request
    # file = request.files['file']
    # save the file to disk
    # file.save('video.mp4')
    # video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "sample1.mp4"),
    #                                              output_file_path=os.path.join(execution_path, "traffic_detected"), frames_per_second=20, log_progress=True)
    # print(video_path)
    # send {message: 'success'}
    return json.dumps({'message': 'success'})


# defalut route


@app.route('/')
@cross_origin()
def home():
    return """<div>
    <h1>API is up and running!</h1>
    <p>Send a POST request to /predict with a video file to get the predicted class</p>
    </div>"""


def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response