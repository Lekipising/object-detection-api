from detect import detector
import json
from flask_cors import CORS, cross_origin
from flask import (
    Flask, request
)

app = Flask(__name__)
cors = CORS(app)


@app.route('/detect', methods=['POST', 'OPTIONS'])
@cross_origin()
def predict():
    videoFile = request.files['file']
    userInput = request.files['json']
    # get json from the file and convert to dict
    userInput = json.loads(userInput.read())
    userInput = userInput['json']
    videoFile.save('video.mp4')
    # save the video file to the server using os
    print("Received request....")
    base64 = detector(userInput, "video.mp4")
    print("Done!")
    if base64['base64'] is False:
        return json.dumps({'base64': "No match found", 'classes': base64["classes"]})
    return json.dumps({'base64': base64["base64"].decode('utf-8')})


@app.route('/')
@cross_origin()
def home():
    return """<div>
    <h1>API is up and running!</h1>
    <p>Send a POST request to /predict with a video file to get the predicted class</p>
    </div>"""
