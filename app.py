import json
from flask_cors import CORS, cross_origin
from flask import (
    Flask, request
)

from detect import detector

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
    base64 = detector(userInput, "video.mp4")
    if base64 is False:
        return json.dumps({'base64': "No match found"})
    return json.dumps({'base64': base64.decode('utf-8')})


@app.route('/')
@cross_origin()
def home():
    return """<div>
    <h1>API is up and running!</h1>
    <p>Send a POST request to /predict with a video file to get the predicted class</p>
    </div>"""


app.run()
