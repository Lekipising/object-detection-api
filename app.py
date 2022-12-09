from detect import detector
import json
from flask_cors import CORS, cross_origin
from flask import (
    Flask, request
)

import logging

app = Flask(__name__)
cors = CORS(app)


gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers


if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)


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
    app.logger.info('Video file saved')
    base64 = detector(userInput, "video.mp4")
    app.logger.info('Video file processed')
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
