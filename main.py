from keras.applications.inception_v3 import InceptionV3
# import keras.utils as image
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import cv2
import math
import base64


def detector(userInput, videoPath):
    # load the model
    model = InceptionV3(weights='imagenet')

    # Load video
    # capturing the video from the given path
    cap = cv2.VideoCapture(videoPath)

    frameRate = cap.get(5)  # frame rate

    x = 1
    count = 0
    top_result = None

    userChoice = userInput
    inPath = ""

    while (cap.isOpened()):
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if (ret is not True):
            break
        if (frameId % math.floor(frameRate) == 0):
            # storing the frames in a new folder named train_1
            filename = 'framess/' + str(frameId) + "_frame%d.jpg" % count
            count += 1
            cv2.imwrite(filename, frame)
            # img_path = ''
            img = image.load_img(filename, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # predict class
            preds = model.predict(x)

            top3 = decode_predictions(preds, top=3)[0]

            for result in top3:
                if userChoice == result[1]:
                    if top_result is None:
                        top_result = result

                    if result[2] >= top_result[2]:
                        top_result = result
                        inPath = filename

    # get jpg image at file path and convert to base64
    with open(inPath, "rb") as img_file:
        my_string = base64.b64encode(img_file.read())
    return my_string
