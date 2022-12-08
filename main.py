from keras.applications.inception_v3 import InceptionV3
import keras.utils as image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import cv2
import math
from glob import glob
from tqdm import tqdm
from google.colab.patches import cv2_imshow

# load the model
model = InceptionV3(weights='imagenet')

#Load video
cap = cv2.VideoCapture('sample1.mp4')   # capturing the video from the given path

frameRate = cap.get(5) #frame rate
!mkdir framess
x=1
count = 0
List =[]
top_result = None


userChoice = "sandbar"
inPath = ""


while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        # storing the frames in a new folder named train_1
        filename ='framess/' + str(frameId) +"_frame%d.jpg" % count;count+=1
        cv2.imwrite(filename, frame)
        # img_path = ''
        img = image.load_img(filename, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # predict class
        preds = model.predict(x)

        # get the top 5 classes
        top3 = decode_predictions(preds, top=3)[0]

        # print top 5 classes
        for result in top3:
            if userChoice == result[1]:
              if top_result is None:
                print("Assigning....")
                top_result = result
                
              print(result[2],top_result[2])
              if result[2] >= top_result[2]:
                top_result = result
                print(result)
                inPath = filename
                
print(top_result)
image = cv2.imread(inPath)
cv2_imshow(image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()

