# Object Detection using Google Inception V3 deep learning model

This is a flask API that uses Google Inception V3 deep learning model to detect objects in videos. The file [`detect.py](/detect.py) contains the logic for object detection.

The flask endpoint `/detect` accepts a video file amd returns a frame with the detected objects.

The API is deployed on Render and can be accessed on https://object-detect.onrender.com

The API is consumed by a frontend that is used in the following way:

1. Upload a video file
2. Type an object in the video
3. Click on `Process Video` button
4. The video is processed and the frame with the object is displayed

The frontend is deployed on Render and can be accessed on [https://detect-object-app.vercel.app](https://detect-object-app.vercel.app/)

The following GIF shows the frontend in action:

![Object Detection](/gif.gif)

## To play with the frontend, visit [https://detect-object-app.vercel.app](https://detect-object-app.vercel.app/)


## Project Contributors
This project was developed by:
- Adio Roheem
- Modester Mwangi
- Brenda Gilisho
- Liplan Lekipising