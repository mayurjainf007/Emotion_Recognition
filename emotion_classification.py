import numpy as np
import cv2
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

model = model_from_json(open("model.json", "r").read())
model.load_weights('model_weights.h5') # trained model_weight if you want to use the recent one change the name...
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
import copy

while True:
    res, frame = cap.read()
    img = copy.deepcopy(frame)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    textX = int(150)
    textY = int(250)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:

        img = cv2.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness =  2)
        roi_gray = gray[y-5:y+h+5,x-5:x+w+5]
        roi_gray=cv2.resize(roi_gray,(48,48))
        image_pixels = img_to_array(roi_gray)
        image_pixels = np.expand_dims(image_pixels, axis = 0)
        image_pixels /= 255
        predictions = model.predict(image_pixels)
        max_index = np.argmax(predictions[0])
        emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        emotion_prediction = emotion_detection[max_index]
        cv2.putText(img, "{}>{}".format(emotion_prediction,str(np.round(np.max(predictions[0])*100,1))+ "%"), (x,y-5), FONT,0.5, (10, 10, 255),2)

    cv2.imshow("frame", img)
    key = cv2.waitKey(1) & 0xFF
    if key== ord('q'):
        break

cap.release()
cv2.destroyAllWindows()