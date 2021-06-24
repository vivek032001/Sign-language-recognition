import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('slmod1.h5')
letters = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    top, right, bottom, left = 75, 350, 300, 590
    roi = frame[top:bottom, right:left]
    roi = cv2.flip(roi,1)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    gray = cv2.resize(gray, (28, 28))
    gray = gray.astype('float') / 255.0
    gray = tf.keras.preprocessing.image.img_to_array(gray)
    gray = np.expand_dims(gray, axis = 0)
    pred = model.predict(gray)
    idx = np.argmax(pred)
    sl = letters[idx]
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.putText(frame, sl, (0, 130), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 2)
    cv2.imshow('SL Recognizer', frame)
    key = cv2.waitKey(1)
    if(key == 27):
        break
cv2.destroyAllWindows()
cap.release()
    