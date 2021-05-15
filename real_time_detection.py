import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

cascade=cv2.CascadeClassifier(r'C:\Users\bertug\AppData\Local\Programs\Python\Python37-32\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
model=load_model('Model/model.h5')

camera=cv2.VideoCapture(0)

while True:
    ret,frame = camera.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=cascade.detectMultiScale(gray,1.3,5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h+30, x:x + w+30]
        cv2.imshow("r",roi_gray)
        roi_gray=cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            probVal = np.amax(prediction)

            print("Emotion: ", label)
            print("Accuracy: ", probVal)

            cv2.putText(frame, "Emotion: " + str(label), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Accuracy: " + str(probVal), (5, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()