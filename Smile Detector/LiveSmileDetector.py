import cv2
import RecognitionThread
import numpy as np


class LiveSmileDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.ret, tmp_img = self.cap.read()
        self.face_in_frame = [0, 0, 224, 224, tmp_img]
        self.prediction_label = 'Not smile'
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0

    def get_face(self):
        return self.face_in_frame

    def set_smile(self, prediction, x, y, w, h):
        self.prediction_label = 'Smile' if prediction.item(1) >= prediction.item(0) else 'Not smile'
        self.x = x
        self.y = y
        self.w = w
        self.h = h


lsd = LiveSmileDetector()
RT_instance = RecognitionThread.RT(lsd)

while True:
    # Capture frame-by-frame
    ret, frame = lsd.cap.read()

    # Changing frame's color order
    colored_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Detecting faces in the frame
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    # Drawing face rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(colored_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Cropping the face out of the frame
        detected_face = colored_frame[y:y + h, x:x + w, ...]

        # Updating the frame and face coordinates in it
        lsd.face_in_frame = [x, y, w, h, colored_frame]

        # Display the resulting prediction
        cv2.putText(colored_frame, lsd.prediction_label, (lsd.x + lsd.w // 4, lsd.y + lsd.h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)

    # Display images from the camera
    cv2.imshow('cropped_frame', colored_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
lsd.cap.release()
cv2.destroyAllWindows()
