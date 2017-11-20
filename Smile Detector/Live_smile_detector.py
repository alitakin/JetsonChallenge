import cv2
import RecognitionThread as RT
import copy
import numpy as np


class Live_smile_detector:
    prediction_label = 'Not smile'

    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
        self.ret, self.img = self.cap.read()
        # cv2.resize(self.img, (224,224))
        self.prediction_label = 'Not smile'

    def getFace(self):
        return copy.deepcopy(self.img)

    def setSmile(self, prediction):
        self.prediction_label = 'Smile' if prediction.item(1) >= prediction.item(0) else 'Not smile'


LSD = Live_smile_detector()
RT_instance = RT.RT(LSD)

while True:
    # Capture cropped_frame-by-cropped_frame
    ret, frame = LSD.cap.read()
    # Our operations on the cropped_frame come here
    colored_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    frame_shape = np.shape(colored_frame)
    frame = cv2.copyMakeBorder(frame, 0, frame_shape[0], 0, frame_shape[1], cv2.BORDER_REFLECT_101)
    # crop_img = colored_frame[8:232, 48:272]  # Crop from x, y, w, h -> 100, 200, 300, 400
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)


    # Detecting and drawing face rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(colored_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        detected_face = colored_frame[y:y + h, x:x + w]

        face_shape = np.shape(detected_face)
        if(face_shape[0]<224):
            detected_face = colored_frame[y-(224-face_shape[0])//2:y+h+(224-face_shape[0]//2)+(face_shape[0]%2),
                                          x-(224-face_shape[1])//2:x+w+(224-face_shape[1]//2)+(face_shape[1]%2)]
        elif(face_shape[0]>224):
            # img_scaled = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)
            cv2.resize(detected_face, (224, 224), detected_face)

        print(np.shape(detected_face))
        LSD.img = detected_face


    # Display the resulting cropped_frame and printing prediction
    cv2.putText(colored_frame, LSD.prediction_label, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, thickness=2)
    cv2.imshow('cropped_frame', colored_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
LSD.cap.release()
cv2.destroyAllWindows()
