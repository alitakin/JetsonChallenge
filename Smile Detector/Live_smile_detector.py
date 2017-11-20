import cv2
import RecognitionThread as RT
import copy
import numpy as np


class Live_smile_detector:
    prediction_label = 'Not smile'

    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.ret, self.img = self.cap.read()
        self.prediction_label = 'Not smile'

    def getFace(self):
        return self.img

    def setSmile(self, prediction):
        self.prediction_label = 'Smile' if prediction.item(1) >= prediction.item(0) else 'Not smile'


LSD = Live_smile_detector()
RT_instance = RT.RT(LSD)

while True:
    # Capture frame-by-frame
    ret, frame = LSD.cap.read()

    # Our operations on the frame come here
    frame_shape = np.shape(frame)
    frame = cv2.copyMakeBorder(frame, frame_shape[0] // 2, frame_shape[0] // 2, frame_shape[1] // 2,
                               frame_shape[1] // 2, cv2.BORDER_CONSTANT)
    colored_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    # Detecting and drawing face rectangles
    for (x, y, w, h) in faces:
        face_rec_len = w
        if w < h:
            face_rec_len = h

        cv2.rectangle(colored_frame, (x, y), (x + face_rec_len, y + face_rec_len), (255, 0, 0), 2)

        if face_rec_len < 224:
            tmp_len = (224 - face_rec_len) // 2
            detected_face = colored_frame[
                            y - tmp_len:y + h + tmp_len + (face_rec_len % 2),
                            x - tmp_len:x + w + tmp_len + (face_rec_len % 2), ...]
        else:
            extra_border = 40
            detected_face = cv2.resize(colored_frame[y - extra_border:y + face_rec_len + extra_border,
                                       x - extra_border:x + face_rec_len + extra_border, ...], (224, 224))

        LSD.img = detected_face

        # Display the resulting prediction
        cv2.putText(colored_frame, LSD.prediction_label, (x + face_rec_len//3, y + face_rec_len - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), thickness=2)

    # Display images from the camera
    cv2.imshow('cropped_frame', colored_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
LSD.cap.release()
cv2.destroyAllWindows()
