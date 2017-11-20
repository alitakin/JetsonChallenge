import cv2
import RecognitionThread as RT
import copy


class Live_smile_detector:
    prediction_label = 'Not smile'

    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
        self.ret, self.cropped_frame = self.cap.read()
        self.prediction_label = 'Not smile'

    def getFrame(self):
        return copy.deepcopy(self.cropped_frame)

    def setSmile(self, prediction):
        self.prediction_label = 'Smile' if prediction.item(1) >= prediction.item(0) else 'Not smile'


LSD = Live_smile_detector()
RT_instance = RT.RT(LSD)

while True:
    # Capture cropped_frame-by-cropped_frame
    ret, frame = LSD.cap.read()

    # Our operations on the cropped_frame come here
    color = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    crop_img = color[8:232, 48:272]  # Crop from x, y, w, h -> 100, 200, 300, 400
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(crop_img, 1.3, 5)

    LSD.cropped_frame = crop_img

    # Detecting and drawing face rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(crop_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # detected_face = crop_img[y:y + h, x:x + w]

    # Display the resulting cropped_frame and printing prediction
    cv2.putText(crop_img, LSD.prediction_label, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, thickness=2)
    cv2.imshow('cropped_frame', crop_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
LSD.cap.release()
cv2.destroyAllWindows()
