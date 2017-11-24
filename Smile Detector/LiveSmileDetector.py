import cv2
import RecognitionThread


class LiveSmileDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.ret, tmp_img = self.cap.read()
        self.face_list_and_frame = [[], tmp_img]
        self.result = [(1, 0), 0, 0, 0, 0]

    def get_face(self):
        return self.face_list_and_frame

    def set_smile(self, result):
        self.result = result


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

    # Updating the frame and face coordinates in it
    lsd.face_list_and_frame = [faces, colored_frame]

    for face in faces:
        # Drawing rectangles around each detected face
        cv2.rectangle(colored_frame, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (255, 0, 0), 2)

    for prediction_coordinate in lsd.result:
        # Display the resulting prediction
        prediction_label = 'Smile' if prediction_coordinate[0].item(1) >= prediction_coordinate[0].item(
            0) else 'Not smile'
        cv2.putText(colored_frame, prediction_label,
                    (prediction_coordinate[1] + prediction_coordinate[3] // 4,
                     prediction_coordinate[2] + prediction_coordinate[4] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)

    # Display images from the camera
    cv2.imshow('cropped_frame', colored_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
lsd.cap.release()
cv2.destroyAllWindows()
