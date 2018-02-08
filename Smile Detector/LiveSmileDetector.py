import cv2
import RecognitionThread


class LiveSmileDetector:
    def __init__(self):
        # Select the desired camera id
        self.cap = cv2.VideoCapture(1)
        self.ret, tmp_img = self.cap.read()
        self.faces_and_frame = [[], tmp_img]
        self.result = [(1, 0), 0, 0, 0, 0]

    def get_face(self):
        return self.faces_and_frame

    def set_smile(self, result):
        self.result = result


lsd = LiveSmileDetector()
RT_instance = RecognitionThread.RT(lsd)

while True:
    # Capture original frame from camera
    ret, original_frame = lsd.cap.read()

    # Changing original_frame's color order
    frame = cv2.cvtColor(original_frame, cv2.COLOR_BGRA2BGR)

    # Shrink frame and detecting faces in the shrinked frame
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    shrink_ratio = 0.5
    new_w = int(frame.shape[1] * shrink_ratio)
    new_h = int(frame.shape[0] * shrink_ratio)
    shrinked_frame = cv2.resize(frame, (new_w, new_h))
    faces = face_cascade.detectMultiScale(shrinked_frame, 1.3, 5)

    # Updating the original_frame and face coordinates in it
    lsd.faces_and_frame = [faces, frame]

    for face in faces:
        # Drawing rectangles around each detected face
        face[0] = face[0] / shrink_ratio
        face[1] = face[1] / shrink_ratio
        face[2] = face[2] / shrink_ratio
        face[3] = face[3] / shrink_ratio
        cv2.rectangle(frame, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (255, 0, 0), 2)

    for prediction_coordinate in lsd.result:
        # Display the resulting prediction
        no_smile_prc = "{:.2f}".format(100 * prediction_coordinate[0].item(0))
        smile_prc = "{:.2f}".format(100 * prediction_coordinate[0].item(1))
        pred_label = 'Smile: %' + smile_prc \
            if prediction_coordinate[0].item(1) >= prediction_coordinate[0].item(0) \
            else 'No smile: %' + no_smile_prc
        cv2.putText(frame, pred_label,
                    (prediction_coordinate[1],
                     prediction_coordinate[2] + prediction_coordinate[4] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)


    # Display images from the camera
    cv2.imshow('cropped_frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
lsd.cap.release()
cv2.destroyAllWindows()