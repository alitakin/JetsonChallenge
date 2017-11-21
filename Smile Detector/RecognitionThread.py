import numpy as np
import threading
import cv2
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet import relu6, DepthwiseConv2D
import tensorflow as tf


class RT(threading.Thread):
    def __init__(self, lsd):
        threading.Thread.__init__(self)
        self.graph = tf.get_default_graph()
        self.daemon = True
        #  Loading model
        self.model_name = 'mobilenet_0.75_224_0_model.h5'
        self.model = load_model(self.model_name, custom_objects={'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D})
        # Initialising the variables
        self.lsd = lsd
        self.w = 0
        self.h = 0
        # Starting the thread
        self.start()

    def run(self):
        # Our operations on the frame come here
        while True:
            # Getting the new frame and coordinates of face in it
            face_in_frame = self.lsd.get_face()
            # Changing frame color order
            frame = cv2.cvtColor(face_in_frame[4], cv2.COLOR_BGR2RGB)
            # Adding zero padding around the image
            frame_size = np.shape(frame)
            frame = cv2.copyMakeBorder(frame, int(frame_size[0] / 2), int(frame_size[0] / 2), int(frame_size[1] / 2),
                                       int(frame_size[1] / 2), cv2.BORDER_CONSTANT)

            # Updating coordinates according to the added zero padding
            x = face_in_frame[0] + int(frame_size[1] / 2)
            y = face_in_frame[1] + int(frame_size[0] / 2)
            self.w = face_in_frame[2]
            self.h = face_in_frame[3]

            # Preprocessing image of the detected face to have size 224x224
            face_rec_len = min(self.w, self.h)
            if face_rec_len <= 224:
                tmp_len = (224 - face_rec_len) // 2
                detected_face = frame[
                                y - tmp_len:y + self.h + tmp_len + (face_rec_len % 2),
                                x - tmp_len:x + self.w + tmp_len + (face_rec_len % 2), ...]
                self.smile_detector(detected_face, x - int(frame_size[1] / 2), y - int(frame_size[0] / 2))
            elif face_rec_len > 224:
                detected_face = cv2.resize(frame[y - int(face_rec_len/2):y + int(1.5 * face_rec_len),
                                           x - int(face_rec_len):x + int(1.5 * face_rec_len), ...], (224, 224))
                self.smile_detector(detected_face, x - int(frame_size[1] / 2), y - int(frame_size[0] / 2))

    def smile_detector(self, detected_face, x, y):
        # Preprocess image od detected face
        img_array = image.img_to_array(detected_face)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        # Smile detecting and passing the result to the main thread
        with self.graph.as_default():
            prediction = self.model.predict(img_array)
            self.lsd.set_smile(prediction, x, y, self.w, self.h)
