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
        # Loading pre trained model
        self.model_name = 'mobilenet_0.75_224_0_model.h5'
        self.model = load_model(self.model_name, custom_objects={'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D})
        # Initialising the variables
        self.lsd = lsd
        # Starting the thread
        self.start()

    def run(self):
        # Retrieving new frames form the main thread in an infinite loop
        while True:
            # Getting the new frame and coordinates of face in it
            face_list_and_frame = self.lsd.get_face()
            # Getting face list
            faces = face_list_and_frame[0]
            # Getting frame and changing its color order
            frame = cv2.cvtColor(face_list_and_frame[1], cv2.COLOR_BGR2RGB)
            # Adding zero padding around the image
            frame_size = np.shape(frame)
            frame = cv2.copyMakeBorder(frame, int(frame_size[0] / 2), int(frame_size[0] / 2), int(frame_size[1] / 2),
                                       int(frame_size[1] / 2), cv2.BORDER_CONSTANT)
            result = []
            for (x, y, w, h) in faces:
                # Updating coordinates of top-left corner of face according to the added zero padding
                padded_x = x + int(frame_size[1] / 2)
                padded_y = y + int(frame_size[0] / 2)

                # Preprocessing image of the detected face to have size 224x224
                face_rec_len = min(w, h)
                if face_rec_len <= 224:
                    tmp_len = (224 - face_rec_len) // 2
                    detected_face = frame[
                                    padded_y - tmp_len:padded_y + h + tmp_len + (face_rec_len % 2),
                                    padded_x - tmp_len:padded_x + w + tmp_len + (face_rec_len % 2), ...]
                    res = self.smile_detector(detected_face)
                elif face_rec_len > 224:
                    detected_face = cv2.resize(
                        frame[padded_y - int(face_rec_len / 2):padded_y + int(1.5 * face_rec_len),
                        padded_x - int(face_rec_len):padded_x + int(1.5 * face_rec_len), ...], (224, 224))
                    res = self.smile_detector(detected_face)

                # Appending result of prediction and its coordinates to the list
                result.append([res, x, y, w, h])

            # Updating smile predictions with their corresponding coordinates
            self.lsd.set_smile(result)

    def smile_detector(self, detected_face):
        # Preprocess image on detected face
        img_array = image.img_to_array(detected_face)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        # Smile detecting and passing the result to the main thread
        with self.graph.as_default():
            return self.model.predict(img_array)
