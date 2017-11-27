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
            faces_and_frame = self.lsd.get_face()
            # Getting face list
            faces = faces_and_frame[0]
            # Getting frame and changing its color order
            frame = cv2.cvtColor(faces_and_frame[1], cv2.COLOR_BGR2RGB)
            # Adding zero padding around the image
            frame_size = np.shape(frame)
            frame = cv2.copyMakeBorder(frame, int(frame_size[0]), int(frame_size[0]), int(frame_size[1]),
                                       int(frame_size[1]), cv2.BORDER_CONSTANT)
            faces_prediction_results = []
            for (x, y, w, h) in faces:
                # Updating coordinates of top-left corner of face according to the added zero padding
                padded_x = x + int(frame_size[1] / 2)
                padded_y = y + int(frame_size[0] / 2)

                # Preprocessing image of the detected face by adding padding to it and resizing it
                face_rec_len = min(w, h)
                if face_rec_len <= 185:
                    print(0)
                    img_padding = (224 - face_rec_len) // 2
                    detected_face = frame[
                                    padded_y - img_padding:padded_y + h + img_padding + (face_rec_len % 2),
                                    padded_x - img_padding:padded_x + w + img_padding + (face_rec_len % 2), ...]
                    prediction_result = self.smile_detector(detected_face)
                elif face_rec_len < 224:
                    print(1)
                    img_padding = 40
                    detected_face = frame[padded_y - img_padding:padded_y + h + img_padding,
                                    padded_x - img_padding:padded_x + w + img_padding, ...]
                    detected_face = cv2.resize(detected_face, (224, 224))
                    prediction_result = self.smile_detector(detected_face)
                else:
                    print(2)
                    detected_face = cv2.resize(
                        frame[padded_y - int(face_rec_len / 2):padded_y + int(1.5 * face_rec_len),
                        padded_x - int(face_rec_len):padded_x + int(1.5 * face_rec_len), ...], (224, 224))
                    prediction_result = self.smile_detector(detected_face)

                # Appending result of prediction and its coordinates to the list
                faces_prediction_results.append([prediction_result, x, y, w, h])

            # Updating smile predictions with their corresponding coordinates
            self.lsd.set_smile(faces_prediction_results)

    def smile_detector(self, detected_face):
        # Preprocess image od detected face
        img_array = image.img_to_array(detected_face)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        # Smile detecting and passing the result to the main thread
        with self.graph.as_default():
            return self.model.predict(img_array)
