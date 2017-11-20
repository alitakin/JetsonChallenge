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
        self.lsd = lsd
        self.daemon = True
        self.model_name = 'mobilenet_0.75_224_0_model.h5'
        self.model = load_model(self.model_name, custom_objects={'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D})
        self.graph = tf.get_default_graph()
        self.start()

    def run(self):
        while True:
            face = self.lsd.getFace()

            # Smile detecting
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            if np.shape(face) != (224,224,3):
                face = face[:224, :224, ...]
            img_array = image.img_to_array(face)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.

            with self.graph.as_default():
                prediction = self.model.predict(img_array)
                self.lsd.setSmile(prediction)
