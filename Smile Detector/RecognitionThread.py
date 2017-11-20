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
        while (True):
            cropped_frame = self.lsd.getFrame()

            # Smile detecting
            cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            cropped_frame = cropped_frame[:224, :224, ...]
            img_array = image.img_to_array(cropped_frame)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.

            with self.graph.as_default():
                prediction = self.model.predict(img_array)
                self.lsd.setSmile(prediction)
