import numpy as np
import threading
import cv2
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet import relu6, DepthwiseConv2D


class RT(threading.Thread):
    def __init__(self, lsd):
        threading.Thread.__init__(self)
        self.lsd = lsd
        self.daemon = True
        self.model_name = 'mobilenet_0.75_224_0_model.h5'
        self.model = load_model(self.model_name, custom_objects={'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D})

        self.model.summary()
        self.start()

    def run(self):
        while (True):
            cropped_frame = self.lsd.getFrame()

            # Smile detecting
            cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGRA2BGR)
            cropped_frame = cropped_frame[8:232, 48:272]  # Crop from img_array, y, w, h -> 100, 200, 300, 400
            img_array = image.img_to_array(cropped_frame)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.

            prediction = self.model.predict(img_array)
            self.lsd.setSmile(prediction)
