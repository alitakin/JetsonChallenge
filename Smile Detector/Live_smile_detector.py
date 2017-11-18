import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet import relu6, DepthwiseConv2D


model_name = 'mobilenet_0.75_224_0_model.h5'
model = load_model(model_name, custom_objects={'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D})

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,224);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,224);

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    color = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    crop_img = color[8:232, 48:272]  # Crop from x, y, w, h -> 100, 200, 300, 400
    faces = face_cascade.detectMultiScale(crop_img, 1.3, 5)

    # Detecting and drawing face rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(crop_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # roi_color = crop_img[y:y + h, x:x + w]

    # Smile detecting
    x = image.img_to_array(crop_img)
    x = np.expand_dims(x, axis=0)
    x /= 255.
    prediction = model.predict(x)
    prediction_label = 'Smile' if prediction.item(1) >= prediction.item(0) else 'Not smile'
    cv2.putText(crop_img, prediction_label, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness=2)

    # Display the resulting frame and printing prediction
    cv2.imshow('frame', crop_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()