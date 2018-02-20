import cv2
from keras.models import load_model
import h5py
from skimage import color, io
import numpy as np
import matplotlib.pyplot as plt
from face_cropper import FaceCropper
import time


def demo():
    cam = cv2.VideoCapture(0)
    time.sleep(3)
    s, img = cam.read()
    if s:
        cv2.imwrite("image.jpg", img)
        cv2.imshow('raw_img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    detector = FaceCropper()
    detector.generate(img, False)

    image = cv2.imread('image1.jpg',0)
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_LINEAR)

    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image = np.array([[[image]]])

    model = load_model('trainingmodels/conv-lstm-model-036.h5')
    prediction = model.predict_classes(image)
    # [0,3,6] --> {anger: 0, happy: 2, neutral: 1}
    if prediction[0] == 0:
        print('Prediction: Anger')
    elif prediction[0] == 1:
        print('Prediction: Neutral')
    else:
        print('Prediction: Happiness')


demo()