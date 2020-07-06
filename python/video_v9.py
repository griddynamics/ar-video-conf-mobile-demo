import os

import cv2
import time
import tensorflow as tf
import imutils as imutils
from keras_unet.models import custom_unet
import numpy as np
from PIL import Image
from tensorflow.keras.optimizers import Adam
from keras_unet.metrics import iou, iou_thresholded

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_model():
    model = custom_unet(
        (256, 256, 4),
        use_batch_norm=False,
        num_classes=1,
        filters=8,
        num_layers=4,
        use_attention=True,
        dropout=0.2,
        output_activation='sigmoid'
    )
    model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy',
        metrics=[iou, iou_thresholded]
    )
    model_filename = 'segm_model_v9.h5'
    model.load_weights(model_filename)
    return model


model = get_model()

print(model.summary())

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorKNN(history=10000)

beach = np.array(Image.open("demo.png").resize((256, 256)))
beach = np.asarray(beach, dtype=np.float32) / 255

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

frames = 0
while True:
    list = []
    ret, frame = cap.read()
    frame = np.fliplr(cv2.resize(imutils.resize(frame, height=256), (256, 256)))
    list.append(frame)
    x = np.asarray(list, dtype=np.float32) / 255

    bilateral_filtered_image = cv2.bilateralFilter(frame, 5, 15, 20)
    edge_detected_image = cv2.Canny(bilateral_filtered_image, 100, 250)
    background_mask = fgbg.apply(frame) / 255

    start = time.perf_counter_ns()
    combined = np.reshape(np.dstack((frame, edge_detected_image)),
                          (1, 256, 256, 4))
    y_pred = model.predict(combined)
    print((time.perf_counter_ns() - start) / 1000000)
    mask = y_pred[0]

    mask_blur = cv2.GaussianBlur(mask, (3, 3), 0)
    mask_blur = np.reshape(mask_blur, (frame.shape[0], frame.shape[1], 1))

    foreground = x[0] * mask_blur
    background = (1.0 - mask_blur) * beach
    outImage = cv2.add(foreground, background)

    cv2.imshow('Input 1', frame)
    cv2.imshow('Input 2', background_mask)
    cv2.imshow('Input 3', edge_detected_image)
    cv2.imshow('Input 4', mask_blur)
    cv2.imshow('Input 5', outImage)
    frames += 1
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
