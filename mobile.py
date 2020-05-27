import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

cap = cv2.VideoCapture(0)

beach = np.array(Image.open("golden-gate.jpg").resize((256, 256)))
beach = np.asarray(beach, dtype=np.float32) / 255

# Load TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path="deeplabv3_257_mv_gpu.tflite")
interpreter = tf.lite.Interpreter(model_path="segm_model_v5_0065_latency_16fp.tflite")
interpreter.allocate_tensors()

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

frames = 0
while True:
    list = []
    ret, frame = cap.read()
    frame = cv2.resize(frame, (256, 256))
    list.append(frame)
    x = np.asarray(list, dtype=np.float32) / 255

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    input_data = x
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # returns 4d array [1, 256, 256, 1], so unwind
    mask = output_data[0]

    mask_blur = cv2.GaussianBlur(mask, (3, 3), 0)
    mask_blur = np.reshape(mask_blur, (frame.shape[0], frame.shape[1], 1))

    foreground = x[0] * mask_blur
    background = (1.0 - mask_blur) * beach
    outImage = cv2.add(foreground, background)

    cv2.imshow('Input', outImage)
    frames += 1
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()