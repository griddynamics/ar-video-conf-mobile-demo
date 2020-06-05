import os
import io
import cv2
import numpy as np
import tensorflow as tf
from array import array
from PIL import Image
import struct

def readimage(path):
    count = os.stat(path).st_size / 2
    with open(path, "rb") as f:
        return bytearray(f.read())

scaledBytes = readimage("scaledBitmap.bytes")
normalizedBytes = readimage("normalized0_1Bitmap.bytes")
maskBytes = readimage("maskCustom0_1Bitmap.bytes")
image = Image.frombytes('RGBA', (256,256), bytes(scaledBytes), 'raw')
rgbImage = Image.new("RGB", image.size, (255, 255, 255))
rgbImage.paste(image, mask=image.split()[3]) # 3 is the alpha channel
#image.show()
# Load TFLite model and allocate tensors.
#interpreter = tf.lite.Interpreter(model_path="deeplabv3_257_mv_gpu.tflite")
interpreter = tf.lite.Interpreter(model_path="segm_model_v5_0065_latency_16fp.tflite")
interpreter.allocate_tensors()


list = []
imageBytes = np.array(rgbImage)
imageNormalizedBytes = np.array(Image.frombytes('RGB', (256,256), bytes(maskBytes), 'raw'))

floats = np.fromfile("normalized0_1Bitmap.bytes", dtype=np.float32, offset=4).reshape(256,256,3) #np.frombuffer(normalizedBytes, dtype=np.float32)
#print(floats)
#print(len(floats))
list.append(imageBytes)
#print(imageNormalizedBytes)
#print(len(imageNormalizedBytes))
#print(len(imageNormalizedBytes[0]))
#print(len(imageNormalizedBytes[0][0]))
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

#outImage = cv2.add(np.array(maskBytes), np.array(maskBytes))
outImage = cv2.add(mask, mask)

cv2.imshow('Input', outImage)
while(True):
    c = cv2.waitKey(1)
    if c == 27:
        break
    
#rgbImage.show()

maskImage = Image.frombytes('RGBA', (256,256), bytes(maskBytes), 'raw')
maskImage.show()

