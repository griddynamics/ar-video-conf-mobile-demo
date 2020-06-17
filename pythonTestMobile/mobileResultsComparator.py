import os
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from array import array
from PIL import Image
import struct

def readimage(path):
    count = os.stat(path).st_size / 2
    with open(path, "rb") as f:
        return bytearray(f.read())

# Load TFLite model and allocate tensors.
#interpreter = tf.lite.Interpreter(model_path="deeplabv3_257_mv_gpu.tflite")
interpreter = tf.lite.Interpreter(model_path="segm_model_v5_0065_latency_16fp.tflite")
interpreter.allocate_tensors()


list = []
#Load normalized [0, 1] image prepared by Android
floats = np.fromfile("normalized0_1Bitmap.bytes", dtype=np.float32, offset=4).reshape(256,256,3) #np.frombuffer(normalizedBytes, dtype=np.float32)
list.append(floats)
x = np.asarray(list, dtype=np.float32) 
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_data = x
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
mask = output_data[0]

outImage0_1 = cv2.add(mask, mask)

list = []
#Load normalized [-1, 1] image prepared by Android
floats = np.fromfile("normalized1_1Bitmap.bytes", dtype=np.float32, offset=4).reshape(256,256,3) #np.frombuffer(normalizedBytes, dtype=np.float32)
list.append(floats)
x = np.asarray(list, dtype=np.float32) 
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_data = x
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
mask = output_data[0]

outImage1_1 = cv2.add(mask, mask)



maskBytes = readimage("maskCustom0_1Bitmap.bytes")
maskImage = Image.frombytes('RGBA', (256,256), bytes(maskBytes), 'raw')


scaledImage = readimage("scaledBitmap.bytes")
androidMask1_1 = readimage("maskCustom1_1Bitmap.bytes")

#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(2,3)
#Results for mask [0, 1]
axarr[0, 0].imshow(Image.frombytes('RGBA', (256,256), bytes(scaledImage), 'raw'))
axarr[0, 0].set_title('Scaled image')
axarr[0, 1].imshow(outImage0_1)
axarr[0, 1].set_title('Python result. Mask [0, 1]')
axarr[0, 2].imshow(maskImage)
axarr[0, 2].set_title('Android result')
#Results for mask [-1, 1]
axarr[1, 0].imshow(Image.frombytes('RGBA', (256,256), bytes(scaledImage), 'raw'))
axarr[1, 0].set_title('Scaled image')
axarr[1, 1].imshow(outImage1_1)
axarr[1, 1].set_title('Python result. Mask [-1, 1]')
axarr[1, 2].imshow(Image.frombytes('RGBA', (256,256), bytes(androidMask1_1), 'raw'))
axarr[1, 2].set_title('Android result')

plt.show()

