import os

import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
from keras_unet.models import custom_unet
from tensorflow.keras.optimizers import Adam
from keras_unet.metrics import iou, iou_thresholded


def get_model():
    model = custom_unet(
        (256, 256, 3),
        use_batch_norm=False,
        num_classes=1,
        filters=16,
        dropout=0.2,
        output_activation='sigmoid'
    )
    model.compile(
        optimizer=Adam(),
        # optimizer=SGD(lr=0.01, momentum=0.99),
        loss='binary_crossentropy',
        # loss=jaccard_distance,
        metrics=[iou, iou_thresholded]
    )
    model_filename = 'best/cp-0065.ckpt'
    model.load_weights(model_filename)
    return model


unique_name = "segm_model_v5_0065"
model = get_model()
model.save("%s.h5" % unique_name, overwrite=True, include_optimizer=False)

model = tf.keras.models.load_model('%s.h5' % unique_name)

optimizations = [
    (tf.lite.Optimize.DEFAULT, tf.float32, "default_32fp"),
    (tf.lite.Optimize.DEFAULT, tf.float16, "default_16fp"),
    (tf.lite.Optimize.OPTIMIZE_FOR_LATENCY, tf.float32, "latency_32fp"),
    (tf.lite.Optimize.OPTIMIZE_FOR_LATENCY, tf.float16, "latency_16fp")
]

for optimization, tensor_type, name in optimizations:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.post_training_quantize = True
    converter.optimizations = [optimization]
    converter.target_spec.supported_types = [tensor_type]
    tflite_model = converter.convert()
    open(os.path.join("tflite", "{}_{}.tflite".format(unique_name, name)), "wb").write(tflite_model)

optimizations = [
    (np.uint16, "16fp"),
    (None, "32fp")
]

for tensor_type, name in optimizations:
    folder = os.path.join("tfjs", name)
    os.mkdir(folder)
    tfjs.converters.save_keras_model(model, folder, quantization_dtype=tensor_type,
                                     weight_shard_size_bytes=1024 * 1024 * 4)
