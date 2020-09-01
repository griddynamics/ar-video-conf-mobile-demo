import glob
import os
import random

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.models import custom_unet
from keras_unet.utils import get_augmented
from keras_unet.utils import plot_imgs
from sklearn.model_selection import train_test_split
from tqdm import tqdm

model_filename = 'segm_model_v17.h5'
checkpoint_path = "training_17/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# From [TensorFlow guide][1]https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        first_gpu = gpus[0]
        tf.config.experimental.set_virtual_device_configuration(first_gpu,
                                                                [tf.config.experimental.VirtualDeviceConfiguration(
                                                                    memory_limit=4000)])
    except RuntimeError as e:
        print(e)

shape = (32, 32)

# model = vanilla_unet(input_shape=(196, 196, 3))

model = custom_unet(
    (32, 32, 3),
    use_batch_norm=True,
    num_classes=1,
    filters=8,
    use_attention=True,
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

print(model.summary())

if os.path.exists(model_filename):
    print("Loading model from checkpoint {}".format(model_filename))
    model.load_weights(model_filename)

while True:
    masks = glob.glob(os.path.join("faces", "masks", "*.png"))
    orgs = list(map(lambda x: x.replace(".png", ".jpg").replace("masks", "images"), masks))

    imgs_list = []
    masks_list = []
    for image, mask in tqdm(zip(orgs, masks)):
        if os.path.exists(image):
            imgs_list.append(np.array(Image.open(image).resize(shape)))
            masks_list.append(np.array(Image.open(mask).resize(shape))[:, :, -1])

    imgs_np = np.asarray(imgs_list)
    masks_np = np.asarray(masks_list)

    # plot_imgs(org_imgs=imgs_np, mask_imgs=masks_np, nm_img_to_plot=10, figsize=6)
    x = np.asarray(imgs_np, dtype=np.float32) / 255
    y = np.asarray(masks_np, dtype=np.float32) / 255
    print(x.max(), y.max())
    del imgs_np
    del masks_np

    y = np.reshape(y, (y.shape[0], y.shape[1], y.shape[2], 1))
    print(x.shape, y.shape)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.30, random_state=0)

    del x
    del y

    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)
    print("x_val: ", x_val.shape)
    print("y_val: ", y_val.shape)

    train_gen = get_augmented(
        x_train, y_train, batch_size=1024,
        data_gen_args=dict(
            rotation_range=15.,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=50,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='constant'
        ))

    callback_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        monitor='val_loss',
        save_best_only=True,
    )

    history = model.fit(
        train_gen,
        steps_per_epoch=35,
        epochs=15150,
        validation_data=(x_val, y_val),
        callbacks=[callback_checkpoint]
    )

# plot_segm_history(history)
#
# model.load_weights(model_filename)
# y_pred = model.predict(x_val)
#
# plot_imgs(org_imgs=x_val, mask_imgs=y_val, pred_imgs=y_pred, nm_img_to_plot=10)
