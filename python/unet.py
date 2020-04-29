import glob
import os
import tensorflow as tf
import numpy as np

from tqdm import tqdm
from datetime import datetime
from keras_unet.models import custom_unet, vanilla_unet
from keras_unet.utils import plot_segm_history
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.utils import plot_imgs
from keras_unet.utils import get_augmented
from sklearn.model_selection import train_test_split
from PIL import Image

model_filename = 'segm_model_v4_32_mega.h5'

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# From [TensorFlow guide][1]https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        first_gpu = gpus[0]
        tf.config.experimental.set_virtual_device_configuration(first_gpu,
                                                                [tf.config.experimental.VirtualDeviceConfiguration(
                                                                    memory_limit=2000)])
    except RuntimeError as e:
        print(e)

shape = (256, 256)

# model = vanilla_unet(input_shape=(196, 196, 3))

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

masks = glob.glob(os.path.join("faces", "masks", "*.png"))[:7000]
orgs = list(map(lambda x: x.replace(".png", ".jpg").replace("masks", "images"), masks))

imgs_list = []
masks_list = []
for image, mask in tqdm(zip(orgs, masks)):
    if os.path.exists(image):
        imgs_list.append(np.array(Image.open(image).resize(shape)))
        masks_list.append(np.array(Image.open(mask).resize(shape))[:, :, -1])

imgs_np = np.asarray(imgs_list)
masks_np = np.asarray(masks_list)

print(imgs_np.shape, masks_np.shape)

# plot_imgs(org_imgs=imgs_np, mask_imgs=masks_np, nm_img_to_plot=10, figsize=6)
x = np.asarray(imgs_np, dtype=np.float32) / 255
y = np.asarray(masks_np, dtype=np.float32) / 255
print(x.max(), y.max())

y = np.reshape(y, (y.shape[0], y.shape[1], y.shape[2], 1))
print(x.shape, y.shape)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.30, random_state=0)

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_val: ", x_val.shape)
print("y_val: ", y_val.shape)

train_gen = get_augmented(
    x_train, y_train, batch_size=2,
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


# tensorboard_callback = TensorBoard(log_dir=logdir)
callback_checkpoint = ModelCheckpoint(
    model_filename,
    verbose=1,
    monitor='val_loss',
    save_best_only=True,
)

history = model.fit_generator(
    train_gen,
    steps_per_epoch=500,
    epochs=350,
    validation_data=(x_val, y_val),
    callbacks=[callback_checkpoint]
)

plot_segm_history(history)

model.load_weights(model_filename)
y_pred = model.predict(x_val)

plot_imgs(org_imgs=x_val, mask_imgs=y_val, pred_imgs=y_pred, nm_img_to_plot=10)
