import os
import gc
import argparse
import json
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.models import custom_unet
from keras_unet.utils import get_augmented
from sklearn.model_selection import train_test_split


PASCAL_VOC_PERSON_CLASS_ID = 15


def prepare_dataset(pascal_voc_home='/home/aholdobin/pascalvoc2012/VOCdevkit/VOC2012/',
                    coco_home='/home/aholdobin/faces/',
                    shape=(32, 32),
                    verbose=False):
    gc.collect()

    Xs = []
    Ys = []
    # LOADING PASCAL_VOC DATA IF PROVIDED
    images = []
    masks = []

    if pascal_voc_home and os.path.exists(pascal_voc_home):
        segmentation_meta_path = pascal_voc_home + 'ImageSets/Segmentation/trainval.txt'
        segmentation_folder = pascal_voc_home + 'SegmentationClass/'
        images_folder = pascal_voc_home + 'JPEGImages/'

        with open(segmentation_meta_path) as f:
            segmentation_files = f.read().split('\n')

        for file in tqdm(segmentation_files):
            if file and os.path.exists(images_folder + file + '.jpg'):
                images.append(np.array(Image.open(
                    images_folder + file + '.jpg').resize(shape), dtype='float32'))
                masks.append(cv2.resize((np.array(Image.open(
                    segmentation_folder + file + '.png')) == PASCAL_VOC_PERSON_CLASS_ID).astype('float32'), dsize=shape))

        x_pv = np.array(images, dtype='float32')/255.
        y_pv = np.array(masks, dtype='float16')
        
        del images
        del masks
        gc.collect()


        right = np.reshape(
            y_pv, [y_pv.shape[0], shape[0]*shape[1]]).mean(axis=1)
        right = (right > .15) | (right == .0)
        if verbose:
            print(sum(right))
        x_pv = x_pv[right]
        y_pv = y_pv[right]

        Xs.append(x_pv)
        Ys.append(y_pv)

        if verbose:
            print("x_pv: ", x_pv.shape)
            print("y_pv: ", y_pv.shape)

        
    # LOADING COCO DATA IF PROVIDED
    if coco_home and os.path.exists(coco_home):
        images = []
        masks = []
        files = os.listdir(coco_home+'images/')
        if verbose:
            print('files len', len(files))
        files = list(
            map(lambda x: x[:-4] if x.endswith('jpg') else None, files))
        for file in tqdm(files):
            if file and os.path.exists(coco_home + f'images/{file}.jpg'):
                images.append(
                    np.array(Image.open(coco_home + f'images/{file}.jpg').resize(shape), dtype='float32'))
                masks.append(
                    np.array(Image.open(coco_home + f'masks/{file}.png').resize(shape))[:, :, -1])

        x_c = np.asarray(images, dtype='float32') / 255
        del images
        gc.collect()
        y_c = np.asarray(masks, dtype='float16') / 255
        del masks
        gc.collect()
        Xs.append(x_c)
        Ys.append(y_c)

    if len(Xs) == 0:
        raise ValueError('Need at least 1 source of data')

    X = np.concatenate(Xs)
    y = np.concatenate(Ys)
    y = np.reshape(y, (y.shape[0], y.shape[1], y.shape[2], 1))

    del Xs
    del Ys
    gc.collect()
    
    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=0.30, random_state=0)

    if verbose:
        print("x_train: ", x_train.shape)
        print("y_train: ", y_train.shape)
        print("x_val: ", x_val.shape)
        print("y_val: ", y_val.shape)

    return x_train, y_train, x_val, y_val


class AUnetBackgroundRemoval:

    DEFAULT_AUGM_ARGS = dict(
        rotation_range=15.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=50,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='constant'
    )

    def __init__(self, input_shape=(32, 32), filters=8, verbose=0, use_attention=True,
                 checkpoints_folder='training_grid'):
        self.__verbose = verbose
        self.__input_shape = (input_shape[0], input_shape[1], 3)
        self.__filters = filters
        self.__unique_name = f"fil_{filters}_shape_{self.__input_shape[0]}x{self.__input_shape[1]}"
        self.__checkpoints_folder = os.path.join(
            checkpoints_folder, self.__unique_name)
        if self.__verbose:
            print(self.__checkpoints_folder)
        self.__model = custom_unet(
            self.__input_shape,
            use_batch_norm=True,
            num_classes=1,
            filters=self.__filters,
            use_attention=use_attention,
            dropout=0.2,
            output_activation='sigmoid'
        )

        self.__model.compile(
            optimizer=Adam(),
            loss='binary_crossentropy',
            metrics=[iou, iou_thresholded]
        )
        if verbose:
            self.__model.summary()

        if not os.path.exists(checkpoints_folder):
            os.mkdir(checkpoints_folder)

        if not os.path.exists(self.__checkpoints_folder):
            os.mkdir(self.__checkpoints_folder)

    def train(self, x_train, y_train, x_val, y_val, epochs=1000, batch_size=1024, augm_args=DEFAULT_AUGM_ARGS):

        train_gen = get_augmented(
            x_train, y_train, batch_size=batch_size,
            data_gen_args=augm_args)

        self.__checkpoints_path = os.path.join(
            self.__checkpoints_folder, '') + "weights.{epoch:04d}-{val_loss:.2f}.hdf5"
        callback_checkpoint = ModelCheckpoint(
            filepath=self.__checkpoints_path,
            verbose=self.__verbose,
            monitor='val_loss',
            save_best_only=True,
        )

        self.__history = self.__model.fit(
            train_gen,
            steps_per_epoch=(len(x_train)+batch_size-1)//batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=[callback_checkpoint]
        )

        # for key in self.__history.history.keys():
        #     print(key, type(self.__history.history[key]), type(self.__history.history[key][0]), self.__history.history[key][0])
        with open(os.path.join(self.__checkpoints_folder, f'history_{self.__unique_name}.json'), 'w') as f:
            json.dump({key: list(map(lambda x: float(x), value))
                       for key, value in self.__history.history.items()}, f)

    def predict(self, x):
        if len(x.shape == 3):
            return self.__model.predict(x.reshape((1, * x.shape)))

    def save_best_tflite(self):
        checkpoints = os.listdir(self.__checkpoints_folder)
        if len(checkpoints) == 0:
            raise Exception('You must train model before restoring weights')
        checkpoints = list(map(lambda x: (x, int(x[8:].split(
            '-')[0])) if x.startswith('weights') else (x, -1), checkpoints))
        checkpoints = sorted(checkpoints, key=lambda x: x[1])
        name = checkpoints[-1][0]

        self.__model.load_weights(os.path.join(
            self.__checkpoints_folder, name))
        optimizations = [
            (tf.lite.Optimize.DEFAULT, tf.float32, "default_32fp"),
            (tf.lite.Optimize.DEFAULT, tf.float16, "default_16fp"),
            (tf.lite.Optimize.OPTIMIZE_FOR_LATENCY, tf.float32, "latency_32fp"),
            (tf.lite.Optimize.OPTIMIZE_FOR_LATENCY, tf.float16, "latency_16fp")
        ]

        path = os.path.join(self.__checkpoints_folder, "tflite")
        if not os.path.exists(path):
            os.mkdir(path)

        for optimization, tensor_type, name in optimizations:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.__model)
            converter.post_training_quantize = True
            converter.optimizations = [optimization]
            converter.target_spec.supported_types = [tensor_type]
            tflite_model = converter.convert()

            with open(os.path.join(path, "{}_{}.tflite".format(self.__unique_name, name)), "wb") as f:
                f.write(tflite_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='AUnetBackgroundRemoval', description='Attention UNet parametrizable model trainer for background removal')
    parser.add_argument('-p', '--pascal-voc-path',
                        required=False, dest='pascal_voc_path')
    parser.add_argument('-c', '--coco-portraints-path',
                        required=False, dest='coco_portraits_path')
    # TODO
    # parser.add_argument('-s', '--supervisely-person-path', required=False, dest='supervisely_person_path')
    parser.add_argument('-up', '--use_pascal-voc-path',
                        action='store_true', required=False, dest='use_pascal_voc')
    parser.add_argument('-uc', '--use-coco-portraints-path',
                        action='store_true', required=False, dest='use_coco_portraits')
    # TODO
    # parser.add_argument('-us', '--use-supervisely-person-path', action='store_true', required=False, dest='use_supervisely_person')
    parser.add_argument('-e', '--epochs', type=int,
                        required=False, dest='epochs', default=1000)
    parser.add_argument('-b', '--batch-size', type=int,
                        required=False, dest='batch_size', default=8)
    parser.add_argument('-f', '--filters', type=int,
                        required=False, dest='filters', default=8)
    parser.add_argument('-sd', '--side', type=int,
                        required=False, dest='side', default=32)
    parser.add_argument('-v', '--verbose', type=int,
                        required=False, dest='verbose', default=0)
    parser.add_argument('--remove-all-previous-results', action='store_true',
                        required=False, dest='remove_all')
    parser.add_argument('-l', '--generate-tflite', action='store_true',
                        required=False, dest='tflite')

    args = parser.parse_args()

    if args.remove_all and os.path.exists('training_grid'):
        import shutil
        shutil.rmtree('training_grid')

    # preparing environment

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # From [TensorFlow guide][1]https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            first_gpu = gpus[0]
            tf.config.experimental.set_virtual_device_configuration(first_gpu,
                                                                    [tf.config.experimental.VirtualDeviceConfiguration(
                                                                        memory_limit=7000)])
        except RuntimeError as e:
            print(e)

    input_shape = (args.side, args.side)

    aunet = AUnetBackgroundRemoval(
        input_shape=input_shape,
        filters=args.filters,
        verbose=args.verbose)

    if args.use_pascal_voc or args.use_coco_portraits:  # TODO add or args.use_supervisely_person
        # Dataset (train and val) preparation
        # prepare args:
        pascal_voc_home = '/home/aholdobin/pascalvoc2012/VOCdevkit/VOC2012/'
        if args.use_pascal_voc:
            if args.pascal_voc_path:
                pascal_voc_home = args.pascal_voc_path
        else:
            pascal_voc_home = None
        coco_home = '/home/aholdobin/faces/'
        if args.use_coco_portraits:
            if args.coco_portraits_path:
                coco_home = args.coco_portraits_path
        else:
            coco_home = None

        x_train, y_train, x_val, y_val = prepare_dataset(
            pascal_voc_home=pascal_voc_home, coco_home=coco_home, shape=input_shape, verbose=args.verbose)
        gc.collect()

        # train model
        aunet.train(x_train, y_train, x_val, y_val,
                    epochs=args.epochs, batch_size=args.batch_size)

    if args.tflite:
        aunet.save_best_tflite()
