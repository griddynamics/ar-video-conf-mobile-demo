import os
import gc
import argparse
import json
# import cv2
# import numpy as np
import tensorflow as tf
# from PIL import Image
# from tqdm import tqdm
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.models import custom_unet
from keras_unet.utils import get_augmented
# from sklearn.model_selection import train_test_split
import data_preparation


class GcCollectCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        gc.collect()


class AUnetBackgroundRemoval:

    def __init__(self, input_shape=(32, 32), filters=8, verbose=0, use_attention=True,
                 checkpoints_folder='training_grid'):
        self.__verbose = verbose
        self.__input_shape = (input_shape[0], input_shape[1], 3)
        self.__filters = filters
        self.__unique_name = f"fil_{filters}_shape_{self.__input_shape[0]}x{self.__input_shape[1]}"
        self.__checkpoints_folder = os.path.join(
            checkpoints_folder, self.__unique_name)
        self.__initial_epoch = 0
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

    def get_model(self):
        return self.__model

    def predict(self, x):
        if len(x.shape == 3):
            return self.__model.predict(x.reshape((1, * x.shape)))

    def load_last_checkpoint(self, warning=False):
        checkpoints = os.listdir(self.__checkpoints_folder)
        checkpoints = [ (x, int(x[8:].split('-')[0])) for x in checkpoints if x.startswith('weights') ]
        if len(checkpoints) == 0:
            if warning:
                print('You must train model before restoring weights!')
                return
            raise Exception('You must train model before restoring weights')
        checkpoints = sorted(checkpoints, key=lambda x: x[1])
        name = checkpoints[-1][0]
        self.__initial_epoch = checkpoints[-1][1]
        self.__model.load_weights(os.path.join(
            self.__checkpoints_folder, name))

    def save_tflite(self, best=False):
        if best:
            self.load_last_checkpoint()
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
    
    def train_from_memory(self, x_train, y_train, x_val, y_val, epochs=1000, batch_size=1024, augm_args=data_preparation.AUGMENTATIONS):

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
            initial_epoch=self.__initial_epoch,
            validation_data=(x_val, y_val),
            callbacks=[callback_checkpoint]
        )

        # for key in self.__history.history.keys():
        #     print(key, type(self.__history.history[key]), type(self.__history.history[key][0]), self.__history.history[key][0])
        with open(os.path.join(self.__checkpoints_folder, f'history_{self.__unique_name}.json'), 'w') as f:
            json.dump({key: list(map(lambda x: float(x), value))
                        for key, value in self.__history.history.items()}, f)

    def train_from_dataset(self, train_ds, val_ds, steps_per_epoch, epochs=1000):

        self.__checkpoints_path = os.path.join(
            self.__checkpoints_folder, '') + "weights.{epoch:04d}-{val_loss:.2f}.hdf5"

        callback_checkpoint = ModelCheckpoint(
            filepath=self.__checkpoints_path,
            verbose=self.__verbose,
            monitor='val_loss',
            save_best_only=True,
        )

        gccp = GcCollectCallback()

        self.__history = self.__model.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data= val_ds,
            callbacks=[callback_checkpoint, gccp]
        )

        # for key in self.__history.history.keys():
        #     print(key, type(self.__history.history[key]), type(self.__history.history[key][0]), self.__history.history[key][0])
        with open(os.path.join(self.__checkpoints_folder, f'history_{self.__unique_name}.json'), 'w') as f:
            json.dump({key: list(map(lambda x: float(x), value))
                        for key, value in self.__history.history.items()}, f)


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
    parser.add_argument('-r', '--tfrecords-path', type=str,
                        required=False, dest='tfrecords_path', default=None)
    parser.add_argument('-e', '--epochs', type=int,
                        required=False, dest='epochs', default=1000)
    parser.add_argument('-b', '--batch-size', type=int,
                        required=False, dest='batch_size', default=8)
    parser.add_argument('-f', '--filters', type=int,
                        required=False, dest='filters', default=8)
    parser.add_argument('-sd', '--side', type=int,
                        required=False, dest='side', default=32)
    parser.add_argument('-e', '--number-of-examples', type=int,
                        required=False, dest='number_of_examples', default=27781)
    parser.add_argument('-ch', '--cache', action='store_true',
                        required=False, dest='cache', default=False)
    parser.add_argument('-v', '--verbose', type=int,
                        required=False, dest='verbose', default=0)
    parser.add_argument('--remove-all-previous-results', action='store_true',
                        required=False, dest='remove_all')
    parser.add_argument('-l', '--generate-tflite', action='store_true',
                        required=False, dest='tflite')
    parser.add_argument('--restore', action='store_true',
                        required=False, dest='restore')

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
    
    if args.restore:
        aunet.load_last_checkpoint(warning=True)

    if args.tfrecords_path:
        tfrecord_path = args.tfrecords_path
        steps_per_epoch = (args.number_of_examples + args.batch_size - 1) // args.batch_size
        ds_train, ds_val = data_preparation.read_augment_tfrecord_dataset(data_preparation.AUGMENTATIONS,
                                                                          ds_home=tfrecord_path,
                                                                          shape=input_shape,
                                                                          cache=args.cache,
                                                                          batch_size=args.batch_size,
                                                                          verbose=args.verbose)

        aunet.train_from_dataset(ds_train, ds_val, steps_per_epoch=steps_per_epoch, epochs=args.epochs)

    elif args.use_pascal_voc or args.use_coco_portraits:  # TODO add or args.use_supervisely_person
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

        x_train, y_train, x_val, y_val = data_preparation.prepare_process_dataset_into_memory(
            pascal_voc_home=pascal_voc_home, coco_home=coco_home, shape=input_shape, verbose=args.verbose)
        gc.collect()

        # train model
        aunet.train_from_memory(x_train, y_train, x_val, y_val,
                    epochs=args.epochs, batch_size=args.batch_size)
    if args.tflite:
        aunet.save_tflite(best=True)


    
