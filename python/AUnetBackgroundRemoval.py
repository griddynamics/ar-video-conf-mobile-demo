import os
import gc
import argparse
import json
# import cv2
import numpy as np
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
        elif len(x.shape == 4):
            return self.__model.predict(x)
        else:
            raise ValueError("Not supported")

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

    def save_tflite(self, ds_home, best=False):
        if best:
            self.load_last_checkpoint()

        path = os.path.join(self.__checkpoints_folder, "tflite")
        if not os.path.exists(path):
            os.mkdir(path)

        ds = data_preparation.representative_tfrecord_dataset(ds_home=ds_home, shape=self.__input_shape, cache=False)
        ds_gen = lambda : data_preparation.representative_tfrecord_dataset_gen(ds)

        optimizations_dict = [
            {
                'opt_name': 'drq', 
                'supported_ops': None,
                'representative_dataset': None
            },
            {
                'opt_name': 'fiq',
                'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
                'representative_dataset': ds_gen
            },
            {
                'opt_name': 'fq',
                'supported_ops': [tf.float16],
                'representative_dataset': None
            },
            {
                'opt_name': '32fp',
                'supported_ops': [tf.float32],
                'representative_dataset': None
            },
            {
                'opt_name': '16fp',
                'supported_ops': [tf.float16],
                'representative_dataset': None
            },
        ]

        for optimization in optimizations_dict:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.__model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if optimization['opt_name'] in {'32fp', '16fp'}:
                converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
                model_file_name = f"{self.__unique_name}_{optimization['opt_name']}.tflite"
                converter.post_training_quantize = True
            else:
                model_file_name = f"post_ds3_{self.__unique_name}_{optimization['opt_name']}.tflite"
            if optimization['supported_ops']:
                converter.target_spec.supported_ops = optimization['supported_ops']
            if optimization['representative_dataset']:
                converter.representative_dataset = optimization['representative_dataset']
            tflite_quant_model = converter.convert()
            with open(os.path.join(path, model_file_name), 'wb') as f:
                f.write(tflite_quant_model)
        del ds, ds_gen
    
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
            validation_data=val_ds,
            validation_steps=np.ceil(steps_per_epoch/0.7*0.3),  # Based on the splitting dataset strategy: 70% - train, 30% - validation
            callbacks=[callback_checkpoint, gccp]
        )

        # for key in self.__history.history.keys():
        #     print(key, type(self.__history.history[key]), type(self.__history.history[key][0]), self.__history.history[key][0])
        with open(os.path.join(self.__checkpoints_folder, f'history_{self.__unique_name}.json'), 'w') as f:
            json.dump({key: list(map(lambda x: float(x), value))
                        for key, value in self.__history.history.items()}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='AUnetBackgroundRemoval', description='Attention UNet parametrizable model trainer for background removal.')
    parser.add_argument('-p', '--pascal-voc-path',
                        required=False, dest='pascal_voc_path',
                        help="specify the path to the PascalVOC2012 dataset. Used for in-memory training (DS v2).")
    parser.add_argument('-c', '--coco-portraints-path',
                        required=False, dest='coco_portraits_path',
                        help="specify the path to the Portraits dataset. Used for in-memory training (DS v1-2).")
    parser.add_argument('-up', '--use-pascal-voc',
                        action='store_true', required=False, dest='use_pascal_voc',
                        help="to enable usage of the PascalVOC2012 dataset. Used for in-memory training way (DS v2). "
                             "If path differs from default, `-p` argument is required.")
    parser.add_argument('-uc', '--use-coco-portraints',
                        action='store_true', required=False, dest='use_coco_portraits',
                        help="to enable usage of the PascalVOC2012 dataset. Used for in-memory training way (DS v1-2). "
                             "If path differs from default, `-c` argument is required.")
    parser.add_argument('-cf', '--checkpoints-folder',
                        required=False, dest='checkpoints_folder', default='training_grid',
                        help="specify the folder for intermediate weights storage and final quantized model storage. "
                             "Default folder: `training_grid`.")
    parser.add_argument('-if', '--interior-folder',
                        required=False, dest='interior_folder',
                        help="specify the folder of interior dataset.")
    parser.add_argument('-ef', '--exterior-folder',
                        required=False, dest='exterior_folder',
                        help="specify the folder of exterior dataset.")
    parser.add_argument('-r', '--tfrecords-path', type=str,
                        required=False, dest='tfrecords_path', default=None,
                        help="specify the path to folder containing tfrecords generated by `data_preparation.py`. "
                             "Used with the dataset v3 and next iterations.")
    parser.add_argument('-e', '--epochs', type=int,
                        required=False, dest='epochs', default=1000,
                        help="specify the total number of training epochs needed. Default: 1000.")
    parser.add_argument('-b', '--batch-size', type=int,
                        required=False, dest='batch_size', default=8,
                        help="specify the batch size. Default: 8.")
    parser.add_argument('-f', '--filters', type=int,
                        required=False, dest='filters', default=8,
                        help="specify number of filters in the first layer of the network. "
                             "Default: 8.")
    parser.add_argument('-sd', '--side', type=int,
                        required=False, dest='side', default=32,
                        help="specify the side of input size (in case of sqare input size). Default: 32.")
    parser.add_argument('-n', '--number-of-examples', type=int,
                        required=False, dest='number_of_examples', default=27781, # Dataset V3-V4
                        help="specify number of examples of datased. Used with `-r`. "
                             "Needed to calculate number of batches for training and validation. "
                             "Default: 27781 (DS V3).")
    parser.add_argument('-ch', '--cache', action='store_true',
                        required=False, dest='cache', default=False,
                        help="specify if datasets caching is needed. Should accelerate model training if "
                             "amount of memory (RAM) is enough.")
    parser.add_argument('-v', '--verbose', type=int,
                        required=False, dest='verbose', default=0)
    parser.add_argument('--remove-all-previous-results', action='store_true',
                        required=False, dest='remove_all',
                        help="used to remove all previous results of training in the `checkpoints_folder` if any. "
                             "Deletes the folder by the path provided in the `checkpoints_folder`")
    parser.add_argument('-l', '--generate-tflite-only', action='store_true',
                        required=False, dest='tflite_only',
                        help="to generate tflite models from the last checkpoint without training.")
    parser.add_argument('--restore', action='store_true',
                        required=False, dest='restore',
                        help="to continue model training after crash or the end of training on "
                             "initial number of epochs.")

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
    background_patterns = (args.interior_folder, args.exterior_folder)
    aunet = AUnetBackgroundRemoval(
        input_shape=input_shape,
        filters=args.filters,
        verbose=args.verbose,
        checkpoints_folder=args.checkpoints_folder)
    
    if args.restore:
        aunet.load_last_checkpoint(warning=True)

    if args.tfrecords_path and not args.tflite_only:
        tfrecord_path = args.tfrecords_path
        steps_per_epoch = (args.number_of_examples + args.batch_size - 1) // args.batch_size
        ds_train, ds_val = data_preparation.read_augment_tfrecord_dataset(data_preparation.AUGMENTATIONS,
                                                                          ds_home=tfrecord_path,
                                                                          background_patterns=background_patterns,
                                                                          shape=input_shape,
                                                                          cache=args.cache,
                                                                          batch_size=args.batch_size,
                                                                          verbose=args.verbose)

        aunet.train_from_dataset(ds_train, ds_val, steps_per_epoch=steps_per_epoch, epochs=args.epochs)
        aunet.save_tflite(best=True, ds_home=args.tfrecords_path)
    # Deprecated (for dataset v2):
    elif args.use_pascal_voc or args.use_coco_portraits:
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
    if args.tflite_only and args.tfrecords_path:
        aunet.save_tflite(best=True, ds_home=args.tfrecords_path)
