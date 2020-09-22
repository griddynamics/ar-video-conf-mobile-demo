import os
import gc
import argparse
import cv2

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
from pprint import pprint

PASCAL_VOC_PERSON_CLASS_ID = 15


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_segmentation_example(x, y):
    x = _bytes_feature(tf.io.serialize_tensor(x))
    y = _bytes_feature(tf.io.serialize_tensor(y))
    example = tf.train.Example(
        features=tf.train.Features(feature={'x': x, 'y': y}))
    return example.SerializeToString()


def read_tfrecord(serialized_example):
    feature_description = {
        'x': tf.io.FixedLenFeature((), tf.string),
        'y': tf.io.FixedLenFeature((), tf.string),
    }
    example = tf.io.parse_single_example(
        serialized_example, feature_description)

    x = tf.io.parse_tensor(example['x'], out_type=tf.float32)
    y = tf.io.parse_tensor(example['y'], out_type=tf.float32)

    return x, y


def preprocess_store_coco(ds_train, ds_val,
                          dataset_home='/home/aholdobin/faces/',
                          tfrecords_dest='/home/aholdobin/tfrecords/',
                          shape=(32, 32),
                          verbose=False):

    def store_chunk(record_name, files, filenames):
        if len(files) == 0:
            raise ValueError("`files` must contain at least 1 element")
        with tf.io.TFRecordWriter(record_name) as writer:
            for file in files:
                if file and os.path.exists(dataset_home + f'images/{file}.jpg'):
                    x = np.array(Image.open(
                        dataset_home + f'images/{file}.jpg').resize(shape), dtype='float32')/255.
                    y = np.array(Image.open(
                        dataset_home + f'masks/{file}.png').resize(shape), dtype='float32')[:, :, -1]/255.
                    serialized_example = serialize_segmentation_example(x, y)
                    writer.write(serialized_example)
        filenames.append(record_name)
        gc.collect()

    if not os.path.exists(tfrecords_dest):
        os.mkdir(tfrecords_dest)

    NUMBER_OF_EXAMPLES_IN_FILE = 200 * 10**6 // (shape[0] * shape[1] * 3 * 4)

    if dataset_home and os.path.exists(dataset_home):
        files = os.listdir(dataset_home+'images/')

        filename_pattern = "cocoportrait_{}x{}_{}_{}.tfrecord"
        if verbose:
            print('files len', len(files))
        files = list(
            map(lambda x: x[:-4] if x.endswith('jpg') else None, files))

        f_train, f_val = train_test_split(files, test_size=0.3, random_state=0)

        FILE_NUM = (len(f_train) + NUMBER_OF_EXAMPLES_IN_FILE -
                    1) // NUMBER_OF_EXAMPLES_IN_FILE
        for i in range(FILE_NUM):
            store_chunk(os.path.join(tfrecords_dest, filename_pattern.format(
                shape[0], shape[1], 'train', i)), f_train[i*NUMBER_OF_EXAMPLES_IN_FILE:(i+1)*NUMBER_OF_EXAMPLES_IN_FILE], ds_train)

        FILE_NUM = (len(f_val) + NUMBER_OF_EXAMPLES_IN_FILE -
                    1) // NUMBER_OF_EXAMPLES_IN_FILE
        for i in range(FILE_NUM):
            store_chunk(os.path.join(tfrecords_dest, filename_pattern.format(
                shape[0], shape[1], 'val', i)), f_val[i*NUMBER_OF_EXAMPLES_IN_FILE:(i+1)*NUMBER_OF_EXAMPLES_IN_FILE], ds_val)

    return ds_train, ds_val


def preprocess_store_pascal(ds_train, ds_val,
                            dataset_home='/home/aholdobin/pascalvoc2012/VOCdevkit/VOC2012/',
                            tfrecords_dest='/home/aholdobin/tfrecords/',
                            shape=(32, 32),
                            verbose=False):

    def store_chunk(record_name, files, filenames, images_folder, segmentation_folder):
        if len(files) == 0:
            raise ValueError("`files` must contain at least 1 element")
        with tf.io.TFRecordWriter(record_name) as writer:
            for file in files:
                if file and os.path.exists(images_folder + f'{file}.jpg'):
                    y = cv2.resize((np.array(Image.open(
                        segmentation_folder + file + '.png')) == PASCAL_VOC_PERSON_CLASS_ID).astype('float32'), dsize=shape)
                    tmean = y.mean()
                    if 0 < tmean <= .15:
                        continue
                    x = np.array(Image.open(images_folder + file +
                                            '.jpg').resize(shape), dtype='float32')/255.
                    serialized_example = serialize_segmentation_example(x, y)
                    writer.write(serialized_example)
        filenames.append(record_name)
        gc.collect()

    if not os.path.exists(tfrecords_dest):
        os.mkdir(tfrecords_dest)

    NUMBER_OF_EXAMPLES_IN_FILE = 200 * 10**6 // (shape[0] * shape[1] * 3 * 4)

    filenames = []
    if dataset_home and os.path.exists(dataset_home):

        segmentation_meta_path = dataset_home + 'ImageSets/Segmentation/trainval.txt'
        segmentation_folder = dataset_home + 'SegmentationClass/'
        images_folder = dataset_home + 'JPEGImages/'

        with open(segmentation_meta_path) as f:
            files = f.read().split('\n')

        i = 0
        FILE_NUM = 0
        filename_pattern = "pascalvoc_{}x{}_{}_{}.tfrecord"

        if verbose:
            print('files len', len(files))

        f_train, f_val = train_test_split(files, test_size=0.3, random_state=0)

        FILE_NUM = (len(f_train) + NUMBER_OF_EXAMPLES_IN_FILE -
                    1) // NUMBER_OF_EXAMPLES_IN_FILE
        for i in range(FILE_NUM):
            store_chunk(os.path.join(tfrecords_dest, filename_pattern.format(shape[0], shape[1], 'train', i)), f_train[i*NUMBER_OF_EXAMPLES_IN_FILE:(
                i+1)*NUMBER_OF_EXAMPLES_IN_FILE], ds_train, images_folder=images_folder, segmentation_folder=segmentation_folder)

        FILE_NUM = (len(f_val) + NUMBER_OF_EXAMPLES_IN_FILE -
                    1) // NUMBER_OF_EXAMPLES_IN_FILE
        for i in range(FILE_NUM):
            store_chunk(os.path.join(tfrecords_dest, filename_pattern.format(shape[0], shape[1], 'val', i)), f_val[i*NUMBER_OF_EXAMPLES_IN_FILE:(
                i+1)*NUMBER_OF_EXAMPLES_IN_FILE], ds_val, images_folder=images_folder, segmentation_folder=segmentation_folder)

    return ds_train, ds_val


def prepare_and_store_dataset(pascal_voc_home='/home/aholdobin/pascalvoc2012/VOCdevkit/VOC2012/',
                              coco_home='/home/aholdobin/faces/',
                              tfrecords_dest='/home/aholdobin/tfrecords/',
                              shape=(32, 32),
                              verbose=False):

    if not os.path.exists(tfrecords_dest):
        os.mkdir(tfrecords_dest)

    ds_train = []
    ds_val = []

    if coco_home and os.path.exists(coco_home):
        preprocess_store_coco(ds_train, ds_val, dataset_home=coco_home,
                              tfrecords_dest=tfrecords_dest, shape=shape, verbose=verbose)
    if pascal_voc_home and os.path.exists(pascal_voc_home):
        preprocess_store_pascal(ds_train, ds_val, dataset_home=pascal_voc_home,
                                tfrecords_dest=tfrecords_dest, shape=shape, verbose=verbose)

    if len(ds_train) == 0:
        raise ValueError('Need at least 1 source of data')

    return tfrecords_dest, ds_train, ds_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='data_preparation', description='Preprocess data and transform in tfrecord format')
    parser.add_argument('-p', '--pascal-voc-path',
                        required=False, dest='pascal_voc_path')
    parser.add_argument('-c', '--coco-portraints-path',
                        required=False, dest='coco_portraits_path')
    # TODO
    # parser.add_argument('-s', '--supervisely-person-path', required=False, dest='supervisely_person_path')
    parser.add_argument('-d', '--tfrecord-destination',
                        required=False, dest='tfrecord_destination')
    parser.add_argument('-up', '--use_pascal-voc-path',
                        action='store_true', required=False, dest='use_pascal_voc')
    parser.add_argument('-uc', '--use-coco-portraints-path',
                        action='store_true', required=False, dest='use_coco_portraits')
    # TODO
    # parser.add_argument('-us', '--use-supervisely-person-path', action='store_true', required=False, dest='use_supervisely_person')
    parser.add_argument('-sd', '--side', type=int,
                        required=False, dest='side', default=32)
    parser.add_argument('-v', '--verbose', type=int,
                        required=False, dest='verbose', default=0)
    parser.add_argument('--remove-all-previous-results', action='store_true',
                        required=False, dest='remove_all')

    args = parser.parse_args()

    input_shape = (args.side, args.side)

    if args.tfrecord_destination:
        tfrecord_destination = args.tfrecord_destination
    else:
        tfrecord_destination = '/home/aholdobin/tfrecords/'

    if args.remove_all and os.path.exists(tfrecord_destination):
        import shutil
        shutil.rmtree(tfrecord_destination)

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

        tfrecord_destination,  ds_train, ds_val = prepare_and_store_dataset(
            pascal_voc_home=pascal_voc_home, coco_home=coco_home, tfrecords_dest=tfrecord_destination, shape=input_shape, verbose=args.verbose)
        pprint(ds_train)
        pprint(ds_val)
        gc.collect()
