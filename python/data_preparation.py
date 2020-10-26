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


ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

SUPERVISELY = 'Supervise.ly'
COCO = 'COCO'

PASCAL_VOC_PERSON_CLASS_ID = 15
AUGMENTATIONS = dict(
    rotation_range=15.,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=50,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='constant'
)
AUTOTUNE = tf.data.experimental.AUTOTUNE
FEATURE_DESCRIPTION = {
    'x': tf.io.FixedLenFeature((), tf.string),
    'y': tf.io.FixedLenFeature((), tf.string),
}

def prepare_process_dataset_into_memory(pascal_voc_home='/home/aholdobin/pascalvoc2012/VOCdevkit/VOC2012/',
                    coco_home='/home/aholdobin/faces/',
                    shape=(32, 32),
                    verbose=False):
    # IMPORTANT NOTE: this method implementats creation of the second dataset version (only COCO and
    # PASCAL VOC datasets are used and from PASCAL VOC dataset selected files where person (people)
    # part is 0 or > .15)
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


def preprocess_store_coco_like(ds_train, ds_val,
                          dataset_home='/home/aholdobin/faces/',
                          tfrecords_dest='/home/aholdobin/tfrecords_v3/',
                          shape=(32, 32),
                          verbose=False,
                          dataset=COCO):

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
                    serialized_example = serialize_segmentation_example(x, y.reshape([y.shape[0], y.shape[1], 1]))
                    writer.write(serialized_example)
        filenames.append(record_name)
        gc.collect()

    if not os.path.exists(tfrecords_dest):
        os.mkdir(tfrecords_dest)

    NUMBER_OF_EXAMPLES_IN_FILE = 200 * 10**6 // (shape[0] * shape[1] * 3 * 4)

    if dataset_home and os.path.exists(dataset_home):

        if verbose:
            ds_dict = {COCO: 'COCO Portrait dataset', SUPERVISELY: 'Supervise.ly Person'}
            print('Processinng and storing {} dataset'.format(ds_dict[dataset]))
        
        files = os.listdir(dataset_home+'images/')

        filename_pattern = {
            COCO: "cocoportrait_{}x{}_{}_{}.tfrecord",
            SUPERVISELY: "supervisely_{}x{}_{}_{}.tfrecord"
        }[dataset]
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
                            tfrecords_dest='/home/aholdobin/tfrecords_v3/',
                            shape=(32, 32),
                            verbose=False):

    # IMPORTANT NOTE: this method implementats creation of the third dataset version (COCO and
    # PASCAL VOC as well as Supervise.ly datasets are used; and from PASCAL VOC dataset selected
    # files where person (people) part is 0 or > .5 in order to reduse numbber of small artifacts)

    def store_chunk(record_name, files, filenames, images_folder, segmentation_folder):
        if len(files) == 0:
            raise ValueError("`files` must contain at least 1 element")
        with tf.io.TFRecordWriter(record_name) as writer:
            for file in files:
                if file and os.path.exists(images_folder + f'{file}.jpg'):
                    y = cv2.resize((np.array(Image.open(
                        segmentation_folder + file + '.png')) == PASCAL_VOC_PERSON_CLASS_ID).astype('float32'), dsize=shape)
                    tmean = y.mean()
                    if 0 < tmean <= .5:
                        continue
                    x = np.array(Image.open(images_folder + file +
                                            '.jpg').resize(shape), dtype='float32')/255.
                    serialized_example = serialize_segmentation_example(x, y.reshape([y.shape[0], y.shape[1], 1]))
                    writer.write(serialized_example)
        filenames.append(record_name)
        gc.collect()

    if not os.path.exists(tfrecords_dest):
        os.mkdir(tfrecords_dest)

    NUMBER_OF_EXAMPLES_IN_FILE = 200 * 10**6 // (shape[0] * shape[1] * 3 * 4)

    filenames = []
    if dataset_home and os.path.exists(dataset_home):

        if verbose:
            print('Processinng and storing PASCAL VOC dataset')

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


def prepare_and_store_tfrecord_dataset(pascal_voc_home='/home/aholdobin/pascalvoc2012/VOCdevkit/VOC2012/',
                              coco_home='/home/aholdobin/faces/',
                              supervisely_home='/home/aholdobin/Supervisely Person Dataset/',
                              tfrecords_dest='/home/aholdobin/tfrecords_v3/',
                              shape=(32, 32),
                              verbose=False):

    # IMPORTANT NOTE: 
    if not os.path.exists(tfrecords_dest):
        os.mkdir(tfrecords_dest)

    ds_train = []
    ds_val = []

    if coco_home and os.path.exists(coco_home):
        preprocess_store_coco_like(ds_train, ds_val, dataset_home=coco_home,
                              tfrecords_dest=tfrecords_dest, shape=shape, verbose=verbose)
    
    if supervisely_home and os.path.exists(supervisely_home):
        preprocess_store_coco_like(ds_train, ds_val, dataset_home=supervisely_home,
                              tfrecords_dest=tfrecords_dest, shape=shape, verbose=verbose, dataset=SUPERVISELY)

    if pascal_voc_home and os.path.exists(pascal_voc_home):
        preprocess_store_pascal(ds_train, ds_val, dataset_home=pascal_voc_home,
                                tfrecords_dest=tfrecords_dest, shape=shape, verbose=verbose)

    if len(ds_train) == 0:
        raise ValueError('Need at least 1 source of data')

    return tfrecords_dest, ds_train, ds_val


def read_tfrecord(serialized_example):
    example = tf.io.parse_single_example(
        serialized_example, FEATURE_DESCRIPTION)

    x = tf.io.parse_tensor(example['x'], out_type=tf.float32)
    y = tf.io.parse_tensor(example['y'], out_type=tf.float32)

    return x, y

def read_augment_tfrecord_dataset(augmentations,
                                  ds_home='/home/aholdobin/tfrecords_v3/',
                                  shape=(32, 32),
                                  batch_size=32,
                                  cache=True,
                                  verbose=False):

    def set_val_shape(x, y):
        x.set_shape((shape[0], shape[1], 3))
        y.set_shape((shape[0], shape[1], 1))
        return x, y
    
    def read_prepare_tfrecordsdataset(ds,
                                      batch_size=None,
                                      augment=None):
        ds = ds.map(read_tfrecord, num_parallel_calls=AUTOTUNE) 
        if batch_size is None:
            raise ValueError()
        if cache:
            ds = ds.cache()
        if augment:
            ds = (ds.shuffle(batch_size*2, seed=0, reshuffle_each_iteration=True)
                  .repeat()
                  .map(augment, num_parallel_calls=AUTOTUNE)
                  .batch(batch_size, drop_remainder=False)
                  .prefetch(AUTOTUNE))
        else:
            ds = (ds.map(set_val_shape, num_parallel_calls=AUTOTUNE)
                  .batch(batch_size, drop_remainder=False)
                  .prefetch(AUTOTUNE))
        return ds
    
    generator = ImageDataGenerator(**augmentations)
    
    def augment(x, y):
        rt = generator.get_random_transform((shape[0], shape[1], 3))
        x_ = generator.apply_transform(x, rt)
        # IMPORTANT: Changing parameters of ImageDataGenerator might require filter out transforms 
        # which changes values of intensities such as brightness, contrast before applying to y
        y_ = generator.apply_transform(y, rt)
        return x_, y_
    
    def augment_tf(x, y):
        x, y = tf.numpy_function(augment, [x, y], (tf.float32, tf.float32))
        x.set_shape((shape[0], shape[1], 3))
        y.set_shape((shape[0], shape[1], 1))
        return x, y
    
    records = os.listdir(ds_home)
    
    shape_str = "{}x{}".format(shape[0], shape[1])
    val_recs = [ds_home + rec for rec in records if rec.find('_val_') > 0 
                    and rec.find(shape_str) >= 0]
    train_recs = [ds_home + rec for rec in records if rec.find('_train_') > 0 
                    and rec.find(shape_str) >= 0]
    
    train_ds = read_prepare_tfrecordsdataset(
        tf.data.TFRecordDataset(train_recs, num_parallel_reads=AUTOTUNE),
        batch_size=batch_size,
        augment=augment_tf
    )
    val_ds = read_prepare_tfrecordsdataset(tf.data.TFRecordDataset(
        val_recs, num_parallel_reads=AUTOTUNE), batch_size=batch_size)
    return train_ds, val_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='data_preparation', description='Preprocess data and transform in tfrecord format')
    parser.add_argument('-p', '--pascal-voc-path',
                        required=False, dest='pascal_voc_path')
    parser.add_argument('-c', '--coco-portraints-path',
                        required=False, dest='coco_portraits_path')
    parser.add_argument('-s', '--supervisely-person-path', 
                        required=False, dest='supervisely_person_path')
    parser.add_argument('-d', '--tfrecord-destination',
                        required=False, dest='tfrecord_destination')
    parser.add_argument('-up', '--use_pascal-voc-path',
                        action='store_true', required=False, dest='use_pascal_voc')
    parser.add_argument('-uc', '--use-coco-portraints-path',
                        action='store_true', required=False, dest='use_coco_portraits')
    parser.add_argument('-us', '--use-supervisely-person-path', 
                        action='store_true', required=False, dest='use_supervisely_person')
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
        tfrecord_destination = '/home/aholdobin/tfrecords_v3/'

    if args.remove_all and os.path.exists(tfrecord_destination):
        import shutil
        shutil.rmtree(tfrecord_destination)

    if args.use_pascal_voc or args.use_coco_portraits or args.use_supervisely_person:
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
        suprevisely_home = '/home/aholdobin/Supervisely Person Dataset/'
        if args.use_supervisely_person:
            if args.supervisely_person_path:
                suprevisely_home = args.supervisely_person_path
        else:
            coco_home = None

        tfrecord_destination,  ds_train, ds_val = prepare_and_store_tfrecord_dataset(
            pascal_voc_home=pascal_voc_home, coco_home=coco_home, supervisely_home=suprevisely_home, tfrecords_dest=tfrecord_destination, shape=input_shape, verbose=args.verbose)
        if args.verbose:
            pprint(ds_train)
            pprint(ds_val)
        gc.collect()
