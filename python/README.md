# About

AR for video conferencing mobile beta.

The current part of the application focuses on implementing a module that enables creation and training a semantic segmentation model for a person's background detection during conference video calls from mobile devices. Such a model then can be used by the main Android application.

Check the [presentation](https://docs.google.com/presentation/d/1D0ka75MLTviAcFQplPsNB-FRha02DDwYiyKvj93bY9E/edit?pli=1#slide=id.g79b1116e15d7db6_1) regarding related work and the implementation. Check CLI in the [Scripts](#Scripts) section.

# Requirements

Software:
* python3
* tensorflow 2.3
* keras-unet 0.1.2 
* opencv-python 4.2

Hardware:
* for training:
  * Nvidia GPU 8GB RAM (CPU training is possible but will take 3-5 weeks instead of 15 hours)
  * Cuda 10.1
  * HDD 40GB
  * RAM minimum 16GB, recommended 32GB
  * Windows/Linux
* for segmentation:
  * no limits

Image matting datasets are available:
* Adobe Deep Image Matting Dataset Follow the instructions to contact the author for the dataset. https://sites.google.com/view/deepimagematting
* [MSCOCO 2014 Train images](http://images.cocodataset.org/zips/train2014.zip)
* PASCAL VOC 
  * [VOC challenge 2012 training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)
* [Supervise.ly Person Dataset](https://supervise.ly/explore/projects/supervisely-person-dataset-23304/datasets)

Datasets for backgroud altering:
* [CMP facade DB base](http://cmp.felk.cvut.cz/~tylecr1/facade/)
* [DAtaset for QUestion Answering on Real-world images](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/visual-turing-challenge)

# Scripts

There are 3 main modules that create the whole pipeline:

*supervisely_preprocessing.py* -> *data_preparation.py* -> *AUnetBackgroundRemoval.py*

* At first, images from Supervise.ly Person Dataset must be preprocessed into the Portrait dataset format (creation 2 folders with images and masks);
* Then we need to create Datasets into TFRecord format to use at the next stage;
* And the last step is to train the model using created TFRecordDataset.

**supervisely_preprocessing.py**

Usage:
```
$ conda activate semsegm
$ cd %APPLICATION_PATH%/python
$ python supervisely_preprocessing.py -h

usage: supervise.ly_preprocessor [-h] [-s SOURCE_PATH] -d DESTINATION

Extract examples from Supervise.ly Person Dataset (preprocessing).

optional arguments:
  -h, --help            show this help message and exit
  -s SOURCE_PATH, --source-path SOURCE_PATH
                        specify the path to the Supervise.ly Person Dataset.
                        Required if different from the default.
  -d DESTINATION, --destination-path DESTINATION
                        specify the destination to store the processed
                        dataset.
```
**data_preparation.py**

Usage:
```
$ conda activate semsegm
$ cd %APPLICATION_PATH%/python
$ python data_preparation.py -h

usage: data_preparation [-h] [-p PASCAL_VOC_PATH] [-c COCO_PORTRAITS_PATH]
                        [-s SUPERVISELY_PERSON_PATH] [-d TFRECORD_DESTINATION]
                        [-up] [-uc] [-us] [-sd SIDE] [-v VERBOSE]
                        [--remove-all-previous-results]

Preprocess data and transform in the TFrecord format for
AUnetBackgroundRemoval.

optional arguments:
  -h, --help            show this help message and exit
  -p PASCAL_VOC_PATH, --pascal-voc-path PASCAL_VOC_PATH
                        specify the path to the PascalVOC2012 dataset.
  -c COCO_PORTRAITS_PATH, --coco-portraints-path COCO_PORTRAITS_PATH
                        specify the path to the Portraits dataset.
  -s SUPERVISELY_PERSON_PATH, --supervisely-person-path SUPERVISELY_PERSON_PATH
                        specify the path to the Supervise.ly Person dataset
                        (processed with supervisely_preprocessing.py).
  -d TFRECORD_DESTINATION, --tfrecord-destination TFRECORD_DESTINATION
                        specify the destination of creating TFRecords.
  -up, --use_pascal-voc
                        to process the PascalVOC2012 dataset.
  -uc, --use-coco-portraints
                        to process the Portrait dataset.
  -us, --use-supervisely-person
                        to process the Supervise.ly Person dataset.
  -sd SIDE, --side SIDE
                        specify the side of downscaled image.
  -v VERBOSE, --verbose VERBOSE
  --remove-all-previous-results
                        delete all files under the directory specified on
                        `-d`.
```
**AUnetBackgroundRemoval.py**

Usage:
```
$ conda activate semsegm
$ cd %APPLICATION_PATH%/python
$ python AUnetBackgroundRemoval.py -husage: AUnetBackgroundRemoval [-h] [-p PASCAL_VOC_PATH]
                              [-c COCO_PORTRAITS_PATH] [-up] [-uc]
                              [-cf CHECKPOINTS_FOLDER] [-if INTERIOR_FOLDER]
                              [-ef EXTERIOR_FOLDER] [-r TFRECORDS_PATH]
                              [-e EPOCHS] [-b BATCH_SIZE] [-f FILTERS]
                              [-sd SIDE] [-n NUMBER_OF_EXAMPLES] [-ch]
                              [-v VERBOSE] [--remove-all-previous-results]
                              [-l] [--restore]

Attention UNet parametrizable model trainer for background removal.

optional arguments:
  -h, --help            show this help message and exit
  -p PASCAL_VOC_PATH, --pascal-voc-path PASCAL_VOC_PATH
                        specify the path to the PascalVOC2012 dataset. Used
                        for in-memory training (DS v2).
  -c COCO_PORTRAITS_PATH, --coco-portraints-path COCO_PORTRAITS_PATH
                        specify the path to the Portraits dataset. Used for
                        in-memory training (DS v1-2).
  -up, --use-pascal-voc
                        to enable usage of the PascalVOC2012 dataset. Used for
                        in-memory training way (DS v2). If path differs from
                        default, `-p` argument is required.
  -uc, --use-coco-portraints
                        to enable usage of the PascalVOC2012 dataset. Used for
                        in-memory training way (DS v1-2). If path differs from
                        default, `-c` argument is required.
  -cf CHECKPOINTS_FOLDER, --checkpoints-folder CHECKPOINTS_FOLDER
                        specify the folder for intermediate weights storage
                        and final quantized model storage. Default folder:
                        `training_grid`.
  -if INTERIOR_FOLDER, --interior-folder INTERIOR_FOLDER
                        specify the folder of interior dataset.
  -ef EXTERIOR_FOLDER, --exterior-folder EXTERIOR_FOLDER
                        specify the folder of exterior dataset.
  -r TFRECORDS_PATH, --tfrecords-path TFRECORDS_PATH
                        specify the path to folder containing tfrecords
                        generated by `data_preparation.py`. Used with the
                        dataset v3 and next iterations.
  -e EPOCHS, --epochs EPOCHS
                        specify the total number of training epochs needed.
                        Default: 1000.
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        specify the batch size. Default: 8.
  -f FILTERS, --filters FILTERS
                        specify number of filters in the first layer of the
                        network. Default: 8.
  -sd SIDE, --side SIDE
                        specify the side of input size (in case of sqare input
                        size). Default: 32.
  -n NUMBER_OF_EXAMPLES, --number-of-examples NUMBER_OF_EXAMPLES
                        specify number of examples of datased. Used with `-r`.
                        Needed to calculate number of batches for training and
                        validation. Default: 27781 (DS V3).
  -ch, --cache          specify if datasets caching is needed. Should
                        accelerate model training if amount of memory (RAM) is
                        enough.
  -v VERBOSE, --verbose VERBOSE
  --remove-all-previous-results
                        used to remove all previous results of training in the
                        `checkpoints_folder` if any. Deletes the folder by the
                        path provided in the `checkpoints_folder`
  -l, --generate-tflite-only
                        to generate tflite models from the last checkpoint
                        without training.
  --restore             to continue model training after crash or the end of
                        training on initial number of epochs.
```