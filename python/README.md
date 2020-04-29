# Requirements

Software:
* python3
* tensorflow 2.1

Hardware:
* for training:
  * Nvidia GPU 4GB RAM
  * Cuda 10.1
  * HDD 40GB
  * RAM minimum 16GB, recommended 32GB
  * Windows/Linux
* for segmentation:
  * no limits

Image matting datasets are available:
* Adobe Deep Image Matting Dataset Follow the instructions to contact the author for the dataset. https://sites.google.com/view/deepimagematting
* MSCOCO 2014 Train images http://images.cocodataset.org/zips/train2014.zip
* PASCAL VOC 
  * VOC challenge 2008 training/validation data http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar
  * The test data for the VOC2008 challenge http://host.robots.ox.ac.uk/pascal/VOC/voc2008/index.html#testdata
  
 In my experiments I used data set from MSCOCO http://images.cocodataset.org/zips/train2014.zip (12,6GB). It is enough to unzip dataset into faces directory(masks and images).

