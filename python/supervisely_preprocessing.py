import json
import base64
import os
import argparse
import zlib
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from mtcnn import MTCNN


DS_FMTS = ('{dsf}/img/', '{dsf}/ann/')
FACE_DET = None
EPS = 1/32
ASPECT_RATIO = 16/9


# Supervisely Format, Bitmap [https://docs.supervise.ly/data-organization/import-export/supervisely-format#bitmap]
def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask

def get_ds_folders(dataset_home):
    return [folder for folder in os.listdir(dataset_home) if folder[:2]=='ds']

def get_files(dataset_home):
    folders = get_ds_folders(dataset_home)
    files_fmt = []
    for folder in folders:
        if folder[:2]=='ds':
            path = DS_FMTS[0].format(dsf=folder)
            files_fmt += list(map(lambda x: dataset_home+folder + '/{}/'+x, os.listdir(dataset_home+path)))
    return files_fmt

def get_paths_from_format(file_fmp):
    return (file_fmp.format('img'), file_fmp.format('ann') + '.json')

def paste_bitmap_meta(zero_mask, obj):
    if 'bitmap' not in obj:
        raise ValueError('Wrong segmentation type')
    segm = base64_2_mask(obj['bitmap']['data'])
    [x, y] = obj['bitmap']['origin']
    zero_mask[y:y + segm.shape[0], x:x + segm.shape[1]][segm] = segm[segm]
    meta = {
        'shape': segm.shape,
        'coords': [x, y],
        'type': 'bitmap',
    }
    return meta

def paste_polygon_meta(zero_mask, obj):
    if 'points' not in obj:
        raise ValueError('Wrong segmentation type')
    exterior = np.array(obj['points']['exterior'])
    cv2.fillPoly(zero_mask, [exterior], 1)
    for inter in obj['points']['interior']:
        cv2.fillPoly(zero_mask, [np.array(inter)], 0)
    left_top = exterior.min(axis=0)
    right_bottom = exterior.max(axis=0)
    shape = right_bottom - left_top; shape = shape[1], shape[0]
    meta = {
        'shape': shape,
        'coords': left_top,
        'type': 'polygon',
    }
    return meta

PASTE_WITH_META = {
    'bitmap' : paste_bitmap_meta,
    'polygon' : paste_polygon_meta
}

def get_masks_meta_from_json_annotation(mask_json_path):
    with open(mask_json_path) as f:
        mask_data = json.load(f)
    
    meta_people = []
    
    height = mask_data['size']['height']
    width = mask_data['size']['width']
    mask = np.zeros([height, width], dtype='float32')
    for obj in mask_data['objects']:
        if obj['classTitle'] in ['person_bmp', 'person_poly']:
            meta_people.append(PASTE_WITH_META[obj['geometryType']](mask, obj))
    return mask, meta_people


def cropped_example(image, mask, meta):
    
    def available_borders(image, meta):
        im_shape = image.shape[:2]
        segm_bbox = meta['coords'], meta['shape']
        top_left = segm_bbox[0][1], segm_bbox[0][0]
        bottom_right = im_shape - np.array([segm_bbox[0][1] + segm_bbox[1][0], segm_bbox[0][0]+segm_bbox[1][1]])
        return np.concatenate([top_left, bottom_right])
        
    def augment_segmentation(image, mask, meta):
        coords = meta['coords']
        shape = meta['shape']
        segmentation_image = image[coords[1]:coords[1]+shape[0], coords[0]:coords[0]+shape[1]]
        segmentation_mask = mask[coords[1]:coords[1]+shape[0], coords[0]:coords[0]+shape[1]]
        
        if segmentation_image.shape[0]/ASPECT_RATIO + EPS < segmentation_image.shape[1]:
            
            needed_height = int(np.ceil(segmentation_mask.shape[1] * ASPECT_RATIO))
            ex_image = np.zeros([needed_height, segmentation_image.shape[1], 3])
            ex_mask = np.zeros([needed_height, segmentation_image.shape[1]])
            
            if needed_height > image.shape[0]:
                ex_image[needed_height - image.shape[0]:, :] = image[:, coords[0]:coords[0]+shape[1]]
                ex_mask[needed_height - image.shape[0]:, :] = mask[:, coords[0]:coords[0]+shape[1]]
            else:
                ab = available_borders(image, meta)
                if ab[0] > needed_height - segmentation_image.shape[0]:
                    bot = 0
                    top = needed_height - segmentation_image.shape[0]
                else:
                    top = ab[0]
                    bot = needed_height - ab[0] - segmentation_image.shape[0]
                ex_image[:, :] = image[coords[1]-top:coords[1]+shape[0]+bot, coords[0]:coords[0]+shape[1]] 
                ex_mask[:, :] = mask[coords[1]-top:coords[1]+shape[0]+bot, coords[0]:coords[0]+shape[1]] 
            
            return ex_image, ex_mask
        else:
            needed_width = int(np.ceil(segmentation_mask.shape[0] / ASPECT_RATIO))
            ex_image = np.zeros([segmentation_image.shape[0], needed_width, 3])
            ex_mask = np.zeros([segmentation_image.shape[0], needed_width])
            
            if needed_width > image.shape[1]:
                ex_image[:, (needed_width - image.shape[1])//2:(needed_width - image.shape[1])//2+image.shape[1]] = image[coords[1]:coords[1]+shape[0],:]
                ex_mask[:, (needed_width - image.shape[1])//2:(needed_width - image.shape[1])//2+image.shape[1]] = mask[coords[1]:coords[1]+shape[0],:]
            else:
                ab = available_borders(image, meta)
                if ab[1] > needed_width - segmentation_image.shape[1]:
                    rgt = 0
                    lft = needed_width - segmentation_image.shape[1]
                else:
                    lft = ab[1]
                    rgt = needed_width - ab[1] - segmentation_image.shape[1]
                ex_image[:, :] = image[coords[1]:coords[1]+shape[0], coords[0]-lft:coords[0]+shape[1]+rgt] 
                ex_mask[:, :] = mask[coords[1]:coords[1]+shape[0], coords[0]-lft:coords[0]+shape[1]+rgt] 
            
            return ex_image, ex_mask

    coords = meta['coords']
    shape = meta['shape']
    height, width = shape[0], shape[1]
    # filtre by size
    if min(width, height) < 128:
        return None
    if abs(height/width - ASPECT_RATIO) < EPS:
        return (
            image[coords[1]:coords[1]+shape[0], coords[0]:coords[0]+shape[1]],
            mask[coords[1]:coords[1]+shape[0], coords[0]:coords[0]+shape[1]]
        )
    else:
        return augment_segmentation(image, mask, meta)


def prepare_example(file_fmt):
    # Get pathes to an image and its mask with metadata and load them
    paths = get_paths_from_format(file_fmt)
    with open(paths[1]) as f:
        mask_data = json.load(f)
    image = np.array(Image.open(paths[0]))
    if len(image.shape) == 2:
        image= np.stack([image]*3, axis=2)
    elif len(image.shape) != 3:
        return None

    # Create mask from metadata
    height = mask_data['size']['height']
    width = mask_data['size']['width']
    mask = np.zeros([height, width], dtype='float32')
    meta = None
    
    for obj in mask_data['objects']:
        if obj['classTitle'] in ['person_bmp', 'person_poly']:
            # If the second person appears, break, as we use only images with singe person 
            if meta:
                return None
            meta = PASTE_WITH_META[obj['geometryType']](mask, obj)
            
    coords = meta['coords']
    shape = meta['shape']
    
    # Check if we can find the face
    try:
        det = FACE_DET.detect_faces(image[coords[1]:coords[1]+shape[1], coords[0]:coords[0]+shape[0]])
    except Exception as ex:
        print(ex, file_fmt)
        return None
    if len(det) > 0:
        return cropped_example(image, mask, meta)
    return None

def store_example(image, mask, index, destination):
    mask_ = np.zeros((image.shape[0], image.shape[1], 4))
    image = cv2.cvtColor(image.astype('float32'), cv2.COLOR_RGB2BGR)
    mask_[:,:,:3] = image
    mask_[:,:,3] = (mask*255).astype('uint8')
    if not os.path.exists(destination):
        os.mkdir(destination)
    image_path = os.path.join(destination, 'images')
    mask_path = os.path.join(destination, 'masks')
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    if not os.path.exists(mask_path):
        os.mkdir(mask_path)
    cv2.imwrite(os.path.join(image_path, "{:04}.jpg".format(index)), image)
    cv2.imwrite(os.path.join(mask_path, "{:04}.png".format(index)), mask_)


def initialize_facedet(initialize_gpu=False):
    if initialize_gpu: 
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # From [TensorFlow guide][1]https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                first_gpu = gpus[0]
                tf.config.experimental.set_virtual_device_configuration(first_gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)])
            except RuntimeError as e:
                print(e)
    
    global FACE_DET
    FACE_DET = MTCNN()


def main():
    parser = argparse.ArgumentParser(
        prog='supervise.ly_preprocessor', description='Extract examples from Supervise.ly '
                                                      'Person Dataset (preprocessing).')
    parser.add_argument('-s', '--source-path',
                        required=False, dest='source_path',
                        help="specify the path to the Supervise.ly Person Dataset. "
                             "Required if different from the default.")
    parser.add_argument('-d', '--destination-path',
                        required=True, dest='destination',
                        help="specify the destination to store the processed dataset.")
    args = parser.parse_args()

    # preparing environment
    initialize_facedet(initialize_gpu=True)
    DATASET_HOME = '/Users/aholdobin/projects/data/Supervisely Person Dataset/'
    if args.source_path:
        DATASET_HOME = args.source_path
    ds_folders = get_ds_folders(DATASET_HOME)
    files = get_files(DATASET_HOME)
    
    index = 0
    for file_fmt in files:
        example = prepare_example(file_fmt=file_fmt)
        if example:
            store_example(example[0], example[1], index, args.destination)
            index += 1
    print("Total number of examples: {}".format(index))

if __name__ == "__main__":
    import sys
    sys.exit(main())
