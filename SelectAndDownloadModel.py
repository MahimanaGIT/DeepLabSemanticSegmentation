#!/usr/bin/env python3
'''
Run this script using python3 because of the urllib script
'''

import os
import shutil
import urllib.request

import tarfile

MODEL_NAME = 'mobilenetv2_coco_voctrainval'

#creating the downlaod link for the model selected
_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'

# Some models to train on
_MODEL_URLS = {
    'mobilenetv2_coco_voctrainaug':
        'deeplabv3_mnv2_pascal_train_aug_2018_01_29',
    'mobilenetv2_coco_voctrainval':
        'deeplabv3_mnv2_pascal_trainval_2018_01_29',
    'xception_coco_voctrainaug':
        'deeplabv3_pascal_train_aug_2018_01_04',
    'xception_coco_voctrainval':
        'deeplabv3_pascal_trainval_2018_01_04',
}
_TARBALL_NAME = 'deeplab_model'

#the distination folder where the model will be saved
#change this if you have a different working dir
DEST_DIR = './models/'

# Name of the object detection model to use.
MODEL = _MODEL_URLS[MODEL_NAME]

#selecting the model
MODEL_FILE = MODEL + '.tar.gz'


#checks if the model has already been downloaded, download it otherwise
if not (os.path.exists(DEST_DIR + MODEL_FILE)):
    print('Downloading Model')
    urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + MODEL_FILE, DEST_DIR + MODEL_FILE)

print('Extracting model')

#unzipping the model and extracting its content
tar = tarfile.open(DEST_DIR + MODEL_FILE)
tar.extractall(path = DEST_DIR)
tar.close()

# creating an output file to save the model while training
os.remove(DEST_DIR + MODEL_FILE)