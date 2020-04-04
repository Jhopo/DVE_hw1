import os
import sys
import glob

import cv2
import numpy as np


def load_imgs(data_dir='../flower/'):
    file_names = sorted(glob.glob(os.path.join(data_dir, '*.JPG'))) + sorted(glob.glob(os.path.join(data_dir, '*.jpg')))
    img_list = []
    filename_list = []
    for file_name in file_names:
        img = cv2.imread(file_name)
        img_list.append(img)
        filename_list.append(file_name.split('/')[-1])

    shutter_speed = [1/float(2**(11-i)) for i in range(9)]

    return img_list, filename_list, shutter_speed
