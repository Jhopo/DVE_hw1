import os
import sys
import glob

import cv2
import numpy as np

from preproc import load_imgs


def threshold_median_intensity(img):
    median = np.median(img)
    new_img = np.array([[255 if i > median else 0 for i in j] for j in img]).astype(np.float32)

    return new_img


def mask(img):
    median = np.median(img)
    mask_img = cv2.inRange(img, median - 10, median + 10)

    return mask_img


def shift(img, xy):
    x, y = xy
    size = img.shape
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted_img = cv2.warpAffine(img, M, (size[1], size[0]))

    return shifted_img


def bit(img):
    bitmap = np.array([[1 if i > 255./2 else 0 for i in j] for j in img])

    return bitmap


def image_alignment(img_list, filename_list, pyramid_level=5):
    # convert to gray-scale
    img_gray = [cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY) for img in img_list]

    # thrsesholding and mask
    imgs = [threshold_median_intensity(img) for img in img_gray]
    mask_img = mask(img_gray[0])

    # alignment
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    img1 = imgs[0]
    size = img1.shape

    shift_list = [(0, 0)]  # first image needs no shifting
    for idx in range(1, len(imgs)):
        print (filename_list[idx] + '...', end='')
        img2 = imgs[idx]

        dx, dy = 0, 0
        for i in range(pyramid_level):
            ratio = pyramid_level - 1 - i

            width, height = int(size[1] / (2**ratio)), int(size[0] / (2**ratio))
            img1_r = bit(cv2.resize(img1, (width, height)))
            img2_r = cv2.resize(img2, (width, height))
            mask_img_r = bit(cv2.resize(mask_img, (width, height)))

            diff_list = []
            for d_idx in range(len(directions)):
                img2_rs = bit(shift(img2_r, (directions[d_idx][0] + dx, directions[d_idx][1] + dy)))
                diff = np.bitwise_and(np.bitwise_xor(img1_r, img2_rs), mask_img_r)
                diff_list.append(np.sum(diff))

            min_direction = directions[np.argmin(diff_list)]
            dx, dy = dx + 2 * min_direction[0], dy + 2 * min_direction[1]

        shift_list.append((dx, dy))
        print ('done', '(dx={}, dy={})'.format(dx, dy))

    return shift_list


def save_alligned_imgs(img_list, filename_list, shift_list, output_dir):
    for (i, img) in enumerate(img_list):
        filename = os.path.join(output_dir, filename_list[i])
        shifted_img = shift(img, shift_list[i])
        cv2.imwrite(filename, shifted_img)


if __name__ == '__main__':
    source_dir = '../flower/'
    output_dir = '../output/'

    img_list, filename_list, shutter_speed = load_imgs(source_dir)
    shift_list = image_alignment(img_list, filename_list)
    save_alligned_imgs(img_list, filename_list, shift_list, output_dir)
