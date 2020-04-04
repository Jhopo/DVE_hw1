import os
import sys
import glob

import cv2
import numpy as np

from preproc import load_imgs


class HDRSolver():
    def __init__(self, source_dir):
        np.random.seed(1124)

        # load images
        self.img_list, self.filename_list, self.shutter_speed = load_imgs(source_dir)
        self.height, self.width, self.channel = self.img_list[0].shape
        self.sampled_pixels = self.sample_pixels()

        # assumes
        self.Z_min = 0
        self.Z_max = 0

        # arguments
        self.ZBm, self.ZGm, self.ZRm = self.build_Z_matrix()
        self.Bm = np.log(self.shutter_speed)
        self.Lm = 40
        self.Wm = self.build_W_matrix()



    def sample_pixels(self, sample_num=50, method='random'):
        pixels_list = []

        if method=='random':
            while len(pixels_list) < sample_num:
                w = np.random.randint(self.width)
                h = np.random.randint(self.height)

                if (w, h) not in pixels_list:
                    pixels_list.append((w, h))

        return pixels_list


    def build_Z_matrix(self):
        ZBm, ZGm, ZRm = [], [], []
        for channel_i, Z in enumerate([ZBm, ZGm, ZRm]):
            for (w, h) in self.sampled_pixels:
                z = []
                for img in self.img_list:
                    z.append(img[h][w][channel_i])
                Z.append(z)

        return np.array(ZBm), np.array(ZGm), np.array(ZRm)


    def build_W_matrix(self):
        pass


if __name__ == '__main__':
    source_dir = '../exposures/'

    solver = HDRSolver(source_dir)
