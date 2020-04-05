import os
import sys
import glob
import math

import cv2
import numpy as np
import seaborn as sb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preproc import load_imgs


class HDRSolver():
    def __init__(self, source_dir):
        #np.random.seed(1124)

        # load images
        self.img_list, self.filename_list, self.shutter_speed = load_imgs(source_dir)
        self.height, self.width, self.channel = self.img_list[0].shape
        self.sampled_pixels = self.sample_pixels()

        # assumes
        self.Z_min = 0
        self.Z_max = 255
        self.Z_mid = (self.Z_min + self.Z_max) / 2

        # arguments
        self.ZBm, self.ZGm, self.ZRm = self.build_Z_matrix()
        self.Bm = np.log(self.shutter_speed)
        self.Lm = 40


    def solve(self):
        Zm = [self.ZBm, self.ZGm, self.ZRm]
        channel_names = ['Blue', 'Green', 'Red']

        self.G = []
        for c_i in range(self.channel):
            Z = Zm[c_i]

            n = 256
            A = np.zeros((Z.shape[0]*Z.shape[1]+n+1, n+Z.shape[0]))
            b = np.zeros((A.shape[0], 1))

            k = 0
            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    wij = self.weighting_func(Z[i][j]+1)
                    A[k][Z[i][j]] = wij
                    A[k][n+i] = -wij
                    b[k][0] = wij * self.Bm[j]
                    k += 1

            A[k][129] = 1
            k += 1

            for i in range(n-1):
                A[k][i] = self.Lm * self.weighting_func(i+1)
                A[k][i+1] = -2 * self.Lm * self.weighting_func(i+1)
                A[k][i+2] = self.Lm * self.weighting_func(i+1)
                k += 1

            x, residuals, rank, s = np.linalg.lstsq(A, b)
            g = x[0:n]
            lE = x[n:]

            self.G.append(g)

        np.save('../saved_G', np.array(self.G))


    def load_G(self):
        self.G = np.load('../saved_G.npy')


    def plot_response_curve(self):
        color = ['b', 'g', 'r']

        for c_i in range(self.channel):
            plt.plot(np.arange(len(self.G[c_i])), self.G[c_i], color[c_i]+'x')

        plt.xlabel('pixel value')
        plt.ylabel('log exposure')
        plt.savefig('../reponse_curve.png', bbox_inches='tight', dpi=300)
        plt.gcf().clear()


    def create_radiance_map(self):
        color = ['blue', 'green', 'red']

        hdr = []
        for c_i in range(self.channel):
            hdr_img = np.zeros((self.height, self.width))
            for h in range(self.height):
                for w in range(self.width):
                    numerator, denominator = 0., 0.
                    for j, img in enumerate(self.img_list):
                        z = img[h][w][c_i]
                        numerator += self.weighting_func(z) * (self.G[c_i][z] - self.Bm[j])
                        denominator += self.weighting_func(z)

                    hdr_img[h][w] = numerator / denominator
            hdr.append(hdr_img)

            # plot
            heat_map = sb.heatmap(hdr_img)
            plt.axis('off')
            plt.savefig('../heatmap_{}.png'.format(color[c_i]), bbox_inches='tight', dpi=300)
            plt.gcf().clear()

        return hdr


    def sample_pixels(self, sample_num=100, method='random'):
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


    def weighting_func(self, z):
        return z - self.Z_min if z <= self.Z_mid else self.Z_max - z



if __name__ == '__main__':
    source_dir = '../output/'
    eval_dir = '../exposures/'

    if sys.argv[1] == 'solve':
        solver = HDRSolver(source_dir)
        solver.solve()
        solver.plot_response_curve()
        hdr = solver.create_radiance_map()

    if sys.argv[1] == 'eval':
        solver = HDRSolver(eval_dir)
        solver.load_G()
        hdr = solver.create_radiance_map()
