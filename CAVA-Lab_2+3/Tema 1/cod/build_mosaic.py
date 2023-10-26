import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pdb

from add_pieces_mosaic import *
from parameters import *


def load_pieces(params: Parameters):
    files = filter(lambda file: file.endswith(params.image_type), os.listdir(params.small_images_dir))
    images = [cv.imread(os.path.join(params.small_images_dir, file)) for file in files]
    if params.show_small_images:
        for i in range(10):
            cv.imshow("Show image", images[i])
            cv.waitKey(0)
            cv.destroyAllWindows()
    params.small_images = np.array(images)


def compute_dimensions(params: Parameters):
    hs, ws, _ = params.small_images[0].shape
    h, w, _ = params.image.shape

    params.num_pieces_vertical = int(np.round((ws * params.num_pieces_horizontal * h) / (w * hs)))
    params.image_resized = cv.resize(params.image, (
        ws * params.num_pieces_horizontal, hs * params.num_pieces_vertical
    ))
    print(params.image_resized.shape)


def build_mosaic(params: Parameters):
    load_pieces(params)
    compute_dimensions(params)

    img_mosaic = None
    if params.layout == 'caroiaj':
        if params.hexagon is True:
            img_mosaic = add_pieces_hexagon(params)
        else:
            img_mosaic = add_pieces_grid(params)
    elif params.layout == 'aleator':
        img_mosaic = add_pieces_random(params)
    else:
        print('Wrong option!')
        exit(-1)

    return img_mosaic

