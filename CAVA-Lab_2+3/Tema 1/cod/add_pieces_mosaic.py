from parameters import *
import numpy as np
import pdb
import timeit


def get_mean_color_small_images(params: Parameters):
    N, H, W, C = params.small_images.shape
    mean_color_pieces = np.zeros((N, C))

    for i in range(N):
        current_image = params.small_images[i]
        for ch in range(C):
            mean_color_pieces[i, ch] = np.float32(current_image[:, :, ch].mean())

    return mean_color_pieces


def get_distance_between_means(mean_color_patch, mean_color_pieces):
    dist = np.linalg.norm(mean_color_pieces - mean_color_patch, axis=1)
    return np.argmin(dist)


def get_mask_hexagon(H, W):
    ind_start = int(np.floor(W / 3))
    ind_end = int(np.ceil(2 * W / 3))
    ind_mij = int(H/2)

    mask = np.zeros((H, W, 1))
    for i in range(ind_mij):
        for j in range(max(0, ind_start), min(ind_end, W)):
            mask[i, j], mask[H - i - 1, j] = 1, 1
        ind_start -= 1
        ind_end += 1
    return mask


def add_pieces_grid(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    N, H, W, C = params.small_images.shape
    h, w, c = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    if params.criterion == 'aleator':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                small_image_index = np.random.randint(0, N, size=1)
                img_mosaic[i * H:(i + 1) * H, j * W:(j + 1) * W, :] = params.small_images[small_image_index]
        print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))
    elif params.criterion == 'distantaCuloareMedie':
        mean_color_pieces = get_mean_color_small_images(params)
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                patch = params.image_resized[i * H:(i + 1) * H, j * W:(j + 1) * W, :]
                mean_color_patch = np.mean(patch, axis=(0, 1))
                minimum_image = get_distance_between_means(mean_color_patch, mean_color_pieces)
                img_mosaic[i * H:(i + 1) * H, j * W:(j + 1) * W, :] = params.small_images[minimum_image]
        print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j) / num_pieces))
    else:
        print('Error! unknown option %s' % params.criterion)
        exit(-1)

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


# @todo
def add_pieces_random(params: Parameters):
    start_time = timeit.default_timer()
    end_time = timeit.default_timer()
    print('running time:', (end_time - start_time), 's')
    return None


def add_pieces_hexagon(params: Parameters):
    start_time = timeit.default_timer()
    N, H, W, C = params.small_images.shape
    h, w, c = params.image_resized.shape
    mask = get_mask_hexagon(H, W)
    mean_color_pieces = get_mean_color_small_images(params)
    bigger_image = np.zeros((h + 2 * H, w + 2 * W, c), np.uint8)
    bigger_image[H:-H, W:-W] = params.image_resized.copy()
    img_mosaic = np.zeros(bigger_image.shape, np.uint8)

    for i in range(H // 2, bigger_image.shape[0] - H, H):
        for j in range(0, bigger_image.shape[1] - W, W + W // 3):
            patch = bigger_image[i:i + H, j:j + W, :]
            mean_color_patch = np.mean(patch, axis=(0, 1))
            minimum_image = get_distance_between_means(mean_color_patch, mean_color_pieces)
            img_mosaic[i:i + H, j:j + W, :] = (img_mosaic[i:i + H, j:j + W, :] * (1 - mask)) + \
                                              mask * params.small_images[minimum_image]

    for i in range(0, bigger_image.shape[0] - H, H):
        for j in range(2 * W // 3, bigger_image.shape[1] - W, W + W // 3):
            patch = bigger_image[i:i + H, j:j + W, :]
            mean_color_patch = np.mean(patch, axis=(0, 1))
            minimum_image = get_distance_between_means(mean_color_patch, mean_color_pieces)
            img_mosaic[i:i + H, j:j + W, :] = (img_mosaic[i:i + H, j:j + W, :] * (1 - mask)) + \
                                              mask * params.small_images[minimum_image]
    img_mosaic = img_mosaic[H:-H, W:-W].copy()
    end_time = timeit.default_timer()
    print('running time:', (end_time - start_time), 's')
    return img_mosaic
