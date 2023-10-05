# 1.1 -> cv2 instalat in virtualenv
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


# 1.5
def dev_show_image(image_name: str, show_image: np.array):
    cv.imshow(image_name, show_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 1.2 importarea bibliotecii
print(f'Version OpenCV: {cv.__version__}')

# 1.3 imagine RGB citire - afisare
butterfly_image = cv.imread('images/butterfly.jpeg')
print(f'Shape of the image: {butterfly_image.shape}')
dev_show_image("RGB Butterfly", butterfly_image)

# 1.4 imagine GrayScale citire - afisare
butterfly_greyscale_image = cv.imread('images/butterfly.jpeg', cv.IMREAD_GRAYSCALE)
print(f'Shape of the grayscale image: {butterfly_greyscale_image.shape}')
dev_show_image("Grayscale Butterfly", butterfly_greyscale_image)

# 1.6
image = cv.resize(cv.cvtColor(cv.imread('images/football.jpg'), cv.COLOR_BGR2GRAY), (100, 100))
print(f'Shape of the image: {image.shape} | Pixels inside the image:\n {image}')

sorted_array = np.sort(image.flatten())
print(f'Sorted array shape: {sorted_array.shape} | Print sorted array:\n {sorted_array}')
plt.plot(np.arange(len(sorted_array)), sorted_array)
plt.show()

copy_quarter_image = image[image.shape[0] // 2:, image.shape[0] // 2:].copy()
dev_show_image("Quarter Image Right-Down", copy_quarter_image)

median = np.median(image)
binary_image = image.copy()
binary_image[binary_image < median], binary_image[binary_image >= median] = 0, 255
dev_show_image("Binary image", binary_image)

mean = image.mean()
c_image = image.copy() - mean
c_image[c_image < 0] = 0
c_image = np.uint8(c_image) # FOARTE IMPORTANT IMAGINEA NU CONTINE NUMERE NEGATIVE!!!
dev_show_image("C image", c_image)

minimum = image.min()
print(np.where(image == minimum))


# 1.7
def colectie_imagini(dir_path):
    files = filter(lambda file_name: file_name.endswith(".jpg"), os.listdir(dir_path))
    image_collection = np.array([
        cv.imread(f'{os.path.join(dir_path, path_image)}') for path_image in files
    ])
    print(f'Shape of the image collection: {image_collection.shape}')

    mean_img_color = np.uint8(np.mean(image_collection, axis=0))
    print(f'Shape of the mean_img_color: {mean_img_color.shape}')
    dev_show_image("Mean color image", mean_img_color)

    gray_images = np.array(list(map(
        lambda color_image: cv.cvtColor(color_image, cv.COLOR_BGR2GRAY), image_collection
    )))
    print(f'Shape of the gray image collection: {gray_images.shape}')
    mean_img_gray = np.uint8(np.mean(gray_images, axis=0))
    dev_show_image("Mean gray image", mean_img_gray)

colectie_imagini('images/set2')

# 1.8
img = cv.imread("images/butterfly.jpeg")
window_size, number_samples = 20, 500

img_crop = img[250:250 + window_size, 250:250 + window_size, :].copy()
y, x = np.random.randint(0, img.shape[0] - window_size, size=number_samples), \
    np.random.randint(0, img.shape[1] - window_size, size=number_samples)

dist = np.zeros(number_samples)
for i in range(number_samples):
    image_patch = img[y[i]:y[i] + window_size, x[i]:x[i] + window_size, :].copy()
    dist[i] = np.linalg.norm(img_crop - image_patch)
minimal_dist = np.argmin(dist)

img[250:250 + window_size, 250:250 + window_size, :] = \
    img[minimal_dist:minimal_dist + window_size, minimal_dist:minimal_dist + window_size, :].copy()
dev_show_image("Altered image", img)
