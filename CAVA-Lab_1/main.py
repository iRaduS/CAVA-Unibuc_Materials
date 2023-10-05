# 1.1 -> cv2 instalat in virtualenv
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


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
