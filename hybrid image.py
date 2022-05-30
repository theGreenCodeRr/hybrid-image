import numpy
import cv2
import math
from numpy.fft import fft2, ifft2, fftshift, ifftshift


def gaussian_blur(rows, cols, sigma, high_pass=True):
    centerI = int(rows / 2) + 1 if rows % 2 == 1 else int(rows / 2)
    centerJ = int(cols / 2) + 1 if cols % 2 == 1 else int(cols / 2)

    def gaussian(i, j):
        coefficient = math.exp(-1.0 * ((i - centerI) **
                               2 + (j - centerJ) ** 2) / (2 * sigma ** 2))
        return 1 - coefficient if high_pass else coefficient

    return numpy.array([[gaussian(i, j) for j in range(cols)] for i in range(rows)])


# get conv value
def fft(image_matrix, filter):
    shiftedDFT = fftshift(fft2(image_matrix))

    filteredDFT = shiftedDFT * filter
    return ifft2(ifftshift(filteredDFT))


def low_pass(image_matrix, sigma):
    n = image_matrix.shape[0]
    m = image_matrix.shape[1]
    return fft(image_matrix, gaussian_blur(n, m, sigma, high_pass=False))


def high_pass(image_matrix, sigma):
    n = image_matrix.shape[0]
    m = image_matrix.shape[1]
    return fft(image_matrix, gaussian_blur(n, m, sigma, high_pass=True))


def hybrid(image_x, image_y, sigh, sigl):
    hi = high_pass(image_x, sigh)
    lo = low_pass(image_y, sigl)
    cv2.imwrite('high.jpg', numpy.real(hi))
    cv2.imwrite('low.jpg', numpy.real(lo))
    return hi + lo


img1 = cv2.imread("cat.jpg")
img2 = cv2.imread("dog.jpg")
img1_gry = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gry = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

cv2.imwrite("hybrid.jpg", numpy.real(hybrid(img1_gry, img2_gry, 25, 15)))
