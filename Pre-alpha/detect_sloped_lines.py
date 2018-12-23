import cv2
import numpy as np
import matplotlib.pyplot as plt


kernel_slope = np.array([1,  1, -1, 1, -1,  0, -1,  0,  0])
kernel_slope = np.reshape(kernel_slope, (3, 3))

kernels = [kernel_slope, np.flip(kernel_slope)]


def get_maximum(arr):
    return np.argmax(arr)


def refine_kernel_results(img):
    second_kernel = np.array([-1, -1, 1, -1, 1, -1, 1, -1, -1])
    second_kernel = np.reshape(second_kernel, (3, 3))

    tmp = cv2.filter2D(img, -1, second_kernel)
    _, output = cv2.threshold(tmp, 2.2, 3., cv2.THRESH_BINARY)
    return output / 3.


class SlopedLinesDetection:
    def __init__(self, image):
        self.image = image

    def get_kernel_results(self):
        all_filters_added = np.zeros(self.image.shape)
        for kernel in kernels:
            res = cv2.filter2D(self.image, -1, kernel)
            _, res = cv2.threshold(res, 2.5, 3., cv2.THRESH_BINARY)
            all_filters_added += res / 3.
        return all_filters_added

    def get_candidate_triple(self):
        kernel_image = self.get_kernel_results()
        kernel_image = refine_kernel_results(kernel_image)
        left_to_right_sum = np.reshape(cv2.reduce(kernel_image, 0, cv2.REDUCE_SUM), (-1,))
        up_down_sum = np.reshape(cv2.reduce(kernel_image, 1, cv2.REDUCE_SUM), (-1,))

        first_index = get_maximum(left_to_right_sum)
        left_to_right_sum[first_index-10:first_index+10] = np.zeros((20,))
        second_index = get_maximum(left_to_right_sum)
        third_index = get_maximum(up_down_sum)

        first_index /= self.image.shape[1]
        second_index /= self.image.shape[1]
        third_index /= self.image.shape[0]

        if first_index < second_index:
            return first_index, second_index, third_index
        else:
            return second_index, first_index, third_index

