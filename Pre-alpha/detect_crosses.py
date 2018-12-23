import cv2
import numpy as np
import matplotlib.pyplot as plt


kernel_wide_v = np.array([-1, 1, 1, 1, 1, -1, 0, -1, 1, 1, -1, 0, 0, 0, -1, -1, 0, 0])
kernel_wide_v = np.reshape(kernel_wide_v, (3, -1))
wide_kernels = [kernel_wide_v, np.flip(kernel_wide_v), kernel_wide_v.transpose(), np.flip(kernel_wide_v).transpose()]


kernel_v = np.array([-1, 1, 1, 1, -1, 0, -1, 1, -1, 0, 0, 0, -1, 0, 0])
kernel_v = np.reshape(kernel_v, (3, -1))
v_kernels = [kernel_v, np.flip(kernel_v), kernel_v.transpose(), np.flip(kernel_v).transpose()]


def get_maximum(arr):
    return np.argmax(arr)


class CrossesDetection:
    def __init__(self, image):
        self.image = image

    def get_wide_kernel_results(self):
        filters_added = np.zeros(self.image.shape)
        for kernel in wide_kernels:
            res = cv2.filter2D(self.image, -1, kernel)
            _, res = cv2.threshold(res, 5.2, 6., cv2.THRESH_BINARY)
            filters_added += res / 6.
        return filters_added

    def get_v_kernel_results(self):
        filters_added = np.zeros(self.image.shape)
        for kernel in v_kernels:
            res = cv2.filter2D(self.image, -1, kernel)
            _, res = cv2.threshold(res, 3.2, 6., cv2.THRESH_BINARY)
            filters_added += res / 4.
        return filters_added

    def get_kernel_results(self):
        return self.get_wide_kernel_results() + self.get_v_kernel_results()

    def get_candidate_triple(self):
        kernel_image = self.get_kernel_results()
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
