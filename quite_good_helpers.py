import cv2
import numpy as np


def number_of_clicks(array):
    clicks = 0
    old = array[0]
    for i in range(1, len(array)):
        if old != array[i]:
            clicks += 1
            old = array[i]
    return clicks


def get_top_down(gray, window_height):
    output = np.zeros(gray.shape)
    for down_i in range(gray.shape[0] - window_height):
        for left_i in range(gray.shape[1]):
            if number_of_clicks(gray[down_i:down_i + window_height, left_i]) > 5:
                output[down_i:down_i + window_height, left_i] += 1.

    return cv2.normalize(output, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def get_left_right(gray, window_height):
    output = np.zeros(gray.shape)
    for down_i in range(gray.shape[0]):
        for left_i in range(gray.shape[1] - window_height):
            if number_of_clicks(gray[down_i, left_i:left_i + window_height]) > 5:
                output[down_i, left_i:left_i + window_height] += 1.

    return cv2.normalize(output, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def extract_two_extremes(array_of_activations):
    arr = np.array(array_of_activations, copy=True)
    arr.sort()
    if len(array_of_activations) == 1729:
        thr = arr[-3]
    else:
        thr = arr[-10]
    tmp = []
    for i in range(len(array_of_activations)):
        if array_of_activations[i] > thr:
            tmp.append(i)
    current = 0
    output = []
    for point in tmp:
        if point-current > 10:
            current = point
            output.append(point)
    if len(output) == 1:
        return output[0], output[0]
    return output[0], output[1]

