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


gray = cv2.imread("plan.jpeg", cv2.IMREAD_GRAYSCALE)
gray = cv2.normalize(gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
_, gray = cv2.threshold(gray, 0.8, 1., cv2.THRESH_BINARY)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
scale = 1
cv2.resizeWindow('image', gray.shape[1]//scale, gray.shape[0]//scale)
cv2.imshow('image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

window_height = 10

left = np.zeros(gray.shape)

for down_i in range(gray.shape[0] - window_height):
    for left_i in range(gray.shape[1]):
        if number_of_clicks(gray[down_i:down_i+window_height, left_i]) > 5:
            left[down_i:down_i+window_height, left_i] += 1.

left = cv2.normalize(left, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
_, left = cv2.threshold(left, 0.6, 1., cv2.THRESH_BINARY)

"""
max_pool_size_up = 10
max_pool_size_left = 3
for down_i in range(0, gray.shape[0] - max_pool_size_up, max_pool_size_up):
    for left_i in range(0, gray.shape[1] - max_pool_size_left, max_pool_size_left):
        left[down_i:down_i+max_pool_size_up, left_i:left_i+max_pool_size_left] \
            = 1. if np.sum(left[down_i:down_i+max_pool_size_up, left_i:left_i+max_pool_size_left]) >= (max_pool_size_up * max_pool_size_left-1) else 0.
"""

dst = cv2.reduce(left, 0, cv2.REDUCE_SUM)[0]
import matplotlib.pyplot as plt
print(type(dst))

sortable = np.array(dst, copy=True)
sortable.sort()
print(sortable)
thr = sortable[len(sortable)-21]
for i in range(len(dst)):
    dst[i] = 0 if dst[i] < thr else dst[i]

plt.plot(range(len(dst)), dst)
plt.show()

cv2.imshow('image', left)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
scale = 1
cv2.resizeWindow('image', gray.shape[1]//scale, gray.shape[0]//scale)
cv2.imshow('image', left)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("plan_up_down.png", left*255)

