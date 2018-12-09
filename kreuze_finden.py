import cv2
import numpy as np


def meets_filter(patch):
    return not patch[0, 0] and patch[0, 1] and patch[0, 2] and patch[0, 3] and patch[0, 4] and not patch[1, 1] \
           and patch[1, 2] and patch[1, 3] and not patch[2, 2]


gray = cv2.imread("plan.jpeg", cv2.IMREAD_GRAYSCALE)
gray = cv2.normalize(gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
_, gray = cv2.threshold(gray, 0.5, 1., cv2.THRESH_BINARY)

cv2.imwrite("plan_thresholded.png", gray*255)

window_height = 3
window_width = 5
filtered_image = np.zeros(gray.shape)
for down_i in range(gray.shape[0] - window_height):
    for left_i in range(gray.shape[1] - window_width):
        if meets_filter(gray[down_i:down_i+window_height, left_i: left_i+window_width]):
            filtered_image[down_i:down_i+window_height, left_i: left_i+window_width] += 1.

filtered_image = cv2.normalize(filtered_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
_, filtered_image = cv2.threshold(filtered_image, 0.6, 1., cv2.THRESH_BINARY)


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
"""

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
scale = 1
cv2.resizeWindow('image', gray.shape[1]//scale, gray.shape[0]//scale)
cv2.imshow('image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("filtered_image.png", filtered_image*255)

