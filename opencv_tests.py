import cv2
import numpy as np

gray = cv2.imread("plan.jpeg", cv2.IMREAD_GRAYSCALE)
gray = cv2.normalize(gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
"""

cv2.imshow('image', gray)
cv2.waitKey(0)
"""
_, gray = cv2.threshold(gray, 0.5, 1., cv2.THRESH_BINARY)


output = []
for i in range(gray.shape[1]):
    nbr_changes = 0
    current = 0
    for j in range(gray.shape[0]):
        if gray[j][i] != current:
            nbr_changes += 1
            current = gray[j][i]
    output.append(nbr_changes)

import matplotlib.pyplot as plt
plt.plot(range(len(output)), output)
plt.show()

print(len(output))

for i in range(len(output)):
    if output[i] > 400:
        for j in range(1729):
            gray[j][i] = 0.
    else:
        for j in range(1729):
            gray[j][i] = 1.


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
scale = 1
cv2.resizeWindow('image', gray.shape[1]//scale, gray.shape[0]//scale)
cv2.imshow('image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(gray.shape)
print("min", gray.min())
print("max", gray.max())
