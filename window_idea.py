import cv2
import numpy as np


def gugus(plan, window_height, window_width):
    h = 400
    w = 900
    plan[h:h+window_height, w:w+window_width, 1] = np.ones((window_height, window_width)) * 250
    plan[h:h+window_height, w:w+window_width, 0] = np.zeros((window_height, window_width))
    plan[h:h+window_height, w:w+window_width, 2] = np.zeros((window_height, window_width))
    cv2.imshow("image", plan)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def compute_side_sum(impatch):
    height, width = impatch.shape
    output = []
    for i in range(height):
        total = 0
        for j in range(width):
            total += impatch[i, j]
        output.append(total)
    return np.array(output)


def compute_down_sum(impatch, window_size):
    output = []
    for i in range(window_size):
        total = 0
        for j in range(window_size):
            total += impatch[j, i]
        output.append(total)
    return np.array(output)


def sum_absolute_diffs(array):
    output = array[0]
    for i in range(1, len(array)):
        output += abs(array[i] - array[i-1])
    return output


def number_of_clicks(array):
    clicks = 0
    old = array[0]
    for i in range(1, len(array)):
        if old != array[i]:
            clicks += 1
            old = array[i]
    return clicks


def is_binary(array):
    return 1. if len(set(array)) == 2 else 0.


def binary_boosted(array):
    nbr_clicks = number_of_clicks(array)
    binary = is_binary(array) == 1.
    return 1. if nbr_clicks > 5 and binary else 0.


plan = cv2.imread("plan.jpeg")
gray = cv2.imread("plan.jpeg", cv2.IMREAD_GRAYSCALE)
gray = cv2.normalize(gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
_, gray = cv2.threshold(gray, 0.5, 1., cv2.THRESH_BINARY)

print(plan.shape)
window_height = 20
window_width = 6
gugus(plan, window_height, window_width)

left_to_right = np.zeros(gray.shape)
up_to_down = np.zeros(gray.shape)

nbr_inter_steps = 2
output_patches = np.empty((gray.shape[0] // window_height * nbr_inter_steps, gray.shape[1] // window_width * nbr_inter_steps))
for down_i in range(0, gray.shape[0] - window_height, window_height//nbr_inter_steps):
    for left_i in range(0, gray.shape[1] - window_width, window_width//nbr_inter_steps):
        patch = gray[down_i:down_i + window_height, left_i:left_i + window_width]
        side_sum = compute_side_sum(patch)
        diff_side = binary_boosted(side_sum)
        # left_to_right[down_i:down_i + window_height, left_i:left_i + window_width] = np.ones(
        #    (window_height, window_width)) * diff_side
        output_patches[down_i // window_height * nbr_inter_steps, left_i // window_width * nbr_inter_steps] = diff_side


output = []
for i in range(gray.shape[1]):
    nbr_changes = 0
    current = 0
    for j in range(gray.shape[0]):
        if gray[j][i] != current:
            nbr_changes += 1
            current = gray[j][i]
    output.append(nbr_changes)
output = list(map(lambda x: True if x > 400 else False, output))

for i in range(gray.shape[0] - window_height):
    for j in range(gray.shape[1] - window_width):
        if True and output_patches[i//(window_height//nbr_inter_steps), j//(window_width//nbr_inter_steps)]:
            gray[i, j] = output_patches[i//(window_height//nbr_inter_steps), j//(window_width//nbr_inter_steps)]
        else:
            gray[i, j] = 0.


left = cv2.normalize(gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

cv2.imshow('image', left)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
scale = 1
cv2.resizeWindow('image', gray.shape[1]//scale, gray.shape[0]//scale)
cv2.imshow('image', left)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("plan.png", left*255)

