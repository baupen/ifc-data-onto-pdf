import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import sys

import quite_good_helpers

assert sys.argv[1] == "-d"
sectors_json_path = sys.argv[2]

gray = cv2.imread(sectors_json_path, cv2.IMREAD_GRAYSCALE)
gray = cv2.normalize(gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
_, gray = cv2.threshold(gray, 0.8, 1., cv2.THRESH_BINARY)

window_size = 10

left_to_right = quite_good_helpers.get_left_right(gray, window_size)
top_down = quite_good_helpers.get_top_down(gray, window_size)

_, top_down = cv2.threshold(top_down, 0.6, 1., cv2.THRESH_BINARY)
_, left_to_right = cv2.threshold(left_to_right, 0.6, 1., cv2.THRESH_BINARY)

# TODO: show the result left_to_right
# todo: show the result top_down

dst_top_down = np.reshape(cv2.reduce(top_down, 0, cv2.REDUCE_SUM)[0], (-1,))
dst_left_to_right = np.reshape(cv2.reduce(left_to_right, 1, cv2.REDUCE_SUM), (-1,))

"""
plt.plot(range(len(dst_left_to_right)), dst_left_to_right)
plt.show()
plt.plot(range(len(dst_top_down)), dst_top_down)
plt.show()
"""

y_1, y_2 = quite_good_helpers.extract_two_extremes(dst_left_to_right)
x_1, x_2 = quite_good_helpers.extract_two_extremes(dst_top_down)



# import the shapes from the JSON
# import the ratio from the JSON
with open("sample.ifc.json") as json_file:
    # draw rectangles as given by points
    data = json.load(json_file)
    ratio = data["ratio"]
    rooms = data["floor"]

output = {
    "upper": x_1 / gray.shape[1],
    "lower": x_2 / gray.shape[1],
    "left": y_1 / gray.shape[0],
    "right": (y_1 + (x_2 - x_1) * ratio) / gray.shape[0]
}

output_name = sectors_json_path[:-5] + "frame.json"
with open(output_name, 'w', encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

rooms_points = list(map(lambda r: r["points"], rooms))

rooms = []
for room in rooms_points:
    room_points = []
    for point in room:
        room_points.append(np.array([point["x"], point["y"]]))
    rooms.append(room_points)

offset = np.array([x_1, y_1])

bla = np.zeros((*gray.shape, 3))

for room in rooms:
    polygon_points = []
    for point in room:
        point *= np.array([(x_2 - x_1), (x_2 - x_1) / ratio])
        point += offset
        polygon_points.append(point)
    polygon_points.append(polygon_points[0])
    polygon_points = np.array(polygon_points, np.int32)
    polygon_points = np.reshape(polygon_points, (-1, 1, 2))
    cv2.polylines(bla, [polygon_points], False, (0, 255, 255), thickness=10)


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
scale = 1
cv2.resizeWindow('image', bla.shape[1]//scale, bla.shape[0]//scale)
cv2.imshow('image', bla)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("gugus.png", bla)


