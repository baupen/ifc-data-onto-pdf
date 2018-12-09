import json
import collections
import intervals as I


def get_all_x_values(list_of_rooms):
    output = set()
    for room in list_of_rooms:
        for point_index in range(len(room) - 1):
            current_point = room[point_index]
            next_point = room[point_index + 1]
            if abs(current_point['x'] - next_point['x']) < 0.00001:
                output.add(current_point['x'])
    return list(output)


def set_up_dict(list_of_x_values):
    output = {}
    for x in list_of_x_values:
        output[x] = []
    return collections.OrderedDict(sorted(output.items()))


def shrink_interval_list(list_of_intervals):
    output = []
    for original_interval in list_of_intervals:
        combined_flag = False
        for i in range(len(output)):
            if original_interval.overlaps(output[i]):
                output[i] = original_interval.union(output[i])
                combined_flag = True
                break
        if not combined_flag:
            output.append(original_interval)
    return output


def turn_intervals_to_size(list_of_intervals):
    """the list shouldn't have intervals that overlap"""
    output = 0.
    for interval in list_of_intervals:
        interval = interval[0]
        output += interval.upper - interval.lower
    return output


with open("sample.ifc.json") as f:
    data = json.load(f)
    print(data['floor'][0].keys())

rooms = list(map(lambda x: x['points'], data['floor']))

all_x_values = get_all_x_values(rooms)
x_distribution = set_up_dict(all_x_values)

for room in rooms:
    for point_index in range(len(room) - 1):
        current_point = room[point_index]
        next_point = room[point_index + 1]
        if abs(current_point['x'] - next_point['x']) < 0.0000001:
            if current_point['y'] < next_point['y']:
                x_distribution[current_point['x']].append(I.closed(current_point['y'], next_point['y']))
            else:
                x_distribution[current_point['x']].append(I.closed(next_point['y'], current_point['y']))

x_distribution = collections.OrderedDict(sorted(x_distribution.items()))


i = 0
while i < len(x_distribution)-1:
    i_th_key = list(x_distribution.keys())[i]
    i_p_1_th_key = list(x_distribution.keys())[i+1]
    if abs(i_th_key-i_p_1_th_key) < 0.001:
        x_distribution[i_th_key] += x_distribution[i_p_1_th_key]
        del x_distribution[i_p_1_th_key]
        continue
    i += 1


keys = list(x_distribution.keys())
for i in range(len(x_distribution)):
    x_distribution[keys[i]] = turn_intervals_to_size(shrink_interval_list(x_distribution[keys[i]]))


for k, v in x_distribution.items():
    print(k, "\t", v)