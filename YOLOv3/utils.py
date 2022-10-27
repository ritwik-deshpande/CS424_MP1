import collections

from scheduling.misc import *
from scheduling.TaskEntity import *
import json
from matplotlib import pyplot as plt

history = read_json_file("scheduling_history.json")


def show_avg_response_time():
    c = get_group_avg_response_time(history)
    print(c)
    # fig, ax = plt.subplots(figsize =(10, 7))
    # ax.hist(c, bins = 1)
    fig, ax = plt.subplots()
    y = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    plt.bar(y, c, align='center')
    # Show plot
    labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels[1] = 'Testing'
    ax.set_xticklabels(labels)
    plt.xlabel('Bbox Groups')
    plt.ylabel('Avg. Response Time')
    plt.show()


def get_inversion_time():
    frame_obj_map = collections.defaultdict(list)

    for _, img in history.items():
        img_path = img['image_path']
        depth = img['depth']
        response_time = img['response_time']
        frame_obj_map[img_path].append((depth, response_time))

    inversion_time = []
    for frame, objs in frame_obj_map.items():
        min_depth = min([d for d, _ in objs])
        # print(min_depth)
        blocking_time = 0
        # print(objs)
        for obj in objs:
            if obj[0] == min_depth:
                break
            else:
                blocking_time += obj[1]

        inversion_time.append(blocking_time)

    # print(inversion_time)

    avg_inversion_time = sum(inversion_time)/len(inversion_time)

    max_inversion_time = max(inversion_time)

    return avg_inversion_time, max_inversion_time


if __name__ == '__main__':
    print(get_inversion_time())
