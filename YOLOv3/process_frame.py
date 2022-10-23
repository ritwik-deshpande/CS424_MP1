import collections

from scheduling.misc import *
from scheduling.TaskEntity import *
import json

# read the input bounding box data from file
box_info = read_json_file('../dataset/waymo_ground_truth_flat.json')

obj_map = collections.defaultdict(int)
w1 = 1
w2 = 1
T = 8
FRAMES = 10

def process_frame(frame):
    """Process frame for scheduling.

    Process a image frame to obtain cluster boxes and corresponding scheduling parameters
    for scheduling. 

    Student's code here.

    Args:
        param1: The image frame to be processed. 

    Returns:
        A list of tasks with each task containing image_path and other necessary information. 
    """

    #Part 1
    # return [TaskEntity(frame.path, coord = [0,0,1920,1280])]


    #Part 2
    # queue = []
    # cluster_boxes_data = get_bbox_info(frame, box_info)
    # for bbox in cluster_boxes_data:
    #     x_upper_left = bbox[0]
    #     y_upper_left = bbox[1]
    #     x_bottom_right = bbox[2]
    #     y_bottom_right = bbox[3]
    #
    #     area = (x_upper_left - x_bottom_right)*(y_upper_left - y_bottom_right)
    #     queue.append((area, bbox))
    #
    #
    # queue = sorted(queue)
    # print(queue)
    # tasks = []
    # priority = 0
    # for _, bbox in queue:
    #     tasks.append(TaskEntity(frame.path,coord = bbox[:4], priority = priority, depth = bbox[4]))
    #     priority += 1
    #
    # return tasks

    #Part 3
    # queue = []
    #
    # cluster_boxes_data = get_bbox_info(frame, box_info)
    # for bbox in cluster_boxes_data:
    #     depth = bbox[4]
    #     queue.append((depth, bbox))
    #
    #
    # queue = sorted(queue)
    # print(queue)
    # tasks = []
    # priority = 0
    # for _, bbox in queue:
    #     tasks.append(TaskEntity(frame.path,coord = bbox[:4], priority = priority, depth = bbox[4]))
    #     priority += 1
    # return tasks

    cluster_boxes_data = get_bbox_info(frame, box_info)
    queue = []
    tasks = []
    for bbox in cluster_boxes_data:
        compute = True
        for key in obj_map.keys():

            coord_diff = sum([abs(key[i] - bbox[i]) for i in range(4)])/4

            depth_diff = abs(key[4] - bbox[4])

            if (w1*coord_diff + w2*depth_diff)/(w1 + w2) < T :
                if obj_map[key] > FRAMES:
                    obj_map[key] = 1
                    compute = True
                else:
                    obj_map[key] += 1
                    compute = False

        if compute:
            queue.append((bbox[4], bbox))
            key = (bbox[0], bbox[1], bbox[2], bbox[3], bbox[4])
            obj_map[key] = 0


    # print(obj_map)
    # print('The queue is', queue)

    priority = 0
    queue = sorted(queue)
    for _, bbox in queue:
        tasks.append(TaskEntity(frame.path, coord=bbox[:4], priority=priority, depth=bbox[4]))
        priority += 1

    return tasks


