from scheduling.misc import *
from scheduling.TaskEntity import *


# read the input bounding box data from file
box_info = read_json_file('../dataset/waymo_ground_truth_flat.json')


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
    queue = []

    cluster_boxes_data = get_bbox_info(frame, box_info)
    for bbox in cluster_boxes_data:
        x_upper_left = bbox[0]
        y_upper_left = bbox[1]
        x_bottom_right = bbox[2]
        y_bottom_right = bbox[3]

        area = (x_upper_left - x_bottom_right)*(y_upper_left - y_bottom_right)
        queue.append((area, bbox))


    queue = sorted(queue)
    tasks = []
    priority = 0
    for _, bbox in queue:
        tasks.append(TaskEntity(frame.path,coord = bbox[:4], priority = priority,depth = bbox[4]))
        priority += 1

    return tasks

    