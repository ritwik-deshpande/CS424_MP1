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
    return [TaskEntity(frame.path, coord = [0,0,1920,1280])]

    # cluster_boxes_data = get_bbox_info(frame, box_info)
    
    # student's code here
    
    