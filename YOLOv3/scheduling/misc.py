import cv2
import time
import os
import numpy as np
import json
import sys
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode


def visualize_history_file(history, Text_colors=(255,255,255)):
    """Visualize scheduling history from dictionary.

    Draw the scheduling order of bounding boxes in the image_out_path.
    Blue for box that meet deadline and red for box that missed.

    Args:
        history: A dictionary of scheduling history read from json file. 
    """
    for order in history:
        entry = history[order]
        if os.path.exists(entry["image_out_path"]):
            image = cv2.imread(entry["image_out_path"])
        else:
            image = cv2.imread(entry["image_path"])
        image_h, image_w, _ = image.shape

        bbox_color = (0,0,255) if (entry["missed"]) else (255,0,0)
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        coor = entry["coord"]
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)
        order_text = "order: " + str(order)
        # get text size
        (text_width, text_height), baseline = cv2.getTextSize(order_text, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                fontScale, thickness=bbox_thick)
        # put filled text rectangle
        cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)

        # put text above rectangle
        cv2.putText(image, order_text, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)
        
        i = entry["image_out_path"].rfind('/')
        output_directory = entry["image_out_path"][:i]
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
                
        cv2.imwrite(entry["image_out_path"], image)


def get_group_avg_response_time(history):
    """Calculate average response time for each depth group.

    Use the scheduling history to calculate the average response time for 
    each depth group. Each group is composed of objects that are in a 10m
    range, such as 0-10m, 10-20m, etc..

    Args:
        history: A dictionary of scheduling history read from json file. 
    
    Returns:
        A list of response time for each depth group. 
        For example,
        [25.293, 31.901, 9.244, 8.324, 3.987, 1.0, 0, 1.0, 1.0, 0]
    """

    res_time = [0] * 10
    group_cnt = [0] * 10
    result = []

    for key in history:
        entry = history[key]
        group_id = int(entry["depth"] / 10)
        res_time[group_id] += entry["response_time"]
        group_cnt[group_id] += 1
    
    for i in range(10):
        if group_cnt[i] != 0:
            result.append(float("{:.3f}".format(res_time[i] / group_cnt[i])))
        else:
            result.append(0)

    return result


def detect_images(model, image_path, box=None, output_path="", id=0, write_file=True, show=False):
    """Object classification of the given image.

    Run the yolo model on the given image. With post process including nms. 
    Save the output image to file or show the image if specified. 

    Args:
        model: The yolo model to be used. 
        image_path: path to the image.
        box: bounding box coordinates. Should be a list like: [x1, y1, x2, y2].
        output_path: path to write the output image. 
        id: index of bounding box for a given frame.
        show: whether to show the image for display.
    """
    original_image = cv2.imread(image_path)
    if box:
        original_image = original_image[box[1]:box[3], box[0]:box[2]]
    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    image_data, old_image_size, new_image_size = utils.image_preprocess(np.copy(original_image))
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    # pred_bbox = model.predict(image_data)
    pred_bbox = model.predict_on_batch(image_data)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = utils.postprocess_boxes(pred_bbox, old_image_size, new_image_size, 0.3)
    bboxes = utils.nms(bboxes, 0.45, method='nms')

    image = utils.draw_bbox(original_image, bboxes)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if id:
        i = output_path.rfind('.')
        output_path = output_path[:i] + '_' + str(id) + output_path[i:]
    if output_path != '' and write_file: 
        i = output_path.rfind('/')
        output_directory = output_path[:i]
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        cv2.imwrite(output_path, image)

    if show:
        # Show the image
        cv2.imshow("predicted image", image)
        # Load and hold the image
        cv2.waitKey(0)
        # To close the window after the required kill value was provided
        cv2.destroyAllWindows()


def load_Yolo_model():
    """Load a yolo model and its weights for inference."""
    input_layer = tf.keras.layers.Input([None, None, 3])
    feature_maps = YOLOv3(input_layer)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    utils.load_weights(model, "./yolov3.weights")
    return model


def extract_png_files(input_path):
    '''Find all png files within the given directory, sorted numerically.'''
    input_files = []
    file_names = os.listdir(input_path)

    for file in file_names:
        if ".png" in file:
            input_files.append(os.path.join(input_path, file))
    input_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return input_files


def read_json_file(filename):
    '''Return a dictionary read from json file.'''
    with open(filename) as json_file:
        data = json.load(json_file)
        return data



def get_bbox_info(frame, cluster_boxes):
    """Find bounding box information for the given frame.

    Get the bounding box information for the input frame from a dictionary.

    Args:
        frame: The image frame to be searched.
        cluster_boxes: a dictionary containing bounding box data.

    Returns:
        A list with the related bounding box data, including coordinates, depth, etc..
        For example, 
        [
            [644, 655, 729, 720, 64.44659992346784, ...],
            [571, 667, 759, 813, 29.452592092432084, ...],
            [1322, 764, 1920, 1214, 9.531812389460798, ...]
        ]
    """
    image_path = frame.path
    i = image_path.rfind('/')
    image_name = image_path[i+1:]

    if image_name in cluster_boxes:
        cluster_box = cluster_boxes[image_name]
        return cluster_box
    else:
        sys.exit("Error: no cluster box info for image {:s}".format(image_path))


def crop_cluster_box(frame, cluster_boxes_data):
    """Crop cluster boxes from frame

    Crop a image frame into several cluster boxes as specified by the info.

    Args:
        frame: The image frame to be cropped.
        cluster_boxes_data: a dictionary containing cluster boxes data.

    Returns:
        A list with each entry having two elements. The first element is the cropped image, 
        the second element the related cluster box data, including coordinates, depth, etc..
        For example:
        [[image1, data], [image2, data], [image3, data]]
    """

    images = []
    for box in cluster_boxes_data:
        images.append([frame.image[box[1]:box[3], box[0]:box[2]], box])
    return images


def list_to_str(l):
    """Function convert a coordinate list to string for printing"""
    return '(' + str(l[0]) + ',' + str(l[1]) + '), (' + str(l[2]) + ',' + str(l[3]) + ')'
