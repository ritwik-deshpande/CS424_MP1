from queue import PriorityQueue
import core.utils as utils
from scheduling.misc import *
from process_frame import *


class Scheduler:
    """Scheduler to process image frames.

    This scheduler simulates how a image stream is processed. 
    The frames information and Yolo model is also stored in this class.

    Attributes:
        time: the simulated timer.
        frame_period: The period to obtain a new frame. 
        image_directory: The path to the image directory. Default is "../dataset/".
        image_list: a list containing all the images to be processed. 
        max_frame_number: the number of frame to be processed. 
        run_queue: a priority queue that sorts task by their priority.
                A lower number means higher priority. 
        history: scheduling history. 
        task_finish_count: number of tasks that have finished. 
        yolo: YoloV3 model. Used for object classification.
    """

    def __init__(self, image_directory = "../dataset/", frame_period = 100):
        self.time = 0
        self.frame_period = frame_period
        self.frame_number = 0
        self.image_directory = image_directory
        self.image_list = extract_png_files(image_directory)
        self.max_frame_number = len(self.image_list)
        self.run_queue = PriorityQueue()
        self.history = []
        self.task_finish_count = 0
        self.yolo = load_Yolo_model()


    def run(self):
        """Main scheduling loop.

        The scheduling loop finishes until all frames have been processed.
        The scheduler gets a new frame for each frame period. The frame should
        be processed by student's code and return a list of tasks corresponding
        to different cluster boxes for that frame.

        The scheduler always run the top task in the run queue.

        """
        while self.frame_number <= self.max_frame_number or not self.run_queue.empty():
            
            # get a frame when frame period arrives
            if self.time % self.frame_period == 0:
                self.frame_arrival(self.frame_number)
                self.frame_number = self.frame_number + 1

            # if there are tasks in the run queue
            if not self.run_queue.empty():
                top_task = self.run_queue.queue[0]
                top_task.remain_time = top_task.remain_time - 1

                # if the task has finished
                if top_task.remain_time == 0:
                    task = self.run_queue.get()
                    self.task_finish_count = self.task_finish_count + 1
                    task.order = self.task_finish_count
                    # add 1 because self.time start from 0
                    task.response_time = self.time - task.enqueue_time + 1
                    if task.response_time > task.deadline:
                        task.missed = 1

                    self.history.append(task)
                    # you can comment out the following line to skip yolo for shorter program runtime
                    # self.run_yolo(task)
            
            self.time = self.time + 1
        
        # save scheduling history to file
        self.save_history()
        print("Scheduling history saved.")

    def get_frame(self, frame_number):
        """Return and Image() object with the specified frame number."""
        if frame_number < self.max_frame_number:
            image_path = self.image_list[frame_number]
            return Image(image_path)

        else:
            return None


    def enqueue_task(self, task_set):
        """Enqueue the task_set into the run queue."""
        for task in task_set:
            task.enqueue_time = self.time
            task.remain_time = self.get_execution_time(task)
            task.exec_time = task.remain_time
            self.run_queue.put(task)


    def frame_arrival(self, frame_number):
        """Get a frame from the image list and return related tasks.
        
        Fetch a frame from the dataset using the given frame number.
        Process the frame to get tasks to be classified.
        Enqueue the tasks to the run queue.

        Args:
            frame_number: The number of frame in the image list.
        """
        frame = self.get_frame(frame_number)
        if frame:
            task_set = process_frame(frame)
            self.enqueue_task(task_set)


    def run_yolo(self, task):
        """Run the yolo model on the given task."""
        print("On image {}".format(task.image_path))
        detect_images(self.yolo, task.image_path, task.coord, task.image_out_path, task.bbox_id)


    def get_execution_time(self, task):
        """Return a simulated execution time for this task."""
        return int(1e-4 * task.img_height * task.img_width) + 1


    def print_image_list(self):
        """Print out the list of images to be processed."""
        print("image list is: ")
        print(self.image_list)


    def print_history(self):
        """Print out the scheduling history of the scheduler."""
        dash = '-' * 70
        print(dash)
        print("history:")
        print('{:<7s}{:<30s}{:>25s}{:>8s}{:>10s}{:>15s}{:>12s}{:>15s}{:>10s}{:>10s}'.format(
            "count", "task_image", "img_coordinates", "depth", "priority",
            "enqueue_time", "exec_time", "response_time", "deadline", "missed"))

        i = 1
        for entry in self.history:
            print('{:<7d}{:s}'.format(i, entry.print()))
            i = i + 1
        print(dash)
    

    def save_history(self):
        """Save the scheduling history as a json file."""
        d = {}
        i = 1
        for entry in self.history:
            d[i] = entry.__dict__
            i = i + 1
        
        with open('scheduling_history.json', 'w') as outfile:
            json.dump(d, outfile, ensure_ascii=False, indent=4)

    def visualize_history(self, Text_colors=(255,255,255)):
        """Visualize scheduling order.

        Draw the scheduling order of bounding boxes in the image_out_path.
        Blue for box that meet deadline and red for box that missed.      
        """
        order = 1
        for task in self.history:
            if os.path.exists(task.image_out_path):
                image = cv2.imread(task.image_out_path)
            else:
                image = cv2.imread(task.image_path)
            image_h, image_w, _ = image.shape

            bbox_color = (0,0,255) if (task.missed) else (255,0,0)
            bbox_thick = int(0.6 * (image_h + image_w) / 1000)
            if bbox_thick < 1: bbox_thick = 1
            fontScale = 0.75 * bbox_thick
            coor = task.coord
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
            
            cv2.imwrite(task.image_out_path, image)

            order = order + 1


