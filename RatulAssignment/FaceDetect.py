import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
import numpy as np


class FaceDetection:

    def __init__(self, model_path, min_size=80, factor=0.709, thresholds=(
                    0.75, 0.75, 0.75)):
        """Loads MTCNN model
                Parameters:
                -----------
                 model_path: str
                    path to MTCNN model
                 min_size: integer
                    minimum face size detection
                 threshold: tuple
                    percentage accuracy threshold for each convolutional
                    network(PNet, RNet, ONet)
                 factor: integer
                    factor to obtain image pyramid

                """
        self.factor = factor
        self.thresholds = thresholds
        self.min_size = min_size

        graph = tf.Graph()
        with graph.as_default():
            with open(model_path, 'rb') as f:
                graph_def = tf.GraphDef.FromString(f.read())
                tf.import_graph_def(graph_def, name='')
        self.graph = graph
        config = tf.ConfigProto(
            allow_soft_placement=True,
            intra_op_parallelism_threads=4,
            inter_op_parallelism_threads=4)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=config)

    def detect_face(self, img, min_size):
        """Takes encoded image as input and returns face rects and landmarks
        Parameters
        ----------
         img: numpy array
            encoded image
         min_size: integer
            minimum face detection size
        Return
        ------
            list,numpy array
            Returns a list of rects and landmarks
        """
        self.min_size = min_size

        feeds = {
            self.graph.get_operation_by_name('input').outputs[0]: img,
            self.graph.get_operation_by_name('min_size').outputs[
                0]: self.min_size,
            self.graph.get_operation_by_name('thresholds').outputs[
                0]: self.thresholds,
            self.graph.get_operation_by_name('factor').outputs[0]: self.factor
        }
        fetches = [self.graph.get_operation_by_name('prob').outputs[0],
                   self.graph.get_operation_by_name('landmarks').outputs[0],
                   self.graph.get_operation_by_name('box').outputs[0]]
        prob, landmarks, boxes = self.sess.run(fetches, feeds)
        boxes = non_max_suppression_fast(boxes, 0.3)

        for box in boxes:
            if box[0] < 0:
                box[0] = 0
            if box[1] < 0:
                box[1] = 0
            if box[2] > img.shape[0]:
                box[2] = img.shape[0]
            if box[3] > img.shape[1]:
                box[3] = img.shape[1]

            temp_2 = box[2]
            box[2] = box[3] - box[1]
            box[3] = temp_2 - box[0]
            temp_0 = box[0]
            box[0] = box[1]
            box[1] = temp_0

        ypart_temp = []
        for landmark in landmarks:
            ypart_temp[:5] = landmark[5:]
            landmark[5:] = landmark[:5]
            landmark[:5] = ypart_temp

        if len(boxes) > 0:
            boxes = boxes.tolist()

        return boxes, landmarks


def non_max_suppression_fast(boxes, overlap_thresh):
    """
    Parameters
    ----------
     boxes: numpy array
        coordinates of all boxes(faces)
     overlap_thresh: integer
        threshold for overlapping boxes
    Return
    ------
        boxes:
            return the picked boxes after this nms

    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x_1 = boxes[:, 0]
    y_1 = boxes[:, 1]
    x_2 = boxes[:, 2]
    y_2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)
    idxs = np.argsort(y_2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx_1 = np.maximum(x_1[i], x_1[idxs[:last]])
        yy_1 = np.maximum(y_1[i], y_1[idxs[:last]])
        xx_2 = np.minimum(x_2[i], x_2[idxs[:last]])
        yy_2 = np.minimum(y_2[i], y_2[idxs[:last]])

        # compute the width and height of the bounding box
        width = np.maximum(0, xx_2 - xx_1 + 1)
        height = np.maximum(0, yy_2 - yy_1 + 1)

        # compute the ratio of overlap
        overlap = (width * height) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(
                                                   overlap > overlap_thresh)[
                                                   0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")
