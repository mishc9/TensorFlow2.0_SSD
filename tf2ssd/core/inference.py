import cv2
import tensorflow as tf
import numpy as np
from tf2ssd.configuration import NUM_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH, OBJECT_CLASSES, training_results_save_dir
from tf2ssd.core.anchor import DefaultBoxes
from tf2ssd.core.ssd import ssd_prediction
from tf2ssd.utils.nms import NMS
from tf2ssd.utils.tools import preprocess_image


class InferenceProcedure(object):
    def __init__(self, model):
        self.model = model
        self.num_classes = NUM_CLASSES + 1
        self.image_size = np.array([IMAGE_HEIGHT, IMAGE_WIDTH], dtype=np.float32)
        self.nms_op = NMS()

    def __get_ssd_prediction(self, image):
        output = self.model(image, training=False)
        pred = ssd_prediction(feature_maps=output, num_classes=self.num_classes)
        return pred, output

    @staticmethod
    def __resize_boxes(boxes, image_height, image_width):
        cx = boxes[..., 0] * image_width
        cy = boxes[..., 1] * image_height
        w = boxes[..., 2] * image_width
        h = boxes[..., 3] * image_height
        xmin = cx - w / 2
        ymin = cy - h / 2
        xmax = cx + w / 2
        ymax = cy + h / 2
        resized_boxes = tf.stack(values=[xmin, ymin, xmax, ymax], axis=-1)
        return resized_boxes

    def __filter_background_boxes(self, ssd_predict_boxes):
        is_object_exist = True
        num_of_total_predict_boxes = ssd_predict_boxes.shape[1]
        scores = tf.nn.softmax(ssd_predict_boxes[..., :self.num_classes])
        classes = tf.math.argmax(input=scores, axis=-1)
        filtered_boxes_list = []
        for i in range(num_of_total_predict_boxes):
            if classes[:, i] != 0:
                filtered_boxes_list.append(ssd_predict_boxes[:, i, :])
        if filtered_boxes_list:
            filtered_boxes = tf.stack(values=filtered_boxes_list, axis=1)
            return is_object_exist, filtered_boxes, scores
        else:
            is_object_exist = False
            return is_object_exist, ssd_predict_boxes, scores

    def __offsets_to_true_coordinates(self, pred_boxes, ssd_output):
        pred_classes = tf.reshape(tensor=pred_boxes[..., :self.num_classes], shape=(-1, self.num_classes))
        pred_coords = tf.reshape(tensor=pred_boxes[..., self.num_classes:], shape=(-1, 4))
        default_boxes = DefaultBoxes(feature_map_list=ssd_output).generate_default_boxes()
        d_cx, d_cy, d_w, d_h = default_boxes[:, 0:1], default_boxes[:, 1:2], default_boxes[:, 2:3], default_boxes[:, 3:4]
        offset_cx, offset_cy, offset_w, offset_h = pred_coords[:, 0:1], pred_coords[:, 1:2], pred_coords[:, 2:3], pred_coords[:, 3:4]
        true_cx = offset_cx * d_w + d_cx
        true_cy = offset_cy * d_h + d_cy
        true_w = tf.math.exp(offset_w) * d_w
        true_h = tf.math.exp(offset_h) * d_h
        true_coords = tf.concat(values=[true_cx, true_cy, true_w, true_h], axis=-1)
        true_classes_and_coords = tf.concat(values=[pred_classes, true_coords], axis=-1)
        true_classes_and_coords = tf.expand_dims(input=true_classes_and_coords, axis=0)
        return true_classes_and_coords

    def get_final_boxes(self, image):
        pred_boxes, ssd_output = self.__get_ssd_prediction(image)
        pred_boxes = self.__offsets_to_true_coordinates(pred_boxes=pred_boxes, ssd_output=ssd_output)
        is_object_exist, filtered_pred_boxes, pred_boxes_class = self.__filter_background_boxes(pred_boxes)
        if is_object_exist:
            pred_boxes_class = tf.reshape(tensor=pred_boxes_class, shape=(-1, self.num_classes))
            pred_boxes_coord = filtered_pred_boxes[..., self.num_classes:]
            pred_boxes_coord = tf.reshape(tensor=pred_boxes_coord, shape=(-1, 4))
            resized_pred_boxes = self.__resize_boxes(boxes=pred_boxes_coord,
                                                     image_height=image.shape[1],
                                                     image_width=image.shape[2])
            box_tensor, score_tensor, class_tensor = self.nms_op.nms(boxes=resized_pred_boxes,
                                                                     box_scores=pred_boxes_class)
            return is_object_exist, box_tensor, score_tensor, class_tensor
        else:
            return is_object_exist, tf.zeros(shape=(1, 4)), tf.zeros(shape=(1,)), tf.zeros(shape=(1,))


def find_class_name(class_id):
    for k, v in OBJECT_CLASSES.items():
        if v == class_id:
            return k


def draw_boxes_on_image(image, boxes, scores, classes):
    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        class_and_score = str(find_class_name(classes[i])) + ": " + str(scores[i].numpy())
        cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 2], boxes[i, 3]), color=(255, 0, 0),
                      thickness=2)
        cv2.putText(img=image, text=class_and_score, org=(boxes[i, 0], boxes[i, 1] - 10),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5, color=(0, 255, 255), thickness=2)
    return image


def test_single_picture(picture_dir, model):
    image_tensor = preprocess_image(picture_dir)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    procedure = InferenceProcedure(model=model)
    is_object_exist, boxes, scores, classes = procedure.get_final_boxes(image=image_tensor)
    if is_object_exist:
        image_with_boxes = draw_boxes_on_image(cv2.imread(picture_dir), boxes, scores, classes)
    else:
        print("No objects were detected.")
        image_with_boxes = cv2.imread(picture_dir)
    return image_with_boxes


def visualize_training_results(pictures, model, epoch):
    # pictures : List of image directories.
    index = 0
    for picture in pictures:
        index += 1
        result = test_single_picture(picture_dir=picture, model=model)
        cv2.imwrite(filename=training_results_save_dir + "epoch-{}-picture-{}.jpg".format(epoch, index), img=result)