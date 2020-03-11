import cv2
import tensorflow as tf

from tf2ssd.configuration import OBJECT_CLASSES
from tf2ssd.core import InferenceProcedure
from tf2ssd.utils.tools import preprocess_image


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
