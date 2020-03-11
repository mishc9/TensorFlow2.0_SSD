import tensorflow as tf
import cv2

from tf2ssd.configuration import save_model_dir, test_picture_dir
from tf2ssd.core import SSD
from tf2ssd.utils import test_single_picture

# shape of boxes : (N, 4)  (xmin, ymin, xmax, ymax)
# shape of scores : (N,)
# shape of classes : (N,)


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    ssd_model = SSD()
    ssd_model.load_weights(filepath=save_model_dir + "saved_model")

    image = test_single_picture(picture_dir=test_picture_dir, model=ssd_model)

    cv2.namedWindow("detect result", flags=cv2.WINDOW_NORMAL)
    cv2.imshow("detect result", image)
    cv2.waitKey(0)
