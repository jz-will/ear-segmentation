import numpy as np
import cv2


def save_images(input, output1, output2, input_path, image_path, max_samples=4):
    """
    :param input:
    :param output1:
    :param output2:
    :param input_path:
    :param image_path:
    :param max_samples:
    :return: None
    """
    image = np.concatenate([output1, output2], axis=2)
    if max_samples > int(image.shape[0]):
        max_samples = int(image.shape[0])

    image = image[0:max_samples, :, :, :]

    image = np.concatenate([image[i, :, :, :] for i in range(max_samples)], axis=0)

    cv2.imwrite(image_path, np.uint8(image.clip(0., 1.) * 255.))

    if input is not None:
        input_data = input[0:max_samples, :, :, :]

        # [1024, 256, 3]
        input_data = np.concatenate([input_data[i, :, :, :] for i in range(max_samples)], axis=0)
        cv2.imwrite(input_path, np.uint8(input_data.clip(0., 1.) * 255.))