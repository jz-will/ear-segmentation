import tensorflow as tf
import numpy as np
import cv2
from model import unet


def main(argv):
    tf_flags = tf.app.flags.FLAGS

    # gpu config
    config = tf.ConfigProto()

    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    if tf_flags.phase == "train":
        with tf.Session(config=config) as sess:
            train_model = unet.UNet(sess, tf_flags)

            train_model.train(tf_flags.batch_size, tf_flags.training_steps,
                              tf_flags.summaty_steps, tf_flags.checkpint_steps, tf_flags.save_steps)
    else:
        with tf.Session(config=config) as sess:
            test_model = unet.UNet(sess, tf_flags)
            test_model.load(tf_flags.checkpint)
            image, output_masks = test_model.test()

            filename_A = "input.png"
            filename_B = "output_masks.png"

            cv2.imwrite(filename_A, np.uint8(image[0].clip(0., 1.) * 255.))
            cv2.imwrite(filename_B, np.uint8(output_masks[0].clip(0., 1.) * 255.))

            print("Saved files : {}, {}".format(filename_A, filename_B))


if __name__ == '__main__':
    tf.app.flags.DEFINE_string("output_dir", "model_output",
                               "checkpoint and summary directory.")
    tf.app.flags.DEFINE_string("phase", "train",
                               "model phase: train/test.")
    tf.app.flags.DEFINE_string("", "./datasets",
                               "dataset path for training.")
    tf.app.flags.DEFINE_string("testing_set", "./datasets/test",
                               "dataset path for testing one image pair.")

    tf.app.flags.DEFINE_integer("batch_size", 64,
                                "batch size for training.")
    tf.app.flags.DEFINE_integer("training_steps", 100000,
                                "total training steps.")
    tf.app.flags.DEFINE_integer("summary_steps", 100,
                                "summary period.")
    tf.app.flags.DEFINE_integer("checkpoint_steps", 1000,
                                "checkpoint period.")
    tf.app.flags.DEFINE_integer("save_steps0", 500,
                                "checkpoint period.")
    tf.app.flags.DEFINE_string("checkpoint", None,
                               "checkpoint name for restoring.")
    tf.app.run(main=main)
