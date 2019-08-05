import os
import logging
import time
from datetime import datetime
import tensorflow as tf

from models import Unet
from utils import save_images
from data import read_tfRecords

import sys

sys.path.append("../data")
import numpy as np
import cv2
import glob


class UNet(object):
    def __init__(self, sess, tf_flags):
        self.sess = sess
        self.dtype = tf.float32

        self.output_dir = tf_flags.output_dir
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoint")

        self.checkpoint_prefix = "model"
        self.saver_name = "checkpoint"

        self.summary_dir = os.path.join(self.output_dir, "summary")

        self.is_training = (tf_flags.phase == "train")
        self.learning_rate = 0.001
        # raw和mask image：1918 * 1280
        self.image_w = 512
        self.image_h = 512
        self.image_c = 3
        # [None, 512, 512, 3]
        self.input_data = tf.placeholder(self.dtype, [None, self.image_h,
                                                      self.image_w, self.image_c])

        # [None, 324, 324, 3]
        self.input_masks = tf.placeholder(self.dtype, [None, 324, 324,
                                                       self.image_c])
        self.lr = tf.placeholder(self.dtype)

        if self.is_training:
            self.training_set = tf_flags.training_set
            self.sample_dir = "train_results"
            # 创建summary_dir，checkpoint_dir，sample_dir
            self._make_aux_dirs()
            # 定义 loss，优化器，summary，saver
            self._build_training()

            log_file = self.output_dir + "Unet.log"
            logging.basicConfig(format='%(asctime)s[%(levelname)s]%(message)s',
                                filename=log_file,
                                level=logging.DEBUG,
                                filemode='w')
            logging.getLogger().addHandler(logging.StreamHandler())
        else:
            self.testing_set = tf_flags.testing_set

            self.output = self._build_test()

    def _make_aux_dirs(self):
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

    def _build_training(self):
        self.output = Unet(name="UNet", in_data=self.input_data, reuse=False)

        # loss
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.input_masks, logits=self.output))

        # 定义Adam优化器
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, name="opt")

        # summary
        tf.summary.scalar('loss', self.loss)

        self.summary = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(
            self.summary_dir, graph=self.sess.graph)

        self.saver = tf.train.Saver(max_to_keep=10, name=self.saver_name)
        self.summary_proto = tf.Summary()

    def _build_test(self):
        output = Unet(name="UNet", in_data=self.input_data, reuse=False)

        self.Saver = tf.train.Saver(max_to_keep=10, name=self.saver_name)

        return output

    def train(self, batch_size, training_steps, summary_steps, checkpoint_steps, save_steps):
        """
        训练
        :param batch_size:
        :param training_steps:  训练迭代步骤数量
        :param summary_steps:  保存summary的步长
        :param checkpoint_steps: 保存checkpoint文件步长
        :param save_steps: 保存图像步长
        :return: None
        """
        step_num = 0

        # restore
        latest_checkpoint = tf.train.latest_checkpoint("model_output_20180314110555/checkpoint")
        if latest_checkpoint:
            step_num = int(os.path.basename(latest_checkpoint).split("-")[1])
            assert step_num > 0, "please ensure checkpoint format is model-*.*."

            self.saver.restore(self.sess, latest_checkpoint)
            logging.info("{}: Resume training from step {}. Loaded checkpoint {}".format(datetime.now(),
                                                                                         step_num, latest_checkpoint))
        else:
            self.sess.run(tf.global_variables_initializer())
            logging.info("{}: Init new training".format(datetime.now()))

        # 定义Read_TFRecords类的对象tf_reader
        tf_reader = read_tfRecords.Read_tfRecord(filename=os.path.join(self.training_set,
                                                                       "Carvana.tfRecords"),
                                                 batch_size=batch_size, image_h=self.image_h,
                                                 image_c=self.image_c)
        # [batch_size,512,512,3], [batch_size,324,324,3]
        images, images_masks = tf_reader.read()
        logging.info("{}: Done init data generators".format(datetime.now()))

        # 线程协调器
        self.coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

        try:
            c_time = time.time()
            lrval = self.learning_rate
            for c_step in range(step_num + 1, training_steps + 1):
                if c_step % 5000 == 0:
                    lrval = self.learning_rate * .5

                batch_images, batch_images_masks = self.sess.run([images, images_masks])

                # 实现反向传播需要的参数
                c_feed_dict = {
                    self.input_data: batch_images,
                    self.input_masks: batch_images_masks,
                    self.lr: lrval
                }
                self.sess.run(self.opt, feed_dict=c_feed_dict)

                # 保存summary
                if c_step % summary_steps == 0:
                    c_summary = self.sess.run(self.summary, feed_dict=c_feed_dict)
                    self.writer.add_summary(c_summary, c_step)

                    e_time = time.time() - c_time
                    time_periter = e_time / summary_steps
                    logging.info("{}: Interaction_{} ({:.4f}/iter){}".format(
                        datetime.now(), c_step, time_periter,
                        self._print_summary(c_summary)))
                    c_time = time.time()

                # 检查checkpoint
                if c_step % checkpoint_steps == 0:
                    self.saver.save(self.sess,
                                    os.path.join(self.checkpoint_dir, self.checkpoint_prefix),
                                    global_step=c_step)
                    logging.info("{}: Iteraction_{} Saved Checkpoint".format(
                        datetime.now(), c_step))

                # 保存图片
                if c_step % save_steps == 0:
                    # 预测的分割mask和ground truth的mask
                    _, output_masks, input_masks = self.sess.run(
                        [self.input_data, self.output, self.input_masks],
                        feed_dict=c_feed_dict)
                    # [batch_size,324,324,1]
                    save_images(None, output_masks, input_masks,
                                # self.sample_dir：train_results
                                input_path='./{}/input_{:04d}.png'.format(self.sample_dir, c_step),
                                image_path='./{}/train_{:04d}.png'.format(self.sample_dir, c_step))

        except KeyboardInterrupt:
            print('Interrupted')
            self.coord.request_stop()
        except Exception as e:
            self.coord.request_stop(e)
        finally:
            self.coord.request_stop()
            self.coord.join(threads)

        logging.info("{}: Done training".format(datetime.now()))

    def _print_summary(self, summary_string):
        # 解析loss summary中的值
        self.summary_proto.ParseFromString(summary_string)
        result = []
        for val in self.summary_proto.value:
            result.append("({}={})".format(val.tag, val.simple_value))
        return " ".join(result)

    def load(self, checkpoint_name=None):
        # restore checkpoint
        print("{}: Loading checkpoint...".format(datetime.now())),
        if checkpoint_name:
            checkpoint = os.path.join(self.checkpoint_dir, checkpoint_name)
            self.saver.restore(self.sess, checkpoint)
            print(" loaded {}".format(checkpoint_name))
        else:
            # restore latest model
            latest_checkpoint = tf.train.latest_checkpoint(
                self.checkpoint_dir)
            if latest_checkpoint:
                self.saver.restore(self.sess, latest_checkpoint)
                print(" loaded {}".format(os.path.basename(latest_checkpoint)))
            else:
                raise IOError(
                    "No checkpoints found in {}".format(self.checkpoint_dir))

    def test(self):
        image_name = glob.glob(os.path.join(self.testing_set, "*.jpg"))
        # image: 1 * 512 * 512 * 3
        image = np.reshape(cv2.resize(cv2.imread(image_name[0], 0),
                                      (self.image_h, self.image_w)),
                           (3, self.image_h, self.image_w, self.image_c)) / 255
        print("{}: Done init data generators".format(datetime.now()))

        c_feed_dict = {
            self.input_data: image
        }
        # output_masks: 1 * 324 * 324 * 3
        output_masks = self.sess.run(
            self.output, feed_dict=c_feed_dict
        )

        return image, output_masks
