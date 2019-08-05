import tensorflow as tf


class Read_tfRecord(object):
    def __init__(self, filename, batch_size=64, image_h=256, image_w=256, image_c=3,
                 num_threads=8, capacity_factor=3, min_after_dequeue=1000):
        """
        :param filename: tfRecord文件路径
        :param batch_size:
        :param image_h: 图像高度
        :param image_w: 图像长度
        :param image_c: 颜色通道
        :param num_threads:
        :param capacity_factor:
        :param min_after_dequeue:
        """
        self.filename = filename
        self.batch_size = batch_size
        self.image_h = image_h
        self.image_w = image_w
        self.image_c = image_c
        self.num_threads = num_threads
        self.capacity_factor = capacity_factor
        self.min_after_dequeue = min_after_dequeue

    def read(self):
        """
        从tfRecord文件中读取数据
        :return: tf.train.batch/tf.train.shuffle_batch object
        """
        # 第一步：生成文件名队列
        filename_queue = tf.data.Dataset.from_tensor_slices([self.filename])

        # 第二步：创建读取器
        reader = tf.TFRecordReader()
        key, serialized_example = reader.read(filename_queue)

        # 第三步：将Example协议缓冲区(protocol buffer)解析为张量字典
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               "image_raw": tf.FixedLenFeature([], tf.string),
                                               "image_label": tf.FixedLenFeature([], tf.string)
                                           })
        # 第四步：对图像张量解码并进行resize、归一化处理
        image_raw = tf.image.decode_jpeg(features["image_raw"], channels=self.image_c,
                                         name="decode_image")
        image_label = tf.image.decode_jpeg(features["image_label"], channels=self.image_c,
                                           name="decode_image")
        # 根据unet输出的矩阵resize：[1918,1280]->[512,512], [1918,1280]-->[324,324]
        if self.image_h is not None and self.image_w is not None:
            image_raw = tf.image.resize_images(image_raw, [self.image_h, self.image_w],
                                               method=tf.image.ResizeMethod.BICUBIC)
            image_label = tf.image.resize_images(image_label, [324, 324],
                                                 method=tf.image.ResizeMethod.BICUBIC)

        # 归一化处理：像素值->tf.float32
        image_raw = tf.cast(image_raw, tf.float32) / 255
        image_label = tf.cast(image_label, tf.float32) / 255

        # 第五步：tf.train.shuffle_batch将训练集打乱，每次返回batch_size份数据
        input_data, input_masks = tf.train.shuffle_batch([image_raw, image_label],
                                                      batch_size=self.batch_size,
                                                      capacity=self.min_after_dequeue,
                                                      num_threads=self.num_threads,
                                                      name="images")
        return input_data, input_masks