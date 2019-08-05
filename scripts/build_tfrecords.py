# 原图像及其mask图像以序列化为字符串的形式写入到一个tfrecord文件中

import os
# import glob # Can use os.listdir(data_dir) replace glob.glob(os.path.join(data_dir, "*.jpg"))
# to get every image name, do not include path.
import tensorflow as tf

if __name__ == '__main__':
    # 灰度图像所在目录
    data_root = "../datasets/CarvanaImages"
    # 将CarvanaImages/train目录下文件名保存进image_names列表中
    image_names = os.listdir(os.path.join(data_root, "train"))  # return JPEG image names.

    # 创建../datasets/tfrecords/目录
    if not os.path.exists(os.path.join("../datasets", "tfrecords")):
        os.makedirs(os.path.join("../datasets", "tfrecords"))

    # tf.python_io.TFRecordWriter.__init__(path)
    # 第1步：创建文件../datasets/tfrecords/Carvana.tfrecords，为该文件创建TFRecordWriter准备写入数据
    writer = tf.python_io.TFRecordWriter(os.path.join("../datasets", "tfrecords",
                                                      "Carvana.tfrecords"))

    for image_name in image_names:
        # 得到训练原图像路径 e.g.datasets/CarvanaImages/train/0cdf5b5d0ce1_01.jpg
        image_raw_file = os.path.join(data_root, "train", image_name)
        # 得到训练图像mask 路径 e.g.datasets/CarvanaImages/train_masks/0cdf5b5d0ce1_01_mask.jpg
        image_label_file = os.path.join(data_root, "train_masks",
                                        image_name[:-4] + "_mask.jpg")

        # 第2步：读取没有经过解码的原图及其mask(即label)
        # tf.gfile.FastGFile('path',mode).read()函数：读取没有经过解码的原始图像，如果要显示读入的图像，那就需要经过解码过程，读取的图像是一个字符串，没法显示
        # tensorflow里面提供解码的函数有两个，tf.image.decode_jepg和tf.image.decode_png分别用于解码jpg格式和png格式的图像进行解码，得到图像的像素值
        image_raw = tf.gfile.FastGFile(image_raw_file, 'rb').read()  # image data type is string.
        # read and binary.
        image_label = tf.gfile.FastGFile(image_label_file, 'rb').read()

        # tfrecord文件包含了tf.train.Example 协议缓冲区(protocol buffer，协议缓冲区包含了特征 Features)。
        # 你可以写一段代码获取你的数据， 将数据填入到Example协议缓冲区(protocol buffer)，将协议缓冲区序列化为一个字符串，
        # 并且通过tf.python_io.TFRecordWriter class写入到TFRecords文件。

        # tf.train.Example(features=tf.train.Features(feature={key:value,key:value,...})
        # value类型：tfrecord支持整型、浮点数和二进制三种格式，分别是
        # tf.train.Feature(int64_list=tf.train.Int64List(value=[int_scalar]))
        # tf.train.Feature(bytes_list=tf.train.BytesList(value=[array_string_or_byte]))
        # tf.train.Feature(bytes_list=tf.train.FloatList(value=[float_scalar]))
        # write bytes to Example proto buffer.

        # 第3步：将raw及其label填入到tfrecord文件的Example缓冲区中
        example = tf.train.Example(features=tf.train.Features(feature={
            "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
            "image_label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label]))
        }))

        # 第4步：将Example缓冲区序列化的写入到datasets/tfrecords/Carvana.tfrecords文件中
        writer.write(example.SerializeToString())  # Serialize To String

    writer.close()
