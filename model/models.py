import tensorflow as tf


# input shape: [batch_size, 512, 512, 1]
# output shape: [batch_size, 324, 324, 1]

def Unet(name, in_data, reuse=False):
    """
    实现unet网络
    :param name: UNet
    :param in_data: 初始矩阵模型
    :param reuse:
    :return: unet网络运行后的矩阵模型
    """
    assert in_data is not None

    with tf.variable_scope(name, reuse=reuse):
        # [None,512,512,1]==>[None,510,510,64]
        conv1_1 = tf.layers.conv2d(in_data, 64, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())
        # [None,510,510,64]==>[None,508,508,64]
        conv1_2 = tf.layers.conv2d(conv1_1, 64, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())
        crop1 = tf.keras.layers.Cropping2D(cropping=((90, 90), (90, 90)))(conv1_2)
        # [None,508,508,64]==>[None,254,254,64]
        pool1 = tf.layers.max_pooling2d(conv1_2, 2, 2)

        # [None,254,254,64]==>[None,252,252,128]
        conv2_1 = tf.layers.conv2d(pool1, 128, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())
        # [None,252,252,128]==>[None,250,250,128]
        conv2_2 = tf.layers.conv2d(conv2_1, 128, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xvaier_initializer())
        crop2 = tf.keras.layers.Cropping2D(cropping=((41, 41), (41, 41)))(conv2_2)
        # [None,250,250,128]==>[None,125,125,128]
        pool2 = tf.layers.max_pooling2d(conv2_2, 2, 2)

        # [None,125,125,128]==>[None,123,123,256]
        conv3_1 = tf.layers.conv2d(pool2, 256, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())
        # [None,123,123,256]==>[None,121,121,256]
        conv3_2 = tf.layers.conv2d(conv3_1, 256, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())
        crop3 = tf.keras.layers.Cropping2D(cropping=((16, 17), (16, 17)))(conv3_2)
        # [None,121,121,256]==>[None,60,60,256]
        pool3 = tf.layers.max_pooling2d(conv3_2, 2, 2)

        # [None,60,60,256]==>[None,58,58,512]
        conv4_1 = tf.layers.conv2d(pool3, 512, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())

        # [None,58,58,512]==>[None,56,56,512]
        conv4_2 = tf.layers.conv2d(conv4_1, 512, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())

        # [None,56,56,512]
        drop4 = tf.layers.dropout(conv4_2)
        # todo? Arguments: inputs, rate=0.5.

        # [None,56,56,512]==>[None,48,48,512]
        crop4 = tf.keras.layers.Cropping2D(cropping=((4, 4), (4, 4)))(drop4)

        # [None,56,56,512]==>[None,28,28,512]
        pool4 = tf.layers.max_pooling2d(drop4, 2, 2)

        # [None,28,28,512]==>[None,26,26,1024]
        conv5_1 = tf.layers.conv2d(pool4, 1024, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())

        # [None,26,26,1024]==>[None,24,24,1024]
        conv5_2 = tf.layers.conv2d(conv5_1, 1024, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())

        # [None,24,24,1024]
        drop5 = tf.layers.dropout(conv5_2)

        # 解码器：
        # 上采样：[None,24,24,1024]==>[None,48,48,1024]
        up6_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(drop5)

        # 每次上采样后有一个2*2的same卷积降维，[None,48,48,1024]=>[None,48,48,512]
        up6 = tf.layers.conv2d(up6_1, 512, 2, padding="SAME", activation=tf.nn.relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer())
        # 每次上采样后与编码器对应分辨率图像合并(concatenate)=>[None,48,48,1024]
        merge6 = tf.concat([crop4, up6], axis=3)  # concat channel
        # values: A list of Tensor objects or a single Tensor.

        # [None,48,48,1024]==>[None,46,46,512]
        conv6_1 = tf.layers.conv2d(merge6, 512, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())
        # [None, 46, 46, 1024]==[None,44,44,512]
        conv6_2 = tf.layers.conv2d(conv6_1, 512, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())

        # 上采样：[None,44,44,512]==>[None,88,88,512]
        up7_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6_2)
        # [None,88,88,256]
        up7 = tf.layers.conv2d(up7_1, 256, 2, padding="SAME", activation=tf.nn.relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer())

        # [None,88,88,512]
        merge7 = tf.concat([crop3, up7], axis=3)  # concat channel

        # [None,88,88,512]==>[None,86,86,256]
        conv7_1 = tf.layers.conv2d(merge7, 256, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())

        # [None,86,86,256]==[None,84,84,256]
        conv7_2 = tf.layers.conv2d(conv7_1, 256, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())

        # [None,84,84,256]==[None,168,168,256]
        up8_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7_2)
        # [None,168,168,256]==>[None,168,168,128]
        up8 = tf.layers.conv2d(up8_1, 128, 2, padding="SAME", activation=tf.nn.relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer())
        # [None,168,168,256]
        merge8 = tf.concat([crop2, up8], axis=3)  # concat channel

        # [None,168,168,256]==>[None,166,166,128]
        conv8_1 = tf.layers.conv2d(merge8, 128, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())

        # [None,166,166,256]==>[None,164,164,128]
        conv8_2 = tf.layers.conv2d(conv8_1, 128, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())

        # [None,164,164,128]==>[None,328,328,128]
        up9_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8_2)
        # [None,328,328,64]
        up9 = tf.layers.conv2d(up9_1, 64, 2, padding="SAME", activation=tf.nn.relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer())

        # [None,328,328,128]
        merge9 = tf.concat([crop1, up9], axis=3)  # concat channel

        # [None,328,328,128]==>[None,326,326,64]
        conv9_1 = tf.layers.conv2d(merge9, 64, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())

        # [None,326,326,64]==>[None,324,324,64]
        conv9_2 = tf.layers.conv2d(conv9_1, 64, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())

        # [None,324,324,2]
        conv9_3 = tf.layers.conv2d(conv9_2, 2, 3, padding="SAME", activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())

        # [None,324,324,2]=>[None,324,324,1]
        conv10 = tf.layers.conv2d(conv9_3, 1, 1,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
        # -> 1 channel.

    return conv10
