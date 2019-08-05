# coding: utf-8
# 将所有的图片统一改为.jpg格式 <- Opencv can not read GIF image directly
import os
from PIL import Image

if __name__ == '__main__':
    # 数据集所在目录
    data_root = "../datasets"

    # 将train目录下文件名保存进image_names列表中
    image_names = os.listdir(os.path.join(data_root, "train"))

    # 分别处理train目录和train_masks目录下的文件
    for filename in ["train", "train_masks"]:
        for image_name in image_names:
            # train目录下的图片
            if filename is "train":
                # 得到每张训练原图片的文件名路径 e.g. ../datasets/train/0cdf5b5d0ce1_01.jpg
                image_file = os.path.join(data_root, filename, image_name)

                # PIL的Image类读取图像
                # convert()函数，用于不同模式图像之间的转换,L表示灰度转换为灰度图像
                image = Image.open(image_file).convert("L")

                # 创建../datasets/CarvanaImages/train/
                if not os.path.exists(os.path.join("../datasets/CarvanaImages", filename)):
                    os.makedirs(os.path.join("../datasets/CarvanaImages", filename))
                # 保存图片路径../datasets/CarvanaImages/train_masks/image_name(0cdf5b5d0ce1_01.jpg)
                image.save(os.path.join("../datasets/CarvanaImages", filename, image_name))

            if filename is "train_masks":
                # 得到每张训练mask图片的文件名路径：e.g. ../datasets/train_masks/0cdf5b5d0ce1_01_mask.gif
                image_file = os.path.join(data_root, filename, image_name[:-4] + "_mask.gif")
                image = Image.open(image_file).convert("L")

                # 创建../datasets/CarvanaImages/train_mask/
                if not os.path.exists(os.path.join("../datasets/CarvanaImages", filename)):
                    os.makedirs(os.path.join("../datasets/CarvanaImages", filename))

                # 保存图片路径：../datasets/CarvanaImages/train_masks/image_name(0cdf5b5d0ce1_01_mask.gif==>0cdf5b5d0ce1_01_mask.jpg)
                image.save(os.path.join("../datasets/CarvanaImages", filename,
                                        image_name[:-4] + "_mask.jpg"))
