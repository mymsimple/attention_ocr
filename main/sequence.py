from keras.preprocessing.sequence import Sequence
from utils import image_utils, label_utils
import os,logging
import numpy as np

logger = logging.getLogger("SequenceData")

# 自定义的数据加载器
class SequenceData(Sequence):
    def __init__(self, label_file, charset_file, batch_size=32):
        self.label_file = label_file
        self.batch_size = batch_size
        self.charsets = label_utils.get_charset(charset_file)

    #返回长度，我理解是一个epoch中的总步数
    def __len__(self):
        return len(self.images_labels)/self.batch_size

    #即通过索引获取a[0],a[1]这种
    def __getitem__(self, idx):
        image_names, label_ids = self.images_labels[idx: idx+self.batch_size]
        images = image_utils.resize_batch_image(image_names)
        return {'images': images, 'labels': label_ids}

    # 一次epoch后，重新shuffle一下样本
    def on_epoch_end(self):
        np.random.shuffle(self.images_labels)

    # 初始加载样本：即每一个文件的路径和对应的识别文字
    # 额外做两件事：
    # 1、check每一个图片文件是否存在
    # 2、看识别文字的字不在字表中，剔除这个样本
    def initialize(self):
        image_file_names, labels = label_utils.read_labeled_image_list(self.label_file,self.charsets, unknow_char)
        self.images_labels = list(zip(self.image_file_names, labels))
        np.random.shuffle(self.images_labels)


