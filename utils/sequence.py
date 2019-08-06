# from keras.preprocessing.sequence import Sequence
from tensorflow.python.keras.utils import Sequence
from utils import image_utils, label_utils
import logging,math
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical

logger = logging.getLogger("SequenceData")

# 自定义的数据加载器
class SequenceData(Sequence):
    def __init__(self, name,label_file, charset_file,conf,batch_size=32):
        self.conf = conf
        self.name = name
        self.label_file = label_file
        self.batch_size = batch_size
        self.charsets = label_utils.get_charset(charset_file)
        self.initialize(conf,)

    # 返回长度，我理解是一个epoch中的总步数
    # 'Denotes the number of batches per epoch'
    def __len__(self):
        return int(math.ceil(len(self.images_labels) / self.batch_size))

    # 即通过索引获取a[0],a[1]这种
    def __getitem__(self, idx):

        # unzip的结果是 [(1,2,3),(a,b,c)]，注意，里面是个tuple，而不是list，所以用的时候还要list()转化一下
        image_names, label_ids = list(zip(*self.images_labels[idx: idx+self.batch_size]))
        images = image_utils.read_and_resize_image(list(image_names),self.conf)

        # labels是[nparray([<3770>],[<3770>],[<3770>]),...]，是一个数组，里面是不定长的3370维度的向量,(N,3770),如： (18, 3861)
        labels = list(label_ids)
        labels = pad_sequences(labels,maxlen=self.conf.MAX_SEQUENCE,padding="post",value=0)
        labels = to_categorical(labels,num_classes=len(self.charsets))

        # 识别结果是STX,A,B,C,D,ETX，seq2seq的decoder输入和输出要错开1个字符
        # labels[:,:-1,:]  STX,A,B,C,D  decoder输入标签
        # labels[:,1: ,:]  A,B,C,D,ETX  doceder验证标签
        # logger.debug("加载批次数据：%r",images.shape)
        return [images,labels[:,:-1,:]],labels[:,1:,:]

    # 一次epoch后，重新shuffle一下样本
    def on_epoch_end(self):
        np.random.shuffle(self.images_labels)
        logger.debug("本次Epoch结束，重新shuffle数据")

    # 初始加载样本：即每一个文件的路径和对应的识别文字
    # 额外做两件事：
    # 1、check每一个图片文件是否存在
    # 2、看识别文字的字不在字表中，剔除这个样本
    def initialize(self,conf):
        image_file_names, labels = label_utils.read_labeled_image_list(self.label_file,self.charsets,conf)
        self.images_labels = list(zip(image_file_names, labels))
        logger.info("加载了%s样本： 标签[%d]个,图像[%d]张", self.name, len(labels), len(self.images_labels))
        np.random.shuffle(self.images_labels)
        logger.info("Shuffle样本数据")