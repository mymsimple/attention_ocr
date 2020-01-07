from utils import image_utils, label_utils
import logging,math,os
import numpy as np

from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# from keras.utils import Sequence
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical

import time

logger = logging.getLogger("SequenceData")


# 自定义的数据加载器
# 几个细节：
# - 不用用多进程方式加载，不知道为何总是卡住，改成multiprocess=False,即使用多线程就好了,参考：https://stackoverflow.com/questions/54620551/confusion-about-multiprocessing-and-workers-in-keras-fit-generator-with-window
# - on_epoch_end确实是所有的样本都轮完了，才回调一次，而，steps_per_epoch改变的是多久callback回调一次，这个可以调的更小一些，两者没关系
class SequenceData(Sequence):
    def __init__(self, name,label_file, charset_file,conf,args,batch_size=32):
        self.conf = conf
        self.name = name
        self.label_file = label_file
        self.batch_size = batch_size
        self.charsets = label_utils.get_charset(charset_file)
        self.initialize(conf,args)
        self.start_time = time.time()

    # 返回长度，我理解是一个epoch中的总步数
    # 'Denotes the number of batches per epoch'
    def __len__(self):
        return int(math.ceil(len(self.data_list) / self.batch_size))

    # 即通过索引获取a[0],a[1]这种,idx是被shuffle后的索引，你获取数据的时候，需要[idx * self.batch_size : (idx + 1) * self.batch_size]
    # 2019.12.30,piginzoo，
    def __getitem__(self, idx):
        start_time = time.time()
        batch_data_list = self.data_list[ idx * self.batch_size : (idx + 1) * self.batch_size]

        images_labelids = label_utils.process_lines(self.charsets,batch_data_list)

        # print(self.name,"Sequence PID:", os.getpid(),",idx=",idx)
        # unzip的结果是 [(1,2,3),(a,b,c)]，注意，里面是个tuple，而不是list，所以用的时候还要list()转化一下
        # zip(*xxx）操作是为了解开[(a,b),(a,b),(a,b)]=>[a,a,a][b,b,b]
        image_names, label_ids = list(zip(*images_labelids))

        # 读取图片，高度调整为32，宽度用黑色padding
        images = image_utils.read_and_resize_image(list(image_names),self.conf)

        # labels是[nparray([<3773>],[<3773>],[<3773>]),...]，是一个数组，里面是不定长的3370维度的向量,(N,3770),如： (18, 3861)
        labels = list(label_ids)
        labels = pad_sequences(labels,maxlen=self.conf.MAX_SEQUENCE,padding="post",value=0)
        labels = to_categorical(labels,num_classes=len(self.charsets))        #to_categorical之后的shape： [N,time_sequence(字符串长度),3773]

        logger.debug("进程[%d],加载一个批次数据，idx[%d],耗时[%f]",
                    os.getpid(),
                    idx,
                    time.time()-start_time)

        # 识别结果是STX,A,B,C,D,ETX，seq2seq的decoder输入和输出要错开1个字符
        # labels[:,:-1,:]  STX,A,B,C,D  decoder输入标签
        # labels[:,1: ,:]  A,B,C,D,ETX  decoder验证标签
        # logger.debug("加载批次数据：%r",images.shape)
        # logger.debug("Decoder输入：%r", labels[:,:-1,:])
        # logger.debug("Decoder标签：%r", labels[:,1:,:])
        return [images,labels[:,:-1,:]],labels[:,1:,:]

    # 一次epoch后，重新shuffle一下样本
    def on_epoch_end(self):
        np.random.shuffle(self.images_labels)
        duration = time.time() - self.start_time
        self.start_time = time.time()
        logger.debug("本次Epoch结束，耗时[%d]秒，重新shuffle数据",duration)

    # 初始加载样本：即每一个文件的路径和对应的识别文字
    # 额外做两件事：
    # 1、check每一个图片文件是否存在
    # 2、看识别文字的字不在字表中，剔除这个样本
    def initialize(self,conf,args):
        logger.info("开始加载[%s]样本和标注",self.name)
        start_time = time.time()
        self.data_list = label_utils.read_data_file(self.label_file,args.preprocess_num)

        logger.info("加载了[%s]样本:[%d]个,耗时[%d]秒", self.name, len(self.data_list),(time.time() - start_time))

        # logger.debug("使用[%d]个进程，开始并发处理所有的[%d]行标签数据", args.preprocess_num,len(data_list))
        # # 使用一个进程池来分别预处理所有的数据（加入STX/ETX，以及过滤非字表字
        # pool_size = args.preprocess_num # 把进程池数和要分箱的数量搞成一致
        # from functools import partial
        # func = partial(label_utils.process_lines, self.charsets) #  函数的功能就是：把一个函数的某些参数给固定住，返回一个新的函数：http://funhacks.net/explore-python/Functional/partial.html
        # pool = multiprocessing.Pool(processes=pool_size,maxtasksperchild=2,)
        # pool_outputs = pool.map(func, data_list) # 使用partial工具类，来自动划分这些数据到每个进程中
        # pool.close()  # no more tasks
        # pool.join()  # wrap up current tasks
        #
        # self.images_labels = [item for sublist in pool_outputs for item in sublist]
        #
        # logger.info("加载了[%s]样本:[%d]个,耗时[%d]秒", self.name, len(self.images_labels),(time.time() - start_time))
        #
        # np.random.shuffle(self.images_labels)
        #
        # logger.info("Shuffle[%s]样本数据", self.name)

