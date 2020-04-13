import tensorflow as tf
import logging
from utils import label_utils
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import io
from main import conf
logger = logging.getLogger(__name__)

# e_output[seq,image_width/4]，对e_output就是对每个字的注意力的概率分布
# 几个细节：
# 1、raw_image是resize之后的
# 2、image_width/4，也就是256/4，是64
# 3、seq最长是30，但是也可能是提前结束ETX了



class TBoardVisual(Callback):

    def __init__(self, tag,tboard_dir,charset,args):
        super().__init__()
        self.tag = tag
        self.args = args
        self.tboard_dir = tboard_dir
        self.charset = charset
        self.font = ImageFont.truetype("data/font/simsun.ttc", 10)  # 设置字体

    def on_batch_end(self, batch, logs=None):

        if batch%self.args.debug_step!=0:return

        # 随机取3张
        # logger.debug("self.validation_data:%r",self.validation_data)

        # 先重排一下验证数据
        np.random.shuffle(self.validation_data.data_list)

        data = self.validation_data.data_list[:9]
        images,labels = self.validation_data.load_image_label(data)

        e_outputs = self.model.get_layer('attention_layer').output[1] #注意力层返回的是：return c_outputs, e_outputs


        functor = K.function([self.model.input[0],self.model.input[1],K.learning_phase()], [e_outputs,self.model.output])

        # 调试用
        # import pdb;
        # pdb.set_trace()

        # 返回注意力分布[B,Seq,W/4]和识别概率[B,Seq,Charset]
        e_outputs_data,output_prob = functor([ images,labels[:,:-1,:],True])

        logger.debug("预测结果：images shape = %r,e_outputs_data=%r",images.shape,e_outputs_data.shape)
        writer = tf.summary.FileWriter(self.tboard_dir)

        for i in range(len(images)):
            # 对一张图片
            image = images[i]
            label = labels[i]


            label = label_utils.prob2str(label,self.charset)
            pred  = label_utils.prob2str(output_prob[i],self.charset)

            logger.debug("label字符串:%r",label)
            logger.debug("pred字符串 :%r",pred)


            # logger.debug("image.shape:%r,e_output.shape:%r",image.shape,e_output.shape)
            tf_img = self.make_image(image, e_outputs_data[i],label,pred)
            summary = tf.Summary(value=[tf.Summary.Value(tag="{}/{}".format(self.tag,i),image=tf_img)])
            writer.add_summary(summary)

        writer.close()

        return

    # 画一张图
    def make_image(self,raw_image,e_output,label,pred):

        # 对每个时间片 1/seq
        for i,words_distribute in enumerate(e_output):

            # 如果预测结果是空格/stx/etx，也就是无效字符，就不画注意力焦点了
            if i>len(pred) or \
                pred[i] == conf.CHAR_STX or \
                pred[i]==conf.CHAR_ETX or \
                pred[i] == conf.CHAR_NULL: continue

            # 对每个字符的分布
            for w_distribute in words_distribute:
                # 找到64个encoder序列中，哪个位置概率最大
                max_index = np.argmax(w_distribute)
                x = max_index*4 # 4直接硬编码了，图像宽度缩小4倍
                y = 16 # 16也是直接硬编码了
                # logger.debug("注意力位置(%d,%d)",x,y)
                cv2.circle(raw_image,(x,y),1, (0, 0, 255),1)

        # 把样本和识别写到图上
        height, width, channel = raw_image.shape
        image = Image.fromarray(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        draw.text((2,2)  , label,'red', self.font)
        draw.text((128,2), pred, 'red', self.font)

        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height,
                                width=width,
                                colorspace=channel,
                                encoded_image_string=image_string)