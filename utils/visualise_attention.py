import tensorflow as tf
import logging
from tensorflow.keras import backend as K
import numpy as np
import cv2
from PIL import Image
import io

logger = logging.getLogger(__name__)

# e_output[seq,image_width/4]，对e_output就是对每个字的注意力的概率分布
# 几个细节：
# 1、raw_image是resize之后的
# 2、image_width/4，也就是256/4，是64
# 3、seq最长是30，但是也可能是提前结束ETX了
def make_image(raw_image,e_output):

    for seq_img_distribute in e_output:
        # seq_img_distribute,是对一个字的图像的概率分布，比如识别出来是ABC，图像是256，那么，就会有3个分布，可以再图像上根据这个分布画3个焦点
        # logger.debug(seq_img_distribute.shape) # =>64

        max_index = np.argmax(seq_img_distribute)
        x = max_index*4 # 4直接硬编码了
        y = 16 # 16也是直接硬编码了

        logger.debug("注意力位置(%d,%d)",x,y)
        cv2.circle(raw_image,(x,y),2, (0, 0, 255),2)

    height, width, channel = raw_image.shape

    image = Image.fromarray(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))

    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


class TBoardVisual(tf.keras.callbacks.Callback):
    def __init__(self, tag,tboard_dir):
        super().__init__()
        self.tag = tag
        self.tboard_dir = tboard_dir

    def on_epoch_end(self, epoch, logs):
        # 随机取3张
        # logger.debug("self.validation_data:%r",self.validation_data)


        data = self.validation_data.data_list[:6]
        images,labels = self.validation_data.load_image_label(data)

        e_outputs = self.model.get_layer('attention_layer').output[1] #注意力层返回的是：return c_outputs, e_outputs

        functor = K.function([self.model.input[0],self.model.input[1],K.learning_phase()], [e_outputs])

        # 调试用
        # import pdb;
        # pdb.set_trace()

        e_outputs_data = functor([ images,labels[:,:-1,:],True])

        # logger.debug("images shape = %r,e_outputs_data=%r",
        #              images.shape,e_outputs_data[0].shape)
        writer = tf.summary.FileWriter(self.tboard_dir)

        for i in range(len(images)):
            image = images[i]
            e_output = e_outputs_data[0][i]
            # logger.debug("image.shape:%r,e_output.shape:%r",image.shape,e_output.shape)
            tf_img = make_image(image, e_output)
            summary = tf.Summary(value=[tf.Summary.Value(tag="{}/{}".format(self.tag,i),image=tf_img)])
            writer.add_summary(summary, epoch)

        writer.close()

        return