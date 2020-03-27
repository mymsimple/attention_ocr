from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Lambda
from tensorflow.keras.backend import squeeze
import logging

from utils.logger import _p
logger = logging.getLogger(__name__)

class Conv():
    '''
        #抽feature，用的cnn网络
        # https://blog.csdn.net/Quincuntial/article/details/77679463
        在CRNN模型中，通过采用标准CNN模型（去除全连接层）中的卷积层和最大池化层来构造卷积层的组件。
        这样的组件用于从输入图像中提取序列特征表示。在进入网络之前，所有的图像需要缩放到相同的高度。
        然后从卷积层组件产生的特征图中提取特征向量序列，这些特征向量序列作为循环层的输入。
        具体地，特征序列的每一个特征向量在特征图上按列从左到右生成。这意味着第i个特征向量是所有特征图第i列的连接。
        在我们的设置中每列的宽度固定为单个像素。

        # 由于卷积层，最大池化层和元素激活函数在局部区域上执行，因此它们是平移不变的。
        因此，特征图的每列对应于原始图像的一个矩形区域（称为感受野），并且这些矩形区域与特征图上从左到右的相应列具有相同的顺序。
        如图2所示，特征序列中的每个向量关联一个感受野，并且可以被认为是该区域的图像描述符。
        :param inputdata: eg. batch*32*100*3 NHWC format
          |
        Conv1  -->  H*W*64          #卷积后，得到的维度
        Relu1
        Pool1       H/2 * W/2 * 64  #池化后得到的维度
          |
        Conv2       H/2 * W/2 * 128
        Relu2
        Pool2       H/4 * W/4 * 128
          |
        Conv3       H/4 * W/4 * 256
        Relu3
          |
        Conv4       H/4 * W/4 * 256
        Relu4
        Pool4       H/8 * W/4 * 64
          |
        Conv5       H/8 * W/4 * 512
        Relu5
        BatchNormal5
          |
        Conv6       H/8 * W/4 * 512
        Relu6
        BatchNormal6
        Pool6       H/16 * W/4 * 512
          |
        Conv7
        Relu7       H/32 * W/4 * 512
          |
          20层
    '''


    #[N,1,256/4,512] => [N,256/4,512]
    def squeeze_wrapper(self,tensor):
        tensor = _p(tensor,"卷积层输出")
        return squeeze(tensor, axis=1)

    # 自定义的卷基层，32x100 => 1 x 25，即（1/32，1/4)
    def build(self, inputs):
        self.layers = []
        # Block 1
        x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(inputs)
        x = LeakyReLU()(x)
        # x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x) #1/2

        # Block 2
        x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
        x = LeakyReLU()(x)
        # x = BatchNormalization()(x
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x) #1/2

        # Block 3
        x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
        x = LeakyReLU()(x)
        # x = BatchNormalization()(x

        # Block 4
        x = Conv2D(256, (3, 3), padding='same', name='block4_conv1')(x)
        # x = BatchNormalization()(x
        x = LeakyReLU()(x)
        x = MaxPooling2D((2, 1), strides=(2, 1), name='block4_pool')(x) # 1/2 <------ pool kernel is (2,1)!!!!!

        # Block 5
        x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)

        # Block 6
        x = Conv2D(512, (3, 3), padding='same', name='block6_conv1')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 1), strides=(2, 1), name='block6_pool')(x) #1/2 <------ pool kernel is (2,1)!!!!!

        # Block 7
        x = Conv2D(512, (2, 2), strides=[2, 1], padding='same', name='block7_conv1')(x) #1/2
        x = LeakyReLU()(x)

        # 输出是(batch,1,Width/4,512),squeeze后，变成了(batch,Width/4,512)
        x = Lambda(self.squeeze_wrapper)(x)

        return x