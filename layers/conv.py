from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.backend import squeeze

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
# 自定义的卷基层，32x100 => 1 x 25，即（1/32，1/4)
def conv_layer(img_input):
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)

    # Block 4
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = MaxPooling2D((2, 1), strides=(2, 1), name='block4_pool')(x) # <------ pool kernel is (2,1)!!!!!

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = BatchNormalization()(x)

    # Block 6
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 1), strides=(2, 1), name='block4_pool')(x) # <------ pool kernel is (2,1)!!!!!

    # Block 7
    x = Conv2D(512, (2, 2), strides=[2, 1], activation='relu', padding='same', name='block7_conv1')(x)

    return squeeze(x,axis=1)


# conv1 = self.__conv_stage(inputdata=inputdata, out_dims=64, name='conv1')  # batch*16*50*64
#
# logger.debug("CNN层第1层输出的Shape:%r", self.shape(conv1))
#
# conv2 = self.__conv_stage(inputdata=conv1, out_dims=128, name='conv2')  # batch*8*25*128
#
# logger.debug("CNN层第2层输出的Shape:%r", self.shape(conv2))
#
# conv3 = self.conv2d(inputdata=conv2, out_channel=256, kernel_size=3, stride=1, use_bias=False,
#                     name='conv3')  # batch*8*25*256
# relu3 = self.relu(conv3)  # batch*8*25*256
#
# logger.debug("CNN层第3层输出的Shape:%r", self.shape(relu3))
#
# conv4 = self.conv2d(inputdata=relu3, out_channel=256, kernel_size=3, stride=1, use_bias=False,
#                     name='conv4')  # batch*8*25*256
# relu4 = self.relu(conv4)  # batch*8*25*256
# # 这里诡异啊，池化用的[2,1]，一般都是正方形池化啊
# max_pool4 = self.maxpooling(inputdata=relu4, kernel_size=[2, 1], stride=[2, 1], padding='VALID')  # batch*4*25*256
#
# logger.debug("CNN层第4层输出的Shape:%r", self.shape(max_pool4))
#
# conv5 = self.conv2d(inputdata=max_pool4, out_channel=512, kernel_size=3, stride=1, use_bias=False,
#                     name='conv5')  # batch*4*25*512
# relu5 = self.relu(conv5)  # batch*4*25*512
# if self.phase.lower() == 'train':
#     bn5 = self.layerbn(inputdata=relu5, is_training=True)
# else:
#     bn5 = self.layerbn(inputdata=relu5, is_training=False)  # batch*4*25*512
#
# logger.debug("CNN层第5层输出的Shape:%r", self.shape(bn5))
#
# conv6 = self.conv2d(inputdata=bn5, out_channel=512, kernel_size=3, stride=1, use_bias=False,
#                     name='conv6')  # batch*4*25*512
# relu6 = self.relu(conv6)  # batch*4*25*512
# if self.phase.lower() == 'train':
#     bn6 = self.layerbn(inputdata=relu6, is_training=True)
# else:
#     bn6 = self.layerbn(inputdata=relu6, is_training=False)  # batch*4*25*512
# max_pool6 = self.maxpooling(inputdata=bn6, kernel_size=[2, 1], stride=[2, 1])  # batch*2*25*512
#
# logger.debug("CNN层第6层输出的Shape:%r", self.shape(max_pool6))
#
# conv7 = self.conv2d(inputdata=max_pool6, out_channel=512, kernel_size=2, stride=[2, 1], use_bias=False,
#                     name='conv7')  # batch*1*25*512
# # ？？？怎么就从batch*2*25*512=>batch*1*25*512了？只是个卷基层啊？晕了
# relu7 = self.relu(conv7)  # batch*1*25*512
# ------------------
# conv = self.conv2d(inputdata=inputdata, out_channel=out_dims, kernel_size=3, stride=1, use_bias=False, name=name)
# relu = self.relu(inputdata=conv)
# max_pool = self.maxpooling(inputdata=relu, kernel_size=2, stride=2)