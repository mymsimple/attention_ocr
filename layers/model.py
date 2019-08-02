# 之前报错：AttributeError: 'Bidirectional' object has no attribute 'outbound_nodes'
# from keras.layers import Bidirectional,Input, GRU, Dense, Concatenate, TimeDistributed,Reshape
# from tensorflow.python.keras.applications.vgg19 import VGG19
# from keras.applications.vgg19 import VGG19
# 原因是不能用keras自带的vgg19+keras自带的bidirectional，靠，肯定是版本不兼容的问题
# 切换到下面的就好了，之前还是试验了用tf的bidirectional+keras的vgg19，也是不行，报错：AttributeError: 'Node' object has no attribute 'output_masks'
# 靠谱的组合是：tf的bidirectional+tf的vgg19
from tensorflow.python.keras.layers import Bidirectional,Masking,Input, GRU, Dense, Concatenate, TimeDistributed,ZeroPadding1D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.optimizers import adam
from layers import conv
from layers.attention import AttentionLayer
import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger("Model")

# # seq需要加padding
# # 输入的图像需要加 0 padding
# def padding_wrapper(conv_output,mask_value):
#     paddings = [[0, 0], [0, 50 - tf.shape(conv_output)[0]],[0,0]]
#     # 给卷基层的输出增加padding，让他可以输入到bi-gru里面去
#     conv_output_with_padding = tf.pad(conv_output, paddings=paddings, constant_values=mask_value)
#     #print("conv_output_with_padding.shape")
#     conv_output_with_padding.set_shape([None, 50, 512])  # 靠！还可以这么玩呢！给丫设置一个shape。
#     return conv_output_with_padding


# 焊接vgg和lstm，入参是vgg_conv5返回的张量
def model(conf):

    # 高度和长度都不定，是None，虽然可以定义高度(32,None,3)，但是一般都是从左到右定义None的，所以第一个写32也就没有意义了
    input_image = Input(shape=(conf.INPUT_IMAGE_HEIGHT,conf.INPUT_IMAGE_WIDTH,3), name='input_image') #高度固定为32，3通道

    # input_image = Masking(0.0)(input_image) <----- 哭：卷基层不支持Mask，欲哭无泪：TypeError: Layer block1_conv1 does not support masking, but was passed an input_mask: Tensor("masking/Any_1:0", shape=(?, 32, ?), dtype=bool)
    # 1. 卷基层，输出是(Batch,Width/32,512)
    conv_output = conv.conv_layer(input_image)

    # 经过padding后，转变为(Batch,50,512)
    # conv_output_with_padding = Lambda(padding_wrapper,arguments={'mask_value':conf.MASK_VALUE})(conv_output)
    # conv_output_mask = Masking(conf.MASK_VALUE)(conv_output)

    # 2.Encoder Bi-GRU编码器
    encoder_bi_gru = Bidirectional(GRU(conf.GRU_HIDDEN_SIZE,return_sequences=True,return_state=True,name='encoder_gru'),name='bidirectional_encoder')
    encoder_out, encoder_fwd_state, encoder_back_state = encoder_bi_gru(conv_output)

    # 3.Decoder GRU解码器，使用encoder的输出当做输入状态
    decoder_inputs = Input(shape=(None,conf.CHARSET_SIZE), name='decoder_inputs')
    # masked_decoder_inputs = Masking(conf.MASK_VALUE)(decoder_inputs)
    decoder_gru = GRU(conf.GRU_HIDDEN_SIZE*2, return_sequences=True, return_state=True, name='decoder_gru')
    decoder_out, decoder_state = decoder_gru(decoder_inputs, initial_state=Concatenate(axis=-1)([encoder_fwd_state, encoder_back_state]))

    # 4.Attention layer注意力层
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_out, decoder_out])

    # concat Attention的输出 + GRU的输出
    decoder_concat_input = Concatenate(axis=-1, name='concat123_layer')([decoder_out, attn_out])

    # 5.Dense layer输出层
    dense = Dense(conf.CHARSET_SIZE, activation='softmax', name='softmax_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense_time(decoder_concat_input)

    #整个模型
    model = Model(inputs=[input_image, decoder_inputs], outputs=decoder_pred)
    opt = adam(lr=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    model.summary()

    return model