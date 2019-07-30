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

# seq需要加padding
# 输入的图像需要加 0 padding

def padding_wrapper(conv_output,mask_value):
    paddings = [[0, 0], [0, 50 - tf.shape(conv_output)[0]],[0,0]]
    # 给卷基层的输出增加padding，让他可以输入到bi-gru里面去
    conv_output_with_padding = tf.pad(conv_output, paddings=paddings, constant_values=mask_value)
    conv_output_with_padding.set_shape([None, 50, 512])  # 靠！还可以这么玩呢！给丫设置一个shape。
    return conv_output_with_padding


# 焊接vgg和lstm，入参是vgg_conv5返回的张量
def model():

    # 高度和长度都不定，是None，虽然可以定义高度(32,None,3)，但是一般都是从左到右定义None的，所以第一个写32也就没有意义了
    input_image = Input(shape=(32,None,3), name='input_image') #高度固定为32，3通道

    # input_image = Masking(0.0)(input_image)
    # 1. 卷基层，输出是(Batch,Width/32,512)
    conv_output = conv.conv_layer(input_image)

    # 经过padding后，转变为(Batch,50,512)
    mask_value = 0
    conv_output_with_padding = Lambda(padding_wrapper,arguments={'mask_value':mask_value})(conv_output)
    conv_output_mask = Masking(mask_value)(conv_output_with_padding)
    print("conv_output_with_padding.shape:",conv_output_with_padding)

    # 2.Encoder Bi-GRU编码器
    encoder_bi_gru = Bidirectional(GRU(64,return_sequences=True,return_state=True,name='encoder_gru'),name='bidirectional_encoder')
    encoder_out, encoder_fwd_state, encoder_back_state = encoder_bi_gru(conv_output_with_padding,mask=conv_output_mask)

    # 3.Decoder GRU解码器，使用encoder的输出当做输入状态
    decoder_inputs = Input(shape=(None,3862), name='decoder_inputs')
    decoder_gru = GRU(64*2, return_sequences=True, return_state=True, name='decoder_gru')
    decoder_out, decoder_state = decoder_gru(decoder_inputs, initial_state=Concatenate(axis=-1)([encoder_fwd_state, encoder_back_state]))

    # 4.Attention layer注意力层
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_out, decoder_out])

    # concat Attention的输出 + GRU的输出
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

    # 5.Dense layer输出层
    dense = Dense(3862, activation='softmax', name='softmax_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense_time(decoder_concat_input)

    #整个模型
    model = Model(inputs=[input_image, decoder_inputs], outputs=decoder_pred)
    opt = adam(lr=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    model.summary()

    return model