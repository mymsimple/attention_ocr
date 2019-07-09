from keras.models import Sequential
from keras.layers.core import  Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from tensorflow.python.keras.layers import Input, GRU, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.python.keras.models import Model
from layers.attention import AttentionLayer
from tensorflow.python.keras import backend as K
import cv2,numpy as np

from keras.applications.vgg16 import VGG16


def vgg_16():
    model = VGG16(include_top=False,weights=None)
    model.load_weights("model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5")
    model.layers.pop()
    outputs = model.layers[-1].output
    return outputs # 去掉VGG16的2个1x1卷积

'''
高度是32，vgg完后的channel

'''
def VGG_19(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    # 到第五层就返回了，还是个feature map呢
    # model.add(Flatten())
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


# 焊接vgg和lstm
def vgg_gru(vgg_conv5):
    """ Defining a NMT model """
    # VGG的Conv5，然后按照宽度展开，把H中的数据concat到一起，是model，model的父类也是layer
    # input_shape = (img_width,img_height,channel)
    # [batch,width,height,channel] => [batch,width,height*channel]
    # [samples, time steps, features]
    b = vgg_conv5.shape[0]
    w = vgg_conv5.shape[1]
    h = vgg_conv5.shape[2]
    c = vgg_conv5.shape[3]
    new_c = c*h
    rnn_input = vgg_conv5.reshape((b,w,h*c)) # 转置

    # 1.Encoder GRU编码器
    encoder_gru = Bidirectional(GRU(64,#写死一个隐含神经元数量
                                    return_sequences=True,
                                    return_state=True,
                                    name='encoder_gru'),
                                name='bidirectional_encoder')
    encoder_out, encoder_fwd_state, encoder_back_state = encoder_gru(rnn_input)

    # 2.Decoder GRU,using `encoder_states` as initial state.
    # 使用encoder的输出当做decoder的输入
    decoder_inputs = Input(batch_shape=(b, 5 - 1, 64), name='decoder_inputs')
    decoder_gru = GRU(64*2, return_sequences=True, return_state=True, name='decoder_gru')
    decoder_out, decoder_state = decoder_gru(
        decoder_inputs, initial_state=Concatenate(axis=-1)([encoder_fwd_state, encoder_back_state])
    )

    # Attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_out, decoder_out])

    # concat Attention的输出 + GRU的输出
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

    # Dense layer,
    dense = Dense(64, activation='softmax', name='softmax_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense_time(decoder_concat_input)

    # Full model
    full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    full_model.compile(optimizer='adam', loss='categorical_crossentropy')

    full_model.summary()

    return full_model

def decode_model(vgg_conv5, hidden_size, batch_size, en_timesteps, en_vsize, fr_timesteps, fr_vsize):
    """ Inference model """
    batch_size = 1

    """ Encoder (Inference) model """
    encoder_inf_inputs = Input(batch_shape=(batch_size, en_timesteps, en_vsize), name='encoder_inf_inputs')
    encoder_inf_out, encoder_inf_fwd_state, encoder_inf_back_state = encoder_gru(encoder_inf_inputs)
    encoder_model = Model(inputs=encoder_inf_inputs, outputs=[encoder_inf_out, encoder_inf_fwd_state, encoder_inf_back_state])

    """ Decoder (Inference) model """
    decoder_inf_inputs = Input(batch_shape=(batch_size, 1, fr_vsize), name='decoder_word_inputs')
    encoder_inf_states = Input(batch_shape=(batch_size, en_timesteps, 2*hidden_size), name='encoder_inf_states')
    decoder_init_state = Input(batch_shape=(batch_size, 2*hidden_size), name='decoder_init')

    decoder_inf_out, decoder_inf_state = decoder_gru(
        decoder_inf_inputs, initial_state=decoder_init_state)
    attn_inf_out, attn_inf_states = attn_layer([encoder_inf_states, decoder_inf_out])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_out, attn_inf_out])
    decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)
    decoder_model = Model(inputs=[encoder_inf_states, decoder_init_state, decoder_inf_inputs],
                          outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_state])
    return encoder_model,decoder_model


if __name__ == "__main__":
    im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1)) #转置，0,1,2=> 2,0,1，channel到了最前面
    im = np.expand_dims(im, axis=0)

    # 先来了一个vgg，输出是conv5的:[batch,w=(W/8),h=(H/8),512]，权重是从pretrain中加载的
    # vgg = VGG_19('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
    # vgg_output是去除了2个1x1伪全连接层的Conv5输出，我觉得是一个张量
    vgg_output = vgg_16(im)

    # 把vgg conv5，转化成[batch,time sequence, w],其实就是[batch, h*512, w]，w就是time sequence
    full_model, infer_enc_model, infer_dec_model = vgg_gru(vgg_output)


    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    full_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(im)
    print np.argmax(out)