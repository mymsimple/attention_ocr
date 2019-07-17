from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, GRU, Dense, Concatenate, TimeDistributed, Bidirectional
from keras.layers.wrappers import Bidirectional
from layers.attention import AttentionLayer
from tensorflow.python.keras.utils import to_categorical
import cv2,numpy as np
import tensorflow as tf
from keras.applications.vgg19 import VGG19


def vgg_19():
    input_image = Input(shape=(224,224,3), name='input_image')
    model = VGG19(input_tensor=input_image,include_top=False,weights=None)
    model.load_weights("model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5")
    model.layers.pop()
    conv5_layer = model.layers[-1]
    outputs = conv5_layer.output #<----看！是output，输出，输出的是一个tensor
    print(outputs)
    return input_image,outputs # 去掉VGG16的2个1x1卷积，返回的是一个张量，VGG16是一个Model（也就是Functional）的模型，不是Sequential的




# 焊接vgg和lstm，入参是vgg_conv5返回的张量
def vgg_gru(input_image,vgg_conv5):
    """ Defining a NMT model """
    # VGG的Conv5，然后按照宽度展开，把H中的数据concat到一起，是model，model的父类也是layer
    # input_shape = (img_width,img_height,channel)
    # [batch,width,height,channel] => [batch,width,height*channel]
    # [samples, time steps, features]
    # vgg_conv5_shape = tf.shape(vgg_conv5)
    # vgg_conv5_shape = vgg_conv5.shape.as_list()

    vgg_conv5_shape = [x if x is not None else -1 for x in vgg_conv5.shape.as_list()] # 支持else的写法
    # vgg_conv5_shape = [x  for x in vgg_conv5.shape.as_list() if x is not None]
    # print(vgg_conv5_shape)
    b = vgg_conv5_shape[0]
    w = vgg_conv5_shape[1]
    h = vgg_conv5_shape[2]
    c = vgg_conv5_shape[3]
    print("(b,w,c*h)",(b,w,c*h))
    rnn_input = tf.reshape(vgg_conv5,(b,w,c*h)) # 转置[batch,width,height,channel] => [batch,width,height*channel]
    # print(tf.shape(rnn_input))
    # print(rnn_input)
    # 1.Encoder GRU编码器
    encoder_gru = Bidirectional(GRU(64,#写死一个隐含神经元数量
                                    return_sequences=True,
                                    return_state=True,
                                    name='encoder_gru'),
                                name='bidirectional_encoder')

    encoder_out, encoder_fwd_state, encoder_back_state = encoder_gru(rnn_input)


    # 2.Decoder GRU,using `encoder_states` as initial state.
    # 使用encoder的输出当做decoder的输入
    decoder_inputs = Input(shape=(5 - 1, 64), name='decoder_inputs')
    decoder_gru = GRU(64*2, return_sequences=True, return_state=True, name='decoder_gru')
    decoder_out, decoder_state = decoder_gru(
        decoder_inputs, initial_state=Concatenate(axis=-1)([encoder_fwd_state, encoder_back_state])
    )

    # Attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    print("encoder_out:",encoder_out)
    print("decoder_out:", decoder_out)
    attn_out, attn_states = attn_layer([encoder_out, decoder_out])

    # concat Attention的输出 + GRU的输出
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

    # Dense layer,
    dense = Dense(64, activation='softmax', name='softmax_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense_time(decoder_concat_input)

    # Full model
    full_model = Model(inputs=[input_image, decoder_inputs], outputs=decoder_pred)
    full_model.compile(optimizer='adam', loss='categorical_crossentropy')

    full_model.summary()

    return full_model


def train(full_model, image, label, batch_size, n_epochs=1):
    """ Training the model """

    for ep in range(n_epochs):
        losses = []
        for bi in range(0, label.shape[0] - batch_size, batch_size):

            label_seq = to_categorical(label[bi:bi + batch_size, :], num_classes=10)# 简单点，先来0-9

            full_model.train_on_batch([image, label_seq[:, :-1, :]], label_seq[:, 1:, :])

        if (ep + 1) % 1 == 0:
            print("Loss in epoch {}: {}".format(ep + 1, np.mean(losses)))


if __name__ == "__main__":
    im = cv2.resize(cv2.imread('data/test5.png'), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    # im = im.transpose((2,0,1)) #转置，0,1,2=> 2,0,1，channel到了最前面
    im = np.expand_dims(im, axis=0)
    # print(im.shape)

    # 先来了一个vgg，输出是conv5的:[batch,w=(W/8),h=(H/8),512]，权重是从pretrain中加载的
    # vgg = VGG_19('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
    # vgg_output是去除了2个1x1伪全连接层的Conv5输出，我觉得是一个张量
    input_image,vgg_output = vgg_19()

    # 把vgg conv5，转化成[batch,time sequence, w],其实就是[batch, h*512, w]，w就是time sequence
    model = vgg_gru(input_image,vgg_output)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    train(model,im,np.array(["12345"]))