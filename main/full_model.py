# 之前报错：AttributeError: 'Bidirectional' object has no attribute 'outbound_nodes'
# from keras.layers import Bidirectional,Input, GRU, Dense, Concatenate, TimeDistributed,Reshape
# from tensorflow.python.keras.applications.vgg19 import VGG19
# from keras.applications.vgg19 import VGG19
# 原因是不能用keras自带的vgg19+keras自带的bidirectional，靠，肯定是版本不兼容的问题
# 切换到下面的就好了，之前还是试验了用tf的bidirectional+keras的vgg19，也是不行，报错：AttributeError: 'Node' object has no attribute 'output_masks'
# 靠谱的组合是：tf的bidirectional+tf的vgg19
from tensorflow.python.keras.layers import Bidirectional,Input, GRU, Dense, Concatenate, TimeDistributed
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import to_categorical
from layers import conv
from layers.attention import AttentionLayer
import numpy as np

# 焊接vgg和lstm，入参是vgg_conv5返回的张量
def model():

    input_image = Input(shape=(32,100,3), name='input_image') #高度固定为32，3通道

    # 1. 卷基层
    conv_output = conv.conv_layer(input_image)

    # 2.Encoder Bi-GRU编码器
    encoder_gru = Bidirectional(GRU(64,return_sequences=True,return_state=True,name='encoder_gru'),name='bidirectional_encoder')
    encoder_out, encoder_fwd_state, encoder_back_state = encoder_gru(conv_output)

    # 3.Decoder GRU解码器，使用encoder的输出当做输入状态
    decoder_inputs = Input(shape=(5,64), name='decoder_inputs')
    decoder_gru = GRU(64*2, return_sequences=True, return_state=True, name='decoder_gru')
    decoder_out, decoder_state = decoder_gru(decoder_inputs, initial_state=Concatenate(axis=-1)([encoder_fwd_state, encoder_back_state]))

    # 4.Attention layer注意力层
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_out, decoder_out])

    # concat Attention的输出 + GRU的输出
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

    # 5.Dense layer输出层
    dense = Dense(64, activation='softmax', name='softmax_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense_time(decoder_concat_input)

    #整个模型
    model = Model(inputs=[input_image, decoder_inputs], outputs=decoder_pred)
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    model.summary()

    return model


def train(full_model, image, label, batch_size, n_epochs=1):
    """ Training the model """

    for ep in range(n_epochs):
        losses = []
        for bi in range(0, label.shape[0] - batch_size, batch_size):

            label_seq = to_categorical(label[bi:bi + batch_size, :], num_classes=10)# 简单点，先来0-9

            full_model.train_on_batch([image, label_seq[:, :-1, :]], label_seq[:, 1:, :])

        if (ep + 1) % 1 == 0:
            print("Loss in epoch {}: {}".format(ep + 1, np.mean(losses)))