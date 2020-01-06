# 之前报错：AttributeError: 'Bidirectional' object has no attribute 'outbound_nodes'
# from keras.layers import Bidirectional,Input, GRU, Dense, Concatenate, TimeDistributed,Reshape
# from tensorflow.python.keras.applications.vgg19 import VGG19
# from keras.applications.vgg19 import VGG19
# 原因是不能用keras自带的vgg19+keras自带的bidirectional，靠，肯定是版本不兼容的问题
# 切换到下面的就好了，之前还是试验了用tf的bidirectional+keras的vgg19，也是不行，报错：AttributeError: 'Node' object has no attribute 'output_masks'
# 靠谱的组合是：tf的bidirectional+tf的vgg19
# from tensorflow.keras.layers import Bidirectional,Input, GRU, Dense, Concatenate, TimeDistributed
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

from keras.layers import Bidirectional,Input, GRU, Dense, Concatenate, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam

from layers.conv import Conv
from layers.attention import AttentionLayer
import tensorflow as tf
import logging

logger = logging.getLogger("Model")

conv = Conv()


# 这个函数废弃了，改用tf自带的pad_seqence了
# # seq需要加padding
# # 输入的图像需要加 0 padding
# def padding_wrapper(conv_output,mask_value):
#     paddings = [[0, 0], [0, 50 - tf.shape(conv_output)[0]],[0,0]]
#     # 给卷基层的输出增加padding，让他可以输入到bi-gru里面去
#     conv_output_with_padding = tf.pad(conv_output, paddings=paddings, constant_values=mask_value)
#     #print("conv_output_with_padding.shape")
#     conv_output_with_padding.set_shape([None, 50, 512])  # 靠！还可以这么玩呢！给丫设置一个shape。
#     return conv_output_with_padding


# y_pred is [batch,seq,charset_size]
# 这个函数的细节，测试了一下，参考：test/test_accuracy.py
# 不过这个是对一个batch的，对于validate中的多个batches，是否还会在多个batches上平均，
# 这个细节就不太了解了....?
def words_accuracy(y_true, y_pred):
    max_idx_p = tf.argmax(y_pred, axis=2)
    max_idx_l = tf.argmax(y_true, axis=2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    _result = tf.map_fn(fn=lambda e: tf.reduce_all(e), elems=correct_pred, dtype=tf.bool)
    return tf.reduce_mean(tf.cast(_result, tf.float32))

# 焊接vgg和lstm，入参是vgg_conv5返回的张量
def model(conf,args):

    # 高度和长度都不定，是None，虽然可以定义高度(32,None,3)，但是一般都是从左到右定义None的，所以第一个写32也就没有意义了
    # fix the width & width,give up the mask idea....
    input_image = Input(shape=(conf.INPUT_IMAGE_HEIGHT,conf.INPUT_IMAGE_WIDTH,3), name='input_image') #高度固定为32，3通道

    # input_image = Masking(0.0)(input_image) <----- 哭：卷基层不支持Mask，欲哭无泪：TypeError: Layer block1_conv1 does not support masking, but was passed an input_mask: Tensor("masking/Any_1:0", shape=(?, 32, ?), dtype=bool)
    # 1. 卷基层，输出是conv output is (Batch,Width/32,512)
    conv_output = conv(input_image)

    # 经过padding后，转变为=>(Batch,50,512)
    # conv_output_with_padding = Lambda(padding_wrapper,arguments={'mask_value':conf.MASK_VALUE})(conv_output)
    # conv_output_mask = Masking(conf.MASK_VALUE)(conv_output)

    # 2.Encoder Bi-GRU编码器
    encoder_bi_gru = Bidirectional(GRU(conf.GRU_HIDDEN_SIZE,
                                       return_sequences=True,
                                       return_state=True,
                                       name='encoder_gru'),
                                   input_shape=(conf.INPUT_IMAGE_WIDTH/4,512),
                                   name='bidirectional_encoder')
    encoder_out, encoder_fwd_state, encoder_back_state = encoder_bi_gru(conv_output)

    # 3.Decoder GRU解码器，使用encoder的输出当做输入状态
    decoder_inputs = Input(shape=(None,conf.CHARSET_SIZE), name='decoder_inputs')

    # masked_decoder_inputs = Masking(conf.MASK_VALUE)(decoder_inputs)

    # GRU的units=GRU_HIDDEN_SIZE*2=512，是解码器GRU输出的维度，至于3770是之后，在做一个全连接才可以得到的
    # units指的是多少个隐含神经元，这个数量要和前面接的Bi-LSTM一致(他是512),这样，才可以接受前面的Bi-LSTM的输出作为他的初始状态输入
    decoder_gru = GRU(units=conf.GRU_HIDDEN_SIZE*2, return_sequences=True, return_state=True, name='decoder_gru')
    decoder_out, decoder_state = decoder_gru(decoder_inputs, initial_state=Concatenate(axis=-1)([encoder_fwd_state, encoder_back_state]))

    # 4.Attention layer注意力层
    attn_layer = AttentionLayer(name='attention_layer')

    # attention层的输入是编码器的输出，和，解码器的输出，他俩的输出是一致的，都是512
    # encoder_out shape=(?, 50, 512) , decoder_out shape=(?, ?, 512)
    logger.debug("模型Attention调用的张量[encoder_out, decoder_out]:%r,%r",encoder_out, decoder_out)
    attn_out, attn_states = attn_layer([encoder_out, decoder_out])#[(N,512,seq),(N,3770)]

    # concat Attention的输出 + GRU的输出
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

    # 5.Dense layer output layer 输出层
    dense = Dense(conf.CHARSET_SIZE, activation='softmax', name='softmax_layer')

    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_prob = dense_time(decoder_concat_input)

    # whole model 整个模型
    train_model = Model(inputs=[input_image, decoder_inputs], outputs=decoder_prob)
    opt = Adam(lr=args.learning_rate)

    # categorical_crossentropy主要是对多分类的一个损失，但是seq2seq不仅仅是一个结果，而是seq_length个多分类问题，是否还可以用categorical_crossentropy？
    # 这个疑惑在这个例子中看到答案：https://keras.io/examples/lstm_seq2seq/
    # 我猜，keras的代码中应该是做了判断，如果是多个categorical_crossentropy，应该会tf.reduce_mean()一下吧。。。
    train_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[words_accuracy])

    train_model.summary()

    # ##################################################################################################
    # 预测用的模型，单独定义！
    # 预测的模型和训练模型不一样，分成2个模型，一个是编码器 encoder model，一个解码器 decoder model
    # ##################################################################################################

    ### encoder model ###

    infer_input_image = Input(shape=(conf.INPUT_IMAGE_HEIGHT,conf.INPUT_IMAGE_WIDTH,3), name='input_image') #高度固定为32，3通道
    infer_conv_output = conv(infer_input_image) # 看！复用了 decoder_gru
    infer_encoder_out, infer_encoder_fwd_state, infer_encoder_back_state = \
        encoder_bi_gru(infer_conv_output)
    infer_encoder_model = Model(inputs=infer_input_image,
                                outputs=[infer_encoder_out, infer_encoder_fwd_state, infer_encoder_back_state])
    infer_encoder_model.summary()

    ### decoder model ###

    infer_decoder_inputs =     Input(shape=(conf.INPUT_IMAGE_WIDTH/4,conf.CHARSET_SIZE), name='decoder_inputs')
    infer_encoder_out_states = Input(shape=(1,2*conf.GRU_HIDDEN_SIZE), name='encoder_out_states')
    infer_decoder_init_state = Input(batch_shape=(1,2*conf.GRU_HIDDEN_SIZE), name='decoder_init_state')

    infer_decoder_out, infer_decoder_state = \
        decoder_gru(infer_decoder_inputs, initial_state=infer_decoder_init_state)  # 看！复用了 decoder_gru

    infer_attn_out, infer_attn_states = \
        attn_layer([infer_encoder_out_states, infer_decoder_out]) # 看！复用了attn_layer

    infer_decoder_concat = Concatenate(axis=-1, name='concat')([infer_decoder_out, infer_attn_out])
    infer_decoder_pred = TimeDistributed(dense)(infer_decoder_concat) # 看！复用了dense
    infer_decoder_model = Model(inputs=[infer_decoder_inputs, infer_encoder_out_states,infer_decoder_init_state],
                                outputs=[infer_decoder_pred,infer_attn_states,infer_decoder_state])
    infer_decoder_model.summary()

    return train_model,infer_decoder_model,infer_encoder_model