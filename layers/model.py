from tensorflow.python.keras.layers import Bidirectional,Input, GRU, Dense, Concatenate, TimeDistributed
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
import tensorflow as tf
from layers.conv import Conv
from layers.attention import AttentionLayer
import logging
from utils.logger import _p_shape,_p
logger = logging.getLogger("Model")


# y_pred is [batch,seq,charset_size]
# 这个函数的细节，测试了一下，参考：test/test_accuracy.py
# 不过这个是对一个batch的，对于validate中的多个batches，是否还会在多个batches上平均，
# 这个细节就不太了解了....?
# 2020.3.19,y_pred第一维度是batch,:https://stackoverflow.com/questions/46663013/what-is-y-true-and-y-pred-when-creating-a-custom-metric-in-keras
def words_accuracy(y_true, y_pred):
    # logger.debug("DEBUG@@@,看看y_pred的shape:%r",K.int_shape(y_pred))
    # 调试结果是======>(None, None, 3864)
    # 第一个维度是batch，第三个维度是词表长度，那第二个维度呢？
    #
    # y_pred = _p_shape(y_pred,"DEBUG@@@,运行态的时候的words_accuracy的入参y_pred的shape")
    # 运行态的时候的words_accuracy的入参y_pred的shape[2 29 3864]
    # 所以y_pred[batch,seq_len,vocabulary_size]
    # 经调试，没问题

    max_idx_p = tf.argmax(y_pred, axis=2)
    max_idx_l = tf.argmax(y_true, axis=2)
    max_idx_p = _p(max_idx_p,"@@@,预测值")
    max_idx_l = _p(max_idx_l, "@@@,标签值")
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    correct_pred = _p(correct_pred, "@@@,words_accuracy(字对字)")
    _result = tf.map_fn(fn=lambda e: tf.reduce_all(e), elems=correct_pred, dtype=tf.bool)
    _result = _p(_result, "@@@,words_accuracy(词对词)")
    result = tf.reduce_mean(tf.cast(_result, tf.float32))
    result = _p(result, "@@@,words_accuracy正确率")
    return result


def model(conf,args):

    conv,input_image = Conv().build()

    encoder_bi_gru1 = Bidirectional(GRU(conf.GRU_HIDDEN_SIZE,
                                       return_sequences=True,
                                       return_state=True,
                                       name='encoder_gru'),
                                       name='bidirectional_encoder')

    # TODO：想不通如何实现2个bi-GRU堆叠，作罢，先继续，未来再回过头来考虑
    # encoder_bi_gru2 = Bidirectional(GRU(conf.GRU_HIDDEN_SIZE,
    #                                    return_sequences=True,
    #                                    return_state=True,
    #                                    name='encoder_gru'),
    #                                input_shape=( int(conf.INPUT_IMAGE_WIDTH/4) ,512),
    #                                name='bidirectional_encoder')

    encoder_out, encoder_fwd_state, encoder_back_state = encoder_bi_gru1(conv)

    decoder_inputs = Input(shape=(None,conf.CHARSET_SIZE), name='decoder_inputs')
    decoder_gru = GRU(units=conf.GRU_HIDDEN_SIZE*2, return_sequences=True, return_state=True, name='decoder_gru')
    decoder_initial_status = Concatenate(axis=-1)([encoder_fwd_state, encoder_back_state])
    decoder_out, decoder_state = decoder_gru(decoder_inputs,initial_state=decoder_initial_status)

    attn_layer = AttentionLayer(name='attention_layer')
    logger.debug("模型Attention调用的张量[encoder_out, decoder_out]:%r,%r",encoder_out, decoder_out)
    attn_out, attn_states = attn_layer([encoder_out, decoder_out]) # c_outputs, e_outputs

    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])
    dense = Dense(conf.CHARSET_SIZE, activation='softmax', name='softmax_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_prob = dense_time(decoder_concat_input)

    train_model = Model(inputs=[input_image, decoder_inputs], outputs=decoder_prob)
    opt = Adam(lr=args.learning_rate)

    # categorical_crossentropy主要是对多分类的一个损失，但是seq2seq不仅仅是一个结果，而是seq_length个多分类问题，是否还可以用categorical_crossentropy？
    # 这个疑惑在这个例子中看到答案：https://keras.io/examples/lstm_seq2seq/
    # 我猜，keras的代码中应该是做了判断，如果是多个categorical_crossentropy，应该会K.reduce_mean()一下吧。。。
    train_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[words_accuracy])

    train_model.summary()

    # ##################################################################################################
    # 预测用的模型，单独定义！
    # 预测的模型和训练模型不一样，分成2个模型，一个是编码器 encoder model，一个解码器 decoder model
    # ##################################################################################################

    ### encoder model ###
    #
    # infer_input_image = Input(shape=(conf.INPUT_IMAGE_HEIGHT,conf.INPUT_IMAGE_WIDTH,3), name='input_image') #高度固定为32，3通道
    # infer_conv_output = Conv().build(infer_input_image) # 看！复用了 decoder_gru
    # infer_encoder_out, infer_encoder_fwd_state, infer_encoder_back_state = \
    #     encoder_bi_gru(infer_conv_output)
    # infer_encoder_model = Model(inputs=infer_input_image,
    #                             outputs=[infer_encoder_out, infer_encoder_fwd_state, infer_encoder_back_state])
    # infer_encoder_model.summary()
    #
    # ### decoder model ###
    # # 训练的时候，解码器的输出是一口气全部得到，因为这个时候输入是确定的（就是标签），
    # # 解码的时候，解码内容是一个一个单独出来的，无法一口气得到，特别是要做beamSearch的话，还要做动态规划
    # # 这个时候，如果还套用之前的attention，这个时候的decoder的状态就是长度就是1
    #
    # # 解码器的输入，是一个one-hot字符
    # infer_decoder_inputs =     Input(shape=(None,conf.CHARSET_SIZE), name='decoder_inputs')
    # # 编码器的所有状态
    # infer_encoder_out_states = Input(shape=(1,2*conf.GRU_HIDDEN_SIZE), name='encoder_out_states')
    # # 解码器的隐状态输入
    # infer_decoder_init_state = Input(batch_shape=(1,2*conf.GRU_HIDDEN_SIZE), name='decoder_init_state')
    #
    # infer_decoder_out, infer_decoder_state = \
    #     decoder_gru(infer_decoder_inputs, initial_state=infer_decoder_init_state)  # 看！复用了 decoder_gru
    #
    # infer_attn_out, infer_attn_states = \
    #     attn_layer([infer_encoder_out_states, infer_decoder_out]) # 看！复用了attn_layer
    #
    # infer_decoder_concat = Concatenate(axis=-1, name='concat')([infer_decoder_out, infer_attn_out])
    # infer_decoder_pred = TimeDistributed(dense)(infer_decoder_concat) # 看！复用了dense
    # infer_decoder_model = Model(inputs=[infer_decoder_inputs, infer_encoder_out_states,infer_decoder_init_state],
    #                             outputs=[infer_decoder_pred,infer_attn_states,infer_decoder_state])
    # infer_decoder_model.summary()


    return train_model,None,None#infer_decoder_model,infer_encoder_model