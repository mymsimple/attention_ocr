import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.python.keras.backend import tile

from utils.logger import _p_shape,_p
import logging

logger = logging.getLogger(__name__)

# 实现了经典的attention模式：https://arxiv.org/pdf/1409.0473.pdf
class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        if initial_state is None and constants is None:
            return super(AttentionLayer, self).__call__(inputs, **kwargs)

    # input_shape是输入的张量的shape，他应该和你的隐含层计算后，输出
    # attn_out, attn_states = attn_layer([encoder_out, decoder_out])#[(N,512,seq),(N,3770)]
    def build(self, input_shape): # input_shape = [(N,512,Seq),(N,3770,Seq)]
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        logger.debug("Attention build时候，input_shape为:%r",input_shape)
        # input_shape:[TensorShape([Dimension(None), Dimension(50),   Dimension(512)]),
        #              TensorShape([Dimension(None), Dimension(None), Dimension(512)])]

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a', # [512,1]
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)
        logger.debug("Attention build时候，W shape为:%r", self.W_a.shape)
        logger.debug("Attention build时候，U shape为:%r", self.U_a.shape)
        logger.debug("Attention build时候，V shape为:%r", self.V_a.shape)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False,mask=None):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        # 注意，encoder_out_seq是一个数组，长度是seq；decoder_out_seq是一个输出。
        encoder_out_seq, decoder_out_seq = inputs

        encoder_out_seq = _p_shape(encoder_out_seq, "注意力调用：入参编码器输出序列：encoder_out_seq")
        decoder_out_seq = _p_shape(decoder_out_seq, "注意力调用：入参解码器输出序列：decoder_out_seq")

        # 实现了能量函数，e_tj=V * tanh ( W * h_j + U * S_t-1 + b )
        # inputs,我理解就是所有的h_j，错！我之前理解错了，这个参数是指某个时刻t，对应的输入！不是所有，是某个时刻的输入。
        #        按为什么还有个s，input+s，是因为batch。
        # states,我理解就是S_t-1
        # decode_outs是不包含seq的，不是一个decode_out_seq，而是decode_out，为何加s呢，是因为batch
        # 但是每一步都是encoder_out_seq全都参与运算的，
        # decoder_out一个和encoder_out_seq一串，对
        def energy_step(decode_outs, states): # decode_outs(batch,dim)
            decode_outs = _p(decode_outs,"energy_step:decode_outs 算能量函数了..........") #decode_outs：[1，20]
            # decoder_seq [b,30,512]
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2] # 30, 512
            de_hidden = decode_outs.shape[-1]
            #  W * h_j
            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden)) #[b,64,512]=> [b*64,512]
            _p(reshaped_enc_outputs,"reshaped_enc_outputs")

            # W_a[512x512],reshaped_enc_outputs[b*64,512] => [b*64,512] => [b,64,512]
            W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
            # U * S_t - 1,decode_outs[b,512],U_a[512,512] => [b,512]    => [b,1,512]
            U_a_dot_h = K.expand_dims(K.dot(decode_outs, self.U_a), axis=1)  # <= batch_size, 1, latent_dim

            # 这个细节很变态，其实就是完成了decoder的输出复制time(64)个，和encoder的输出【64,512】，相加的过程

            # tanh ( W * h_j + U * S_t-1 + b ),[b,64,512] = [b*64,512]
            reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
            # V * tanh ( W * h_j + U * S_t-1 + b ), [b*64,512]*[512,1] => [b*64,1] => [b,64]
            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))
            # softmax(e_tj)
            e_i = K.softmax(e_i)
            e_i = _p(e_i,"energy_step:e_i")
            return e_i, [e_i]

        # 这个step函数有意思，特别要关注他的入参：
        # encoder_out_seq: 编码器的各个time sequence的输出h_i，[batch,ts,dim]
        # states:
        # inputs：某个时刻，这个rnn的输入，这里，恰好是之前那个能量函数eij对应这个时刻的概率
        # ----------------------------
        # "step_do 这个函数，这个函数接受两个输入：step_in 和 states。
        #   其中 step_in 是一个 (batch_size, input_dim) 的张量，
        #   代表当前时刻的样本 xt，而 states 是一个 list，代表 yt−1 及一些中间变量。"
        # e 是30次中的一次，他是一个64维度的概率向量
        def context_step(e, states): # e (batch,dim),其实每个输入就是一个e
            e = _p(e,"context_step:e")
            states = _p(states,"context_step:states")
            # encoder_out_seq[b,64,512] * e[64,1]
            # dot是矩阵相乘，*是对应位置元素相乘
            # [b,64,512] * e[64,1]shape不一样，居然也可以乘，我试了，没问题
            # 其实，就是实现了encoder ouput根据softmax概率分布，加权求和
            c_i = K.sum(encoder_out_seq * K.expand_dims(e, -1), axis=1)
            c_i = _p(c_i,"context_step:c_i,算h的期望，也就是注意力了---------------------\n")
            return c_i, [c_i]

        #    (batch_size, enc_seq_len, latent_dim) (b,64,512)
        # => (batch_size, hidden_size)
        # 这个函数是，作为GRU的初始状态值，
        def create_inital_state(inputs, hidden_size):# hidden_size=64
            # print("inputs",inputs)
            # print("hidden_size",hidden_size)
            # print("type(hidden_size)", type(hidden_size))
            # We are not using initial states, but need to pass something to K.rnn funciton
            fake_state = K.zeros_like(inputs)  # [b,64,512]<= (batch_size, enc_seq_len, latent_dim)
            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
            # print(fake_state)
            # print("------")
            # print(tf.shape(fake_state))
            # print("hidden_size:",hidden_size)

            fake_state = tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim) (b,latent_dim=64)
            return fake_state

        # encoder_out_seq = (batch_size, enc_seq_len, latent_dim)
        # fake_state_c ==   (batch_size, latent_dim)
        # fake_state_e ==   (batch_size, enc_seq) , 最后这个维度不好理解，其实就是attention模型里面的decoder对应的每个步骤的attention这个序列，是一个值
        # K.rnn(计算函数，输入x，初始状态）: K.rnn 这个函数，接受三个基本参数，其中第一个参数就是刚才写好的 step_do 函数，第二个参数则是输入的时间序列，第三个是初始态
        # 这个rnn就是解码器，输入 eji=a(s_i-1,hj)，其中j要遍历一遍，这个k.rnn就是把每个hj对应的eij都计算一遍
        # 输出e_outputs，就是一个概率序列

        # eij(i不变,j是一个encoder的h下标），灌入到一个新的rnn中，让他计算出对应的输出，这个才是真正的Decoder！！！
        shape = encoder_out_seq.shape.as_list()
        # print("encoder_out_seq.shape:",shape)
        # shape[1]是seq=64，序列长度
        fake_state_e = create_inital_state(encoder_out_seq,shape[1])# encoder_out_seq.shape[1]) ， fake_state_e (batch,enc_seq_len)
        fake_state_e = _p_shape(fake_state_e, "fake_state_e")

        # 输出是一个e的序列，是对一个时刻而言的
        ########### ########### ########### K.rnn|K.rnn|K.rnn|K.rnn|K.rnn|K.rnn|K.rnn|K.rnn|K.rnn|K.rnn|K.rnn|K.rnn|K.rnn|
        # 这个步骤是做了30次(decoder，也就是字符串长度），每次得到一个64维度（encoder的time_sequence）的概率向量
        last_out, e_outputs, _ = K.rnn(
            step_function=energy_step,
            inputs=decoder_out_seq,
            initial_states=[fake_state_e],# (bx64)decoder_out_seq是一个序列，不是一个单个值
        )
        # e_outputs [30,64]

        e_outputs = _p_shape(e_outputs,"能量函数e输出：：：：")
        # shape[-1]是encoder的隐含层
        fake_state_c = create_inital_state(encoder_out_seq,encoder_out_seq.shape[-1])  #
        fake_state_c = _p_shape(fake_state_c, "fake_state_c")
        # print("e_outputs:", e_outputs)

        ########### ########### ########### K.rnn|K.rnn|K.rnn|K.rnn|K.rnn|K.rnn|K.rnn|K.rnn|K.rnn|K.rnn|K.rnn|K.rnn|K.rnn|
        last_out, c_outputs, _ = K.rnn( # context_step算注意力的期望，sum(eij*encoder_out), 输出的(batch,encoder_seq,)
            step_function=context_step,
            inputs=e_outputs,
            initial_states=[fake_state_c],
        )
        #c_outputs [b,64,512]
        c_outputs = _p_shape(c_outputs,"注意力c输出：：：：")

        # 输出的是注意力的向量(batch,图像seq,)，
        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]