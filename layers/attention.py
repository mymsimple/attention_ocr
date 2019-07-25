import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K

def _p(t,name):
    print("调试计算图定义："+name, t)
    return tf.Print(t,[tf.shape(t)],name)

# 实现了经典的attention模式：https://arxiv.org/pdf/1409.0473.pdf
class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)


    # def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
    #     if initial_state is None and constants is None:
    #         return super(AttentionLayer, self).__call__(inputs, **kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        # 注意，encoder_out_seq是一个数组，长度是seq；decoder_out_seq是一个输出。
        encoder_out_seq, decoder_out_seq = inputs

        encoder_out_seq = _p(encoder_out_seq, "编码器隐含层输出")

        # 实现了能量函数，e_tj=V * tanh ( W * h_j + U * S_t-1 + b )
        # inputs,我理解就是所有的h_j，错！我之前理解错了，这个参数是指某个时刻t，对应的输入！不是所有，是某个时刻的输入。
        #        按为什么还有个s，input+s，是因为batch。
        # states,我理解就是S_t-1
        def energy_step(inputs, states): # inputs(batch,dim)
            inputs = _p(inputs,"energy_step:inputs 算能量函数了..........") #inputs：[1，20]


            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            print("en_seq_len, en_hidden:",en_seq_len, en_hidden)
            de_hidden = inputs.shape[-1]

            #  W * h_j
            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
            _p(reshaped_enc_outputs,"reshaped_enc_outputs")
            W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))

            # U * S_t - 1
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim

            # tanh ( W * h_j + U * S_t-1 + b )
            reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))

            # V * tanh ( W * h_j + U * S_t-1 + b )
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
        def context_step(inputs, states): # inputs (batch,dim)
            inputs = _p(inputs,"context_step:inputs")
            states = _p(states,"context_step:states")
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            c_i = _p(c_i,"context_step:c_i,算h的期望，也就是注意力了---------------------\n")
            return c_i, [c_i]

        # (batch_size, enc_seq_len, latent_dim)
        # => (batch_size, hidden_size)
        def create_inital_state(inputs, hidden_size):
            # print("inputs",inputs)
            # print("hidden_size",hidden_size)
            # print("type(hidden_size)", type(hidden_size))
            # We are not using initial states, but need to pass something to K.rnn funciton
            fake_state = K.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim)
            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
            print(fake_state)
            print("------")
            print(tf.shape(fake_state))
            fake_state = K.tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim)
            return fake_state

        # encoder_out_seq = (batch_size, enc_seq_len, latent_dim)
        # fake_state_c ==   (batch_size, latent_dim)
        # fake_state_e ==   (batch_size, enc_seq) , 最后这个维度不好理解，其实就是attention模型里面的decoder对应的每个步骤的attention这个序列，是一个值
        # K.rnn(计算函数，输入x，初始状态）: K.rnn 这个函数，接受三个基本参数，其中第一个参数就是刚才写好的 step_do 函数，第二个参数则是输入的时间序列，第三个是初始态
        # 这个rnn就是解码器，输入 eji=a(s_i-1,hj)，其中j要遍历一遍，这个k.rnn就是把每个hj对应的eij都计算一遍
        # 输出e_outputs，就是一个概率序列


        # eij(i不变,j是一个encoder的h下标），灌入到一个新的rnn中，让他计算出对应的输出，这个才是真正的Decoder！！！
        shape = encoder_out_seq.shape.as_list()
        print("encoder_out_seq.shape:",shape)
        fake_state_e = create_inital_state(encoder_out_seq,shape[1])#encoder_out_seq.shape[1])  # fake_state_e (batch,enc_seq_len)
        fake_state_e = _p(fake_state_e, "fake_state_e")
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )


        e_outputs = _p(e_outputs,"能量函数e输出：：：：")


        fake_state_c = create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])  #
        fake_state_c = _p(fake_state_c, "fake_state_c")
        print("e_outputs:", e_outputs)
        last_out, c_outputs, _ = K.rnn( # context_step算注意力的期望，sum(eij*encoder_out), 输出的(batch,encoder_seq,)
            context_step, e_outputs, [fake_state_c],
        )
        c_outputs = _p(c_outputs,"注意力c输出：：：：")

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]