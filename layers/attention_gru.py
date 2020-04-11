from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import Layer
from keras.layers import RNN
import logging
from utils.logger import _p
logger = logging.getLogger(__name__)

'''
这个是我自己亲手实现的attention，是拷贝自keras库的 keras.layers.recurrent中的GRU和GRUCell
并参考了./depreated_attentions中的实现，严格尊崇了注意力的Bahdanau注意力，
此代码仅实现Bahdanau注意力，Bahdanau注意力实现更复杂，必须要使用自定义的GRU，
而Luong注意力实现相对简单，可以参考我的master分支，实现的就是Lusong注意力，

一些trick：
- 解码器的输入是contact[注意力,识别字符(1-hot)]，但是Keras的输入不能是数组，最后使用了Keras的[constants](https://keras.io/layers/recurrent/)机制
- 因为输入是concat的，所以内部的输入参数维度dim_input计算都要调整为：字符集数 + 解码器的units
- 注意力仅使用了W和V，注意细节，V需要是[10241]的 
'''
def _generate_dropout_mask(ones, rate, training=None, count=1):
    def dropped_inputs():
        return K.dropout(ones, rate)

    if count > 1:
        return [K.in_train_phase(
            dropped_inputs,
            ones,
            training=training) for _ in range(count)]
    return K.in_train_phase(
        dropped_inputs,
        ones,
        training=training)

# GRUCell结构：https://upload-images.jianshu.io/upload_images/6983308-f3d8a02ed1b8b24d.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp
class AttentionGRUCell(Layer):
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 reset_after=False,
                 **kwargs):
        super(AttentionGRUCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # 为注意力建立一套初始化、正则化和约束参数
        self.attention_initializer = initializers.get('glorot_uniform')
        self.attention_regularizer = regularizers.get(None)
        self.attention_constraint = constraints.get(None)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.reset_after = reset_after
        self.state_size = self.units
        self.output_size = self.units + 64 # TODO 解码器的Cell输出是512+64维度，回头修改成传入
        self._dropout_mask = None
        self._recurrent_dropout_mask = None


    def build(self, input_shape):

        # ######## 修改 ##########
        # 内部参数维度计算
        if type(input_shape) == list:
            # Decoder inputs:[None,3864] , Encoder outputs:[None,64,512]
            input_dim = sum(shape[-1] for shape in input_shape)
            encoder_dim = input_shape[1][-1]
        else:
            input_dim = input_shape[-1]
            encoder_dim = 0
        # 注意力参数维度计算
        attention_dim = encoder_dim + self.units
        logger.debug("注意力参数Wa的隐层维度：%d",attention_dim)

        self.kernel = self.add_weight(shape=(input_dim, self.units * 3),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if not self.reset_after:
                bias_shape = (3 * self.units,)
            else:
                # separate biases for input and recurrent kernels
                # Note: the shape is intentionally different from CuDNNGRU biases
                # `(2 * 3 * self.units,)`, so that we can distinguish the classes
                # when loading and converting saved weights.
                bias_shape = (2, 3 * self.units)
            self.bias = self.add_weight(shape=bias_shape,
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            if not self.reset_after:
                self.input_bias, self.recurrent_bias = self.bias, None
            else:
                # NOTE: need to flatten, since slicing in CNTK gives 2D array
                self.input_bias = K.flatten(self.bias[0])
                self.recurrent_bias = K.flatten(self.bias[1])
        else:
            self.bias = None

        # ######## 修改 ##########
        # 注意力参数,2个：Wa，Va，参考：https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%7B%5Ccolor%7BRed%7D+%7Bc_t%7D%7D+%26%3D%5Csum+%5ET_%7Bi%3D1%7D+%5Calpha_%7Bti%7Dh_i%5C%5C+%5Calpha_%7Bti%7D+%26%3D%5Cfrac%7Bexp%28e_%7Bti%7D%29%7D%7B%5Csum%5ET_%7Bk%3D1%7Dexp%28e_%7Btk%7D%29%7D%5C%5C+e_%7Bti%7D+%26%3Dv_a%5E%7B%5Ctop%7Dtanh%28W_a%5Bs_%7Bi-1%7D%2Ch_i%5D%29%5C%5C+%5Cend%7Baligned%7D%5C%5C
        self.Wa = self.add_weight(shape=(attention_dim, self.units),
                                      name='attention_W',
                                      initializer=self.attention_initializer,
                                      regularizer=self.attention_regularizer,
                                      constraint=self.attention_constraint)

        self.Va = self.add_weight(shape=(self.units,1), # 细节：第一个shape是1，否则softmax无法计算
                                      name='attention_V',
                                      initializer=self.attention_initializer,
                                      regularizer=self.attention_regularizer,
                                      constraint=self.attention_constraint)

        # update gate
        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
        # reset gate
        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:,
                                                        self.units:
                                                        self.units * 2]
        # new gate
        self.kernel_h = self.kernel[:, self.units * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:]

        if self.use_bias:
            # bias for inputs
            self.input_bias_z = self.input_bias[:self.units]
            self.input_bias_r = self.input_bias[self.units: self.units * 2]
            self.input_bias_h = self.input_bias[self.units * 2:]
            # bias for hidden state - just for compatibility with CuDNN
            if self.reset_after:
                self.recurrent_bias_z = self.recurrent_bias[:self.units]
                self.recurrent_bias_r = (
                    self.recurrent_bias[self.units: self.units * 2])
                self.recurrent_bias_h = self.recurrent_bias[self.units * 2:]
        else:
            self.input_bias_z = None
            self.input_bias_r = None
            self.input_bias_h = None
            if self.reset_after:
                self.recurrent_bias_z = None
                self.recurrent_bias_r = None
                self.recurrent_bias_h = None
        self.built = True


    # ######## 修改 ##########
    # 参考：https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%7B%5Ccolor%7BRed%7D+%7Bc_t%7D%7D+%26%3D%5Csum+%5ET_%7Bi%3D1%7D+%5Calpha_%7Bti%7Dh_i%5C%5C+%5Calpha_%7Bti%7D+%26%3D%5Cfrac%7Bexp%28e_%7Bti%7D%29%7D%7B%5Csum%5ET_%7Bk%3D1%7Dexp%28e_%7Btk%7D%29%7D%5C%5C+e_%7Bti%7D+%26%3Dv_a%5E%7B%5Ctop%7Dtanh%28W_a%5Bs_%7Bi-1%7D%2Ch_i%5D%29%5C%5C+%5Cend%7Baligned%7D%5C%5C
    def compute_attention(self, decoder_state,encoder_states):
        if (type(encoder_states)==list or type(encoder_states)==tuple) and len(encoder_states)==1:
            encoder_states = encoder_states[0]

        repeated_decoder_states = K.repeat(decoder_state, K.int_shape(encoder_states)[1])

        # if(self.use_bias): Was=K.bias_add(Was, self.Wa_bias)
        # 把编码器输出和解码器的repeat(seq)次的向量，concat到一起，严格遵照"参考"的实现
        concat_states = K.tanh(K.concatenate([repeated_decoder_states,encoder_states],axis=-1))

        _tanh_result = K.tanh(K.dot(concat_states, self.Wa))

        eij = K.dot(_tanh_result,self.Va)

        alpha_ij=K.softmax(eij, axis=1)

        c_t = K.batch_dot(alpha_ij, encoder_states, axes=1)

        c_t = K.squeeze(c_t,1) # ？？？？为何要squeeze
        alpha_ij = K.squeeze(alpha_ij,axis=-1)

        return c_t,alpha_ij # 返回注意力，以及注意力分布

    def call(self, inputs, states, training=None,constants=None):
        h_tm1 = states[0]  # previous memory

        # ######## 修改 ##########
        h_tm1 = _p(h_tm1,"解码器中间状态h_tm1")
        c_t,alpha_ij = self.compute_attention(h_tm1,constants) # 计算这一时刻的注意力
        c_t = _p(c_t,"注意力向量")
        alpha_ij = _p(alpha_ij, "注意力分布")

        inputs = K.concatenate([inputs,c_t],axis=-1) # 原始输入 concat上 注意力 作为总的输入

        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=3)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(h_tm1),
                self.recurrent_dropout,
                training=training,
                count=3)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        if 0. < self.dropout < 1.:
            inputs_z = inputs * dp_mask[0]
            inputs_r = inputs * dp_mask[1]
            inputs_h = inputs * dp_mask[2]
        else:
            inputs_z = inputs
            inputs_r = inputs
            inputs_h = inputs

        x_z = K.dot(inputs_z, self.kernel_z)
        x_r = K.dot(inputs_r, self.kernel_r)
        x_h = K.dot(inputs_h, self.kernel_h)
        if self.use_bias:
            x_z = K.bias_add(x_z, self.input_bias_z)
            x_r = K.bias_add(x_r, self.input_bias_r)
            x_h = K.bias_add(x_h, self.input_bias_h)

        if 0. < self.recurrent_dropout < 1.:
            h_tm1_z = h_tm1 * rec_dp_mask[0]
            h_tm1_r = h_tm1 * rec_dp_mask[1]
            h_tm1_h = h_tm1 * rec_dp_mask[2]
        else:
            h_tm1_z = h_tm1
            h_tm1_r = h_tm1
            h_tm1_h = h_tm1

        recurrent_z = K.dot(h_tm1_z, self.recurrent_kernel_z)
        recurrent_r = K.dot(h_tm1_r, self.recurrent_kernel_r)
        if self.reset_after and self.use_bias:
            recurrent_z = K.bias_add(recurrent_z, self.recurrent_bias_z)
            recurrent_r = K.bias_add(recurrent_r, self.recurrent_bias_r)

        z = self.recurrent_activation(x_z + recurrent_z)
        r = self.recurrent_activation(x_r + recurrent_r)

        # reset gate applied after/before matrix multiplication
        if self.reset_after:
            recurrent_h = K.dot(h_tm1_h, self.recurrent_kernel_h)
            if self.use_bias:
                recurrent_h = K.bias_add(recurrent_h, self.recurrent_bias_h)
            recurrent_h = r * recurrent_h
        else:
            recurrent_h = K.dot(r * h_tm1_h, self.recurrent_kernel_h)

        hh = self.activation(x_h + recurrent_h)

        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh

        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True

        logger.debug("AttenCell.alpha_ij.shape:%r",alpha_ij.shape)
        logger.debug("AttenCell.h.shape:%r", h.shape)

        alpha_ij = _p(alpha_ij,"注意力")
        h = _p(h, "解码器状态h")
        return K.concatenate([alpha_ij,h],axis=-1),[h]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation':
                      activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'reset_after': self.reset_after}
        base_config = super(AttentionGRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttentionGRU(RNN):
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 reset_after=False,
                 **kwargs):


        cell = AttentionGRUCell(units,
                       activation=activation,
                       recurrent_activation=recurrent_activation,
                       use_bias=use_bias,
                       kernel_initializer=kernel_initializer,
                       recurrent_initializer=recurrent_initializer,
                       bias_initializer=bias_initializer,
                       kernel_regularizer=kernel_regularizer,
                       recurrent_regularizer=recurrent_regularizer,
                       bias_regularizer=bias_regularizer,
                       kernel_constraint=kernel_constraint,
                       recurrent_constraint=recurrent_constraint,
                       bias_constraint=bias_constraint,
                       dropout=dropout,
                       recurrent_dropout=recurrent_dropout,
                       reset_after=reset_after)
        super(AttentionGRU, self).__init__(cell,
                                  return_sequences=return_sequences,
                                  return_state=return_state,
                                  go_backwards=go_backwards,
                                  stateful=stateful,
                                  unroll=unroll,
                                  **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)


    def call(self, inputs, mask=None, training=None, initial_state=None,constants=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None

        return super(AttentionGRU, self).call(inputs,
                                     mask=mask,
                                     training=training,
                                     initial_state=initial_state,
                                     constants=constants)
    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def reset_after(self):
        return self.cell.reset_after

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation':
                      activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer':
                      regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'reset_after': self.reset_after}
        base_config = super(AttentionGRU, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)