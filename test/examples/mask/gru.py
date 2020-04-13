# https://stackoverflow.com/questions/54989442/rnn-in-tensorflow-vs-keras-depreciation-of-tf-nn-dynamic-rnn
import numpy as np
from tensorflow.python.keras.layers import GRU,Masking,Input
from tensorflow.python.keras import Model

test_input = np.array([
    [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
    [[2, 2, 2], [2, 2, 2], [0, 0, 0], [0, 0, 0]],
    [[3, 3, 3], [0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=int)
mask = Masking(mask_value=0)
gru = GRU(
    1,
    return_sequences=True,
    activation="sigmoid",
    recurrent_activation="sigmoid",
    kernel_initializer='ones',
    recurrent_initializer='zeros',
    use_bias=True,
    bias_initializer='ones'
)
x = Input(shape=test_input.shape[1:])

# 对比两个结果是不一样的，mask的是后续就不变了，无mask后面还会变
# m1 = Model(inputs=x, outputs=gru(x))
m1 = Model(inputs=x, outputs=gru(mask(x)))

print(m1.predict(test_input).squeeze())


# seq_length = tf.constant(np.array([4,2,1], dtype=int))
# def old_rnn(inputs):
#     rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
#         rnn.cell,
#         inputs,
#         dtype=tf.float32,
#         sequence_length=seq_length
#     )
#     return rnn_outputs
# sess = K.get_session()
# print(sess.run(x,feed_dict={x: test_input}))#.squeeze()